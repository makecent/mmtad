# Adapted from ActionFormer https://github.com/happyharrycn/actionformer_release/blob/main/libs/modeling/meta_archs.py
# Up to commit b91ef51 on May 15 2023.
# We break down the ActionFormer to a backbone, and fpn neck, a detection head, and a post-processing module.
# The backbone is defined as a ActionFormer neck. The FPN neck is deprecated as it's almost a nn.Identity layer.
# The head is defined as ActionFormer head, as in this file. The post-processing module is defined in the dataset.
import math
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
from mmcv.cnn import Scale
from mmcv.ops import batched_nms
from mmdet.models.dense_heads import AnchorFreeHead
from mmdet.models.utils import multi_apply
# from my_modules.nms1d import batched_nms
from mmdet.models.utils import unpack_gt_instances
from mmdet.registry import MODELS
from mmdet.structures import SampleList
from mmdet.structures.bbox import bbox_overlaps
from mmdet.structures.bbox import (get_box_tensor, get_box_wh,
                                   scale_boxes)
from mmdet.utils import (ConfigType, OptInstanceList, reduce_mean)
from mmdet.utils import InstanceList
from mmengine.config import ConfigDict
from mmengine.structures import InstanceData
from torch import Tensor
from torch.nn import functional as F
from mmdet.models.task_modules.prior_generators import MlvlPointGenerator
from my_modules.layers.actionformer_layers import MaskedConv1d, MaskedConv1dModule

INF = 1e8


@MODELS.register_module('ActionFormerHead')
class ActionFormerHead(AnchorFreeHead):
    """
        Transformer based model for single stage action localization
    Args:
        in_channels: feature dim on FPN
        feat_channels: feature dim for head
        regression_range: regression range on each level of FPN
        num_classes: number of action classes
        strides: strides of the input features
    """

    def __init__(
            self,
            num_classes,
            in_channels=512,
            feat_channels=512,  # (256, 384, 512, 768, 1024) are used for different configs
            stacked_convs: int = 2,
            strides=(1, 2, 4, 8, 16, 32),  # strides of the input features
            regress_ranges=((0, 4), (4, 8), (8, 16), (16, 32), (32, 64), (64, 10000)),
            loss_cls: ConfigType = dict(type='FocalLoss', use_sigmoid=True, loss_weight=1.0, reduction='none'),
            loss_bbox: ConfigType = dict(type='DIoU1dLoss', loss_weight=1.0, reduction='none'),
            bbox_coder: ConfigType = dict(type='DistancePointBBox1dCoder'),
            label_smoothing=0.0,  # (0.0, 0.1) are used for different configs
            center_sampling=True,
            center_sampling_radius=1.5,
            prior_prob=0.01,
            init_loss_norm=100.0,
            empty_cls=(),
            test_cfg=None,
            **kwargs):
        # as we use (1, t) to mimic the (h, w) the strides on height dimension are fixed as 1.
        # note that the format of strides used in mmdet is (w, h), not (h, w).
        strides = tuple((x, 1) for x in strides)
        super().__init__(num_classes=num_classes,
                         in_channels=in_channels,
                         feat_channels=feat_channels,
                         stacked_convs=stacked_convs,
                         strides=strides,
                         loss_cls=loss_cls,
                         loss_bbox=loss_bbox,
                         bbox_coder=bbox_coder,
                         test_cfg=test_cfg,
                         **kwargs)

        assert self.loss_cls.reduction == 'none', "reduction should be 'none' to use loss_normalizer"
        assert self.loss_bbox.reduction == 'none', "reduction should be 'none' to use loss_normalizer"
        assert self.use_sigmoid_cls, "we should use binary classification here"
        self.regress_ranges = regress_ranges
        self.label_smoothing = label_smoothing
        self.center_sampling = center_sampling
        self.center_sample_radius = center_sampling_radius
        self.prior_prob = prior_prob
        self.empty_cls = empty_cls
        self.scales = nn.ModuleList([Scale(1.0) for _ in self.strides])

        # maintain an EMA of #foreground to stabilize the loss normalizer
        # useful for small mini-batch training
        self.init_loss_norm = init_loss_norm
        self.loss_normalizer = init_loss_norm
        self.loss_normalizer_momentum = 0.9

    def _init_cls_convs(self) -> None:
        """Initialize classification conv layers of the head."""
        self.cls_convs = nn.ModuleList()
        for i in range(self.stacked_convs):
            chn = self.in_channels if i == 0 else self.feat_channels
            self.cls_convs.append(
                MaskedConv1dModule(chn, self.feat_channels, 3, stride=1, padding=1, norm_cfg=dict(type='tLN')))

    def _init_reg_convs(self) -> None:
        """Initialize bbox regression conv layers of the head."""
        self.reg_convs = nn.ModuleList()
        for i in range(self.stacked_convs):
            chn = self.in_channels if i == 0 else self.feat_channels
            self.reg_convs.append(
                MaskedConv1dModule(chn, self.feat_channels, 3, stride=1, padding=1, norm_cfg=dict(type='tLN')))

    def _init_predictor(self) -> None:
        """Initialize predictor layers of the head."""
        self.conv_cls = MaskedConv1d(
            self.feat_channels, self.cls_out_channels, 3, padding=1)
        self.conv_reg = MaskedConv1d(self.feat_channels, 2, 3, padding=1)

    def init_weights(self) -> None:
        """Initialize the weights."""
        super().init_weights()
        # use prior in model initialization to improve stability
        # this will overwrite other weight init
        if self.prior_prob > 0:
            bias_value = -(math.log((1 - self.prior_prob) / self.prior_prob))
            torch.nn.init.constant_(self.conv_cls.bias, bias_value)

        # a quick fix to empty categories:
        # the weights assocaited with these categories will remain unchanged
        # we set their bias to a large negative value to prevent their outputs
        if len(self.empty_cls) > 0:
            bias_value = -(math.log((1 - 1e-6) / 1e-6))
            for idx in self.empty_cls:
                torch.nn.init.constant_(self.conv_cls.bias[idx], bias_value)

    def forward(self, inputs) -> Tuple[List[Tensor], List[Tensor]]:
        # %% pass modules
        mlvl_feats, mlvl_masks = inputs
        cls_scores, bbox_offsets = [], []
        for feats, masks, scale, stride in zip(mlvl_feats, mlvl_masks, self.scales, self.strides):
            cls_feat = feats
            reg_feat = feats
            for cls_layer in self.cls_convs:
                cls_feat, _ = cls_layer(cls_feat, masks)
            cls_score, _ = self.conv_cls(cls_feat, masks)

            for reg_layer in self.reg_convs:
                reg_feat, _ = reg_layer(reg_feat, masks)
            bbox_offset, _ = self.conv_reg(reg_feat, masks)
        # %% rescale the bbox_offset
            bbox_offset = scale(bbox_offset).float().clamp(min=0)
            if not self.training:
                # as we use (1, t) to mimic the (h, w), the strides on width dimension are used for scaling
                # note that the format of strides used in mmdet is (w, h), not (h, w).
                bbox_offset *= stride[0]
        # %% filter out invalid positions
            # logits of padded positions (masks = 0) are set to -20, making their scores close to 0 after sigmoid
            # during prediction, these prediction on padded positions will be filtered out by score thresholding
            cls_score = cls_score * masks - 20 * ~masks
        # %% unsqueeze [N, C, T] to mimic [N C H=1 W=T] and pack
            cls_score = cls_score.unsqueeze(2)
            bbox_offset = bbox_offset.unsqueeze(2)
            # (left, right) to (left, top=0, right, bottom=0)
            zeros = torch.zeros_like(bbox_offset)[:, :1, :]
            bbox_offset = torch.cat([bbox_offset[:, :1, :], zeros, bbox_offset[:, 1:, :], zeros], dim=1)
            cls_scores.append(cls_score)
            bbox_offsets.append(bbox_offset)
        return cls_scores, bbox_offsets

    def loss(self, inputs, batch_data_samples: SampleList) -> dict:
        # %% forward modules and prepare loss input
        mlvl_feats, mlvl_masks = inputs
        outs = self.forward(inputs)

        outputs = unpack_gt_instances(batch_data_samples)
        (batch_gt_instances, batch_gt_instances_ignore,
         batch_img_metas) = outputs

        loss_inputs = outs + (batch_gt_instances, batch_img_metas,
                              batch_gt_instances_ignore)
        # %% compute losses
        losses = self.loss_by_feat(*loss_inputs)

        # %% masked out loss for invalid positions (which typically come from padding)
        # we do not mask localization loss because itself naturally ignores invalid positions.
        # specifically, localization losses are computed on only positive positions (positions inside gt bboxes)
        # while invalid (padded) positions should not be determined as positive if nothing went wrong.
        loss_cls, loss_bbox = losses['loss_cls'], losses['loss_bbox']
        flatten_masks = torch.cat([mask.reshape(-1) for mask in mlvl_masks])
        invalid_loss = loss_cls[~flatten_masks]
        valid_loss = loss_cls[flatten_masks]
        loss_cls = valid_loss.sum() + invalid_loss.sum() * 0

        # %% normalize the losses with the number of positive samples (maintained in EMA-style)
        losses['loss_cls'] = loss_cls / self.loss_normalizer
        losses['loss_bbox'] = loss_bbox.sum() / self.loss_normalizer
        return losses

    def loss_by_feat(self,
                     cls_scores: List[Tensor],
                     bbox_offsets: List[Tensor],
                     batch_gt_instances: InstanceList,
                     batch_img_metas: List[dict],
                     batch_gt_instances_ignore: OptInstanceList = None):
        # %% get classification and localization targets of points
        assert len(cls_scores) == len(bbox_offsets)

        featmap_sizes = [featmap.size()[-2:] for featmap in cls_scores]
        all_level_points = self.prior_generator.grid_priors(
            featmap_sizes,
            dtype=bbox_offsets[0].dtype,
            device=bbox_offsets[0].device)

        labels, bbox_targets = self.get_targets(all_level_points, batch_gt_instances)

        # %% flatten cls_scores, bbox_preds
        # cls_score [N, C, H, W] to [NxHxW, C]; bbox_pred [N, 4, H, W] to [NxHxW, 4]
        # mask [N, H, W] to [NxHxW] (H=1 and W=T for TAD)

        num_imgs = cls_scores[0].size(0)
        flatten_cls_scores = [
            cls_score.permute(0, 2, 3, 1).reshape(-1, self.cls_out_channels)
            for cls_score in cls_scores
        ]
        flatten_bbox_offsets = [
            bbox_pred.permute(0, 2, 3, 1).reshape(-1, 4)
            for bbox_pred in bbox_offsets
        ]

        flatten_cls_scores = torch.cat(flatten_cls_scores)
        flatten_bbox_offsets = torch.cat(flatten_bbox_offsets)
        flatten_labels = torch.cat(labels)
        flatten_bbox_targets = torch.cat(bbox_targets)
        # repeat points to align with bbox_preds
        flatten_points = torch.cat([points.repeat(num_imgs, 1) for points in all_level_points])

        # %% Compute positive mask and update loss normalizer
        losses = dict()

        pos_mask = flatten_labels.sum(-1) > 0  # the labels are one-hot format (soft)
        pos_inds = pos_mask.nonzero().reshape(-1)
        num_pos = torch.tensor(len(pos_inds)).type_as(flatten_cls_scores)
        num_pos = max(reduce_mean(num_pos), 1.0)

        # update the loss normalizer
        self.loss_normalizer = self.loss_normalizer_momentum * self.loss_normalizer + (
                1 - self.loss_normalizer_momentum) * num_pos

        # %% Compute classification loss
        # optional label smoothing
        flatten_labels *= (1 - self.label_smoothing)
        flatten_labels += self.label_smoothing / (self.num_classes + 1)

        loss_cls = self.loss_cls(flatten_cls_scores, flatten_labels)

        # %% Compute localization loss
        # For localization, only positive points are considered
        pos_bbox_offsets = flatten_bbox_offsets[pos_inds]
        pos_bbox_targets = flatten_bbox_targets[pos_inds]

        if len(pos_inds) > 0:
            # TODO: check if this decode process is necessary as it only add same constants to pred and gt offsets
            pos_points = flatten_points[pos_inds]
            pos_decoded_bbox_preds = self.bbox_coder.decode(
                pos_points, pos_bbox_offsets)
            pos_decoded_target_preds = self.bbox_coder.decode(
                pos_points, pos_bbox_targets)
            loss_bbox = self.loss_bbox(pos_decoded_bbox_preds, pos_decoded_target_preds)
        else:
            loss_bbox = pos_bbox_offsets.sum() * 0

        losses['loss_cls'] = loss_cls
        losses['loss_bbox'] = loss_bbox
        return losses

    def get_targets(
            self, points: List[Tensor], batch_gt_instances: InstanceList
    ) -> Tuple[List[Tensor], List[Tensor]]:
        """Compute regression and classification targets for points
        in multiple images.

        Args:
            points (list[Tensor]): Points of each fpn level, each has shape
                (num_points, 2).
            batch_gt_instances (list[:obj:`InstanceData`]): Batch of
                gt_instance.  It usually includes ``bboxes`` and ``labels``
                attributes.

        Returns:
            tuple: Targets of each level.

            - concat_lvl_labels (list[Tensor]): Labels of each level.
            - concat_lvl_bbox_targets (list[Tensor]): BBox targets of each \
            level.
        """
        assert len(points) == len(self.regress_ranges)
        num_levels = len(points)
        # expand regress ranges to align with points
        expanded_regress_ranges = [
            points[i].new_tensor(self.regress_ranges[i])[None].expand_as(
                points[i]) for i in range(num_levels)
        ]
        # concat all levels points and regress ranges
        concat_regress_ranges = torch.cat(expanded_regress_ranges, dim=0)
        concat_points = torch.cat(points, dim=0)

        # the number of points per img, per lvl
        num_points = [center.size(0) for center in points]

        # get labels and bbox_targets of each image
        labels_list, bbox_targets_list = multi_apply(
            self._get_targets_single,
            batch_gt_instances,
            points=concat_points,
            regress_ranges=concat_regress_ranges,
            num_points_per_lvl=num_points)

        # from imgs[concat(levels)] to levels[concat(imgs)]
        # split to per img, per level
        labels_list = [labels.split(num_points, 0) for labels in labels_list]
        bbox_targets_list = [
            bbox_targets.split(num_points, 0)
            for bbox_targets in bbox_targets_list
        ]

        # concat per level image
        concat_lvl_labels = []
        concat_lvl_bbox_targets = []
        for i in range(num_levels):
            concat_lvl_labels.append(
                torch.cat([labels[i] for labels in labels_list]))
            bbox_targets = torch.cat(
                [bbox_targets[i] for bbox_targets in bbox_targets_list])
            # scale down the bbox_targets(left, top, right, bottom) by strides(w, h)
            bbox_targets[:, ::2] /= self.strides[i][0]
            bbox_targets[:, 1::2] /= self.strides[i][1]
            concat_lvl_bbox_targets.append(bbox_targets)
        return concat_lvl_labels, concat_lvl_bbox_targets

    def _get_targets_single(
            self, gt_instances: InstanceData, points: Tensor,
            regress_ranges: Tensor,
            num_points_per_lvl: List[int]) -> Tuple[Tensor, Tensor]:
        """Compute regression and classification targets for a single image."""
        num_points = points.size(0)
        num_gts = len(gt_instances)
        gt_bboxes = gt_instances.bboxes
        gt_labels = gt_instances.labels

        if num_gts == 0:
            return gt_labels.new_full((num_points,), self.num_classes), \
                gt_bboxes.new_zeros((num_points, 4))

        areas = (gt_bboxes[:, 2] - gt_bboxes[:, 0]) * (
                gt_bboxes[:, 3] - gt_bboxes[:, 1])
        areas = areas[None].repeat(num_points, 1)
        regress_ranges = regress_ranges[:, None, :].expand(
            num_points, num_gts, 2)
        gt_bboxes = gt_bboxes[None].expand(num_points, num_gts, 4)
        xs, ys = points[:, 0], points[:, 1]
        xs = xs[:, None].expand(num_points, num_gts)
        ys = ys[:, None].expand(num_points, num_gts)

        left = xs - gt_bboxes[..., 0]
        right = gt_bboxes[..., 2] - xs
        top = ys - gt_bboxes[..., 1]
        bottom = gt_bboxes[..., 3] - ys
        bbox_targets = torch.stack((left, top, right, bottom), -1)

        if self.center_sampling:
            # condition1: inside a `center bbox`
            radius = self.center_sample_radius
            center_xs = (gt_bboxes[..., 0] + gt_bboxes[..., 2]) / 2
            center_ys = (gt_bboxes[..., 1] + gt_bboxes[..., 3]) / 2
            center_gts = torch.zeros_like(gt_bboxes)
            center_w = center_xs.new_zeros(center_xs.shape)
            center_h = center_xs.new_zeros(center_xs.shape)

            # project the points on current lvl back to the `original` sizes
            lvl_begin = 0
            for lvl_idx, num_points_lvl in enumerate(num_points_per_lvl):
                lvl_end = lvl_begin + num_points_lvl
                center_w[lvl_begin:lvl_end] = self.strides[lvl_idx][0] * radius
                center_h[lvl_begin:lvl_end] = self.strides[lvl_idx][1] * radius
                lvl_begin = lvl_end

            x_mins = center_xs - center_w
            y_mins = center_ys - center_h
            x_maxs = center_xs + center_w
            y_maxs = center_ys + center_h
            center_gts[..., 0] = torch.where(x_mins > gt_bboxes[..., 0],
                                             x_mins, gt_bboxes[..., 0])
            center_gts[..., 1] = torch.where(y_mins > gt_bboxes[..., 1],
                                             y_mins, gt_bboxes[..., 1])
            center_gts[..., 2] = torch.where(x_maxs > gt_bboxes[..., 2],
                                             gt_bboxes[..., 2], x_maxs)
            center_gts[..., 3] = torch.where(y_maxs > gt_bboxes[..., 3],
                                             gt_bboxes[..., 3], y_maxs)

            cb_dist_left = xs - center_gts[..., 0]
            cb_dist_right = center_gts[..., 2] - xs
            cb_dist_top = ys - center_gts[..., 1]
            cb_dist_bottom = center_gts[..., 3] - ys
            center_bbox = torch.stack(
                (cb_dist_left, cb_dist_top, cb_dist_right, cb_dist_bottom), -1)
            inside_gt_bbox_mask = center_bbox.min(-1)[0] > 0
        else:
            # condition1: inside a gt bbox
            inside_gt_bbox_mask = bbox_targets.min(-1)[0] > 0

        # condition2: limit the regression range for each location
        max_regress_distance = bbox_targets.max(-1)[0]
        inside_regress_range = (
                (max_regress_distance >= regress_ranges[..., 0])
                & (max_regress_distance <= regress_ranges[..., 1]))

        # if there are still more than one objects for a location, we choose the one with minimal area as target.
        # however, if areas of these objects are very similar (or even the same),
        # then we take all of them as the classification targets by using a soft classification label.
        areas[inside_gt_bbox_mask == 0] = INF
        areas[inside_regress_range == 0] = INF
        min_area, min_area_inds = areas.min(dim=1)
        min_area_mask = torch.logical_and(
            (areas <= (min_area[:, None] + 1e-3)), (areas < INF)
        ).to(bbox_targets.dtype)
        gt_labels_one_hot = F.one_hot(gt_labels, self.num_classes).to(bbox_targets.dtype)
        cls_targets = min_area_mask @ gt_labels_one_hot
        # to prevent multiple GT actions with the same label and boundaries
        cls_targets.clamp_(min=0.0, max=1.0)

        # set points that do not match any ground truth as BG.
        cls_targets[min_area == INF] = 0
        # retain bbox_targets the ones with respect to their minimal gt bbox
        # [points, num_gts, 4] -> [points, 4]
        bbox_targets = bbox_targets[range(num_points), min_area_inds]

        return cls_targets, bbox_targets

    def _bbox_post_process(self,
                           results: InstanceData,
                           cfg: ConfigDict,
                           rescale: bool = False,
                           with_nms: bool = True,
                           img_meta: Optional[dict] = None) -> InstanceData:
        """bbox post-processing method.

        The boxes would be rescaled to the original image scale and do
        the nms operation. Usually `with_nms` is False is used for aug test.

        Args:
            results (:obj:`InstaceData`): Detection instance results,
                each item has shape (num_bboxes, ).
            cfg (ConfigDict): Test / postprocessing configuration,
                if None, test_cfg would be used.
            rescale (bool): If True, return boxes in original image space.
                Default to False.
            with_nms (bool): If True, do nms before return boxes.
                Default to True.
            img_meta (dict, optional): Image meta info. Defaults to None.

        Returns:
            :obj:`InstanceData`: Detection results of each image
            after the post process.
            Each item usually contains following keys.

                - scores (Tensor): Classification scores, has a shape
                  (num_instance, )
                - labels (Tensor): Labels of bboxes, has a shape
                  (num_instances, ).
                - bboxes (Tensor): Has a shape (num_instances, 4),
                  the last dimension 4 arrange as (x1, y1, x2, y2).
        """
        # %% scale the bboxes
        if rescale:
            assert img_meta.get('scale_factor') is not None
            scale_factor = [1 / s for s in img_meta['scale_factor']]
            results.bboxes = scale_boxes(results.bboxes, scale_factor)

        # %% scale the scores
        if hasattr(results, 'score_factors'):
            # TODO: Add sqrt operation in order to be consistent with
            #  the paper.
            score_factors = results.pop('score_factors')
            results.scores = results.scores * score_factors

        # %% filter small size bboxes
        if cfg.get('min_bbox_size'):
            w, h = get_box_wh(results.bboxes)
            valid_mask = (w > cfg.min_bbox_size[0]) & (h > cfg.min_bbox_size[1])
            if not valid_mask.all():
                results = results[valid_mask]

        # %% perform NMS
        # TODO: deal with `with_nms` and `nms_cfg=None` in test_cfg
        if with_nms and results.bboxes.numel() > 0:
            bboxes = get_box_tensor(results.bboxes)
            det_bboxes_scores, keep_idxs = batched_nms(bboxes, results.scores, results.labels, cfg.nms)
            det_bboxes = det_bboxes_scores[:, :-1]
            det_scores = det_bboxes_scores[:, -1]
            det_labels = results.labels[keep_idxs]
            # %% perform score voting
            if cfg.get('with_score_voting', False) and len(det_bboxes) > 0:
                det_bboxes = self.score_voting(det_bboxes, det_labels,
                                               results.bboxes, results.scores,
                                               iou_thr=cfg.voting_iou_thr,
                                               score_thr=cfg.voting_score_thr)
            results = InstanceData()
            results.bboxes = det_bboxes
            results.scores = det_scores
            results.labels = det_labels

            # %% Select top-k result
            sorted_results = results[results.scores.sort(descending=True).indices]
            results = sorted_results[:cfg.max_per_img]

        return results

    def score_voting(self,
                     det_bboxes: Tensor, det_labels: Tensor,
                     all_bboxes: Tensor, all_scores: Tensor,
                     iou_thr: float = 0.01,
                     score_thr: float = 0.0,
                     class_agnostic: bool = False) -> Tensor:
        """Implementation of score voting method works on each remaining boxes
        after NMS procedure.

        Args:
            det_bboxes (Tensor): Remaining boxes after NMS procedure,
                with shape (k, 4).
            det_labels (Tensor): The label of remaining boxes, with shape
                (k, 1),Labels are 0-based.
            all_bboxes (Tensor): All boxes before the NMS procedure,
                with shape (num_anchors,4).
            all_scores (Tensor): The scores of all boxes which is used
                in the NMS procedure, with shape (num_anchors, num_class)
            iou_thr (float): The IoU threshold of bboxes for voting.
            score_thr (float): The score threshold of bboxes for voting.
            class_agnostic (bool): Perform voting class-wisely if False,
            otherwise perform voting across all bboxes.

        Returns:
            det_bboxes_voted (Tensor): Re-weighted det_boxes after
                    score voting procedure, with shape (k, 4).
        """

        def bbox_voting(bboxes, voting_bboxes, voting_bbox_scores):
            # IoU tables of shape [num_after_nms, num_before_nms]
            iou_table = bbox_overlaps(bboxes, voting_bboxes)
            pos_scores = voting_bbox_scores * (iou_table > iou_thr).astype(voting_bbox_scores)
            weights = pos_scores * iou_table
            weights = weights / weights.sum(dim=1, keepdim=True)
            voted_bboxes = weights @ voting_bboxes
            return voted_bboxes

        # %% Filter all boxes that have score less than score_thr
        pos_score_mask = all_scores > score_thr
        pos_score_inds, pos_score_labels = pos_score_mask.nonzero()

        all_bboxes = all_bboxes[pos_score_inds]
        all_scores = all_scores[pos_score_inds]
        all_labels = pos_score_labels

        if class_agnostic:
            det_bboxes_voted = bbox_voting(det_bboxes, all_bboxes, all_scores)
        else:
            # %% Perform score voting class-wisely
            det_bboxes_voted = []
            for cls in range(self.cls_out_channels):
                pos_cls_mask_all = (all_labels == cls)
                pos_cls_mask_det = (det_labels == cls)
                if not pos_cls_mask_all.any():
                    continue
                _all_scores = all_scores[pos_cls_mask_all]
                _all_bboxes = all_bboxes[pos_cls_mask_all]
                _det_bboxes = det_bboxes[pos_cls_mask_det]

                _det_bboxes_voted = bbox_voting(_det_bboxes, _all_bboxes, _all_scores)
                det_bboxes_voted.append(_det_bboxes_voted)
            det_bboxes_voted = torch.cat(det_bboxes_voted, dim=0)
        return det_bboxes_voted
