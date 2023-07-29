from typing import Dict, List, Tuple

import torch
import torch.nn as nn
from mmdet.models.dense_heads import DINOHead
from mmdet.models.layers import inverse_sigmoid
from mmdet.registry import MODELS
from mmdet.structures.bbox import bbox_cxcywh_to_xyxy
from mmdet.structures.bbox import bbox_overlaps
from mmdet.utils import InstanceList
from mmdet.utils import reduce_mean
from mmengine.model import bias_init_with_prob, constant_init
from torch import Tensor

from my_modules.layers.pseudo_layers import Pseudo4DRegLinear
from my_modules.loss.custom_loss import PositionFocalLoss


@MODELS.register_module()
class CustomDINOHead(DINOHead):
    """
    Customized DINO Head to support Temporal Action Detection.
    1. We modify the regression branches to remove the unused FC nodes (x1, y1, x2, y2) -> (x1, x2).
    Note that this modification is optional since we have already modified the loss functions to
    make sure that the y1, y2 will not contribute to the loss and cost. See my_modules/loss/custom_loss.py
    2. Modify the loss function to support Position-supervised Focal Loss (Stable-DINO)
    3. Correct the split_outputs, now support turning off the de-noising branch.
    """

    def _init_layers(self) -> None:
        """Change the regression output dimension from 4 to 2"""
        super()._init_layers()
        for reg_branch in self.reg_branches:
            reg_branch[-1] = Pseudo4DRegLinear(self.embed_dims)

    def init_weights(self) -> None:
        """Initialize weights of the Deformable DETR head."""
        if self.loss_cls.use_sigmoid:
            bias_init = bias_init_with_prob(0.01)
            for m in self.cls_branches:
                nn.init.constant_(m.bias, bias_init)
        for m in self.reg_branches:
            constant_init(m[-1], 0, bias=0)
        nn.init.constant_(self.reg_branches[0][-1].bias.data[1:], -2.0)

    def forward(self, hidden_states: Tensor,
                references: List[Tensor]) -> Tuple[Tensor, Tensor]:
        """Forward function.

        Args:
            hidden_states (Tensor): Hidden states output from each decoder
                layer, has shape (num_decoder_layers, bs, num_queries, dim).
            references (list[Tensor]): List of the reference from the decoder.
                The first reference is the `init_reference` (initial) and the
                other num_decoder_layers(6) references are `inter_references`
                (intermediate). The `init_reference` has shape (bs,
                num_queries, 4) when `as_two_stage` of the detector is `True`,
                otherwise (bs, num_queries, 2). Each `inter_reference` has
                shape (bs, num_queries, 4) when `with_box_refine` of the
                detector is `True`, otherwise (bs, num_queries, 2). The
                coordinates are arranged as (cx, cy) when the last dimension is
                2, and (cx, cy, w, h) when it is 4.

        Returns:
            tuple[Tensor]: results of head containing the following tensor.

            - all_layers_outputs_classes (Tensor): Outputs from the
              classification head, has shape (num_decoder_layers, bs,
              num_queries, cls_out_channels).
            - all_layers_outputs_coords (Tensor): Sigmoid outputs from the
              regression head with normalized coordinate format (cx, cy, w,
              h), has shape (num_decoder_layers, bs, num_queries, 4) with the
              last dimension arranged as (cx, cy, w, h).
        """
        # TODO: directly use the reference points of nex layer as output_coord?
        all_layers_outputs_classes = []
        all_layers_outputs_coords = []

        for layer_id in range(len(hidden_states)):
            reference = inverse_sigmoid(references[layer_id])
            # NOTE The last reference will not be used.
            hidden_state = hidden_states[layer_id]
            outputs_class = self.cls_branches[layer_id](hidden_state)
            tmp_reg_preds = self.reg_branches[layer_id](hidden_state)
            if reference.shape[-1] == 4:
                # When `layer` is 0 and `as_two_stage` of the detector
                # is `True`, or when `layer` is greater than 0 and
                # `with_box_refine` of the detector is `True`.
                tmp_reg_preds += reference
            else:
                # When `layer` is 0 and `as_two_stage` of the detector
                # is `False`, or when `layer` is greater than 0 and
                # `with_box_refine` of the detector is `False`.
                assert reference.shape[-1] == 2
                tmp_reg_preds[..., :2] += reference
            outputs_coord = tmp_reg_preds.sigmoid()
            all_layers_outputs_classes.append(outputs_class)
            all_layers_outputs_coords.append(outputs_coord)
        #
        # all_layers_outputs_classes = torch.cat(all_layers_outputs_classes, dim=0)
        # all_layers_outputs_coords = torch.cat(all_layers_outputs_coords, dim=0)

        return all_layers_outputs_classes, all_layers_outputs_coords

    def loss_by_feat_single(self, cls_scores: Tensor, bbox_preds: Tensor,
                            batch_gt_instances: InstanceList,
                            batch_img_metas: List[dict]) -> Tuple[Tensor]:
        """Loss function for outputs from a single decoder layer of a single
        feature level.

        Args:
            cls_scores (Tensor): Box score logits from a single decoder layer
                for all images, has shape (bs, num_queries, cls_out_channels).
            bbox_preds (Tensor): Sigmoid outputs from a single decoder layer
                for all images, with normalized coordinate (cx, cy, w, h) and
                shape (bs, num_queries, 4).
            batch_gt_instances (list[:obj:`InstanceData`]): Batch of
                gt_instance. It usually includes ``bboxes`` and ``labels``
                attributes.
            batch_img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.

        Returns:
            Tuple[Tensor]: A tuple including `loss_cls`, `loss_box` and
            `loss_iou`.
        """
        num_imgs = cls_scores.size(0)
        cls_scores_list = [cls_scores[i] for i in range(num_imgs)]
        bbox_preds_list = [bbox_preds[i] for i in range(num_imgs)]

        num_targets = len(batch_gt_instances)
        if num_targets < num_imgs:  # When SQR applied, queries are recollected from layers, increasing the batch_size.
            assert num_imgs % num_targets == 0
            repeats = num_imgs // num_targets
            batch_gt_instances = batch_gt_instances * repeats
            batch_img_metas = batch_img_metas * repeats

        cls_reg_targets = self.get_targets(cls_scores_list, bbox_preds_list,
                                           batch_gt_instances, batch_img_metas)
        (labels_list, label_weights_list, bbox_targets_list, bbox_weights_list,
         num_total_pos, num_total_neg) = cls_reg_targets
        labels = torch.cat(labels_list, 0)
        label_weights = torch.cat(label_weights_list, 0)
        bbox_targets = torch.cat(bbox_targets_list, 0)
        bbox_weights = torch.cat(bbox_weights_list, 0)

        # classification loss
        cls_scores = cls_scores.reshape(-1, self.cls_out_channels)
        # construct weighted avg_factor to match with the official DETR repo
        cls_avg_factor = num_total_pos * 1.0 + \
                         num_total_neg * self.bg_cls_weight
        if self.sync_cls_avg_factor:
            cls_avg_factor = reduce_mean(
                cls_scores.new_tensor([cls_avg_factor]))
        cls_avg_factor = max(cls_avg_factor, 1)

        # Compute the average number of gt boxes across all gpus, for
        # normalization purposes
        num_total_pos = cls_avg_factor.new_tensor([num_total_pos])
        num_total_pos = torch.clamp(reduce_mean(num_total_pos), min=1).item()

        # construct factors used for rescale bboxes
        factors = []
        for img_meta, bbox_pred in zip(batch_img_metas, bbox_preds):
            img_h, img_w, = img_meta['img_shape']
            factor = bbox_pred.new_tensor([img_w, img_h, img_w,
                                           img_h]).unsqueeze(0).repeat(
                bbox_pred.size(0), 1)
            factors.append(factor)
        factors = torch.cat(factors, 0)

        # DETR regress the relative position of boxes (cxcywh) in the image,
        # thus the learning target is normalized by the image size. So here
        # we need to re-scale them for calculating IoU loss
        bbox_preds = bbox_preds.reshape(-1, 4)
        bboxes = bbox_cxcywh_to_xyxy(bbox_preds) * factors
        bboxes_gt = bbox_cxcywh_to_xyxy(bbox_targets) * factors

        # regression IoU loss, defaultly GIoU loss
        loss_iou = self.loss_iou(
            bboxes, bboxes_gt, bbox_weights, avg_factor=num_total_pos)

        # regression L1 loss
        loss_bbox = self.loss_bbox(
            bbox_preds, bbox_targets, bbox_weights, avg_factor=num_total_pos)

        # classification loss
        if isinstance(self.loss_cls, PositionFocalLoss):
            bboxes = bboxes.detach()
            bboxes[:, 1::2] = bboxes_gt[:, 1::2]
            ious = bbox_overlaps(bboxes, bboxes_gt, is_aligned=True).clamp(min=1e-6).unsqueeze(-1)
            loss_cls = self.loss_cls(
                cls_scores, labels, ious, label_weights, avg_factor=cls_avg_factor)
        else:
            loss_cls = self.loss_cls(
                cls_scores, labels, label_weights, avg_factor=cls_avg_factor)
        return loss_cls, loss_bbox, loss_iou

    def _loss_dn_single(self, dn_cls_scores: Tensor, dn_bbox_preds: Tensor,
                        batch_gt_instances: InstanceList,
                        batch_img_metas: List[dict],
                        dn_meta: Dict[str, int]) -> Tuple[Tensor]:
        """Denoising loss for outputs from a single decoder layer.

        Args:
            dn_cls_scores (Tensor): Classification scores of a single decoder
                layer in denoising part, has shape (bs, num_denoising_queries,
                cls_out_channels).
            dn_bbox_preds (Tensor): Regression outputs of a single decoder
                layer in denoising part. Each is a 4D-tensor with normalized
                coordinate format (cx, cy, w, h) and has shape
                (bs, num_denoising_queries, 4).
            batch_gt_instances (list[:obj:`InstanceData`]): Batch of
                gt_instance. It usually includes ``bboxes`` and ``labels``
                attributes.
            batch_img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.
            dn_meta (Dict[str, int]): The dictionary saves information about
              group collation, including 'num_denoising_queries' and
              'num_denoising_groups'. It will be used for split outputs of
              denoising and matching parts and loss calculation.

        Returns:
            Tuple[Tensor]: A tuple including `loss_cls`, `loss_box` and
            `loss_iou`.
        """
        num_preds, num_targets = dn_bbox_preds.shape[0], len(batch_gt_instances)
        if num_targets < num_preds:  # When SQR applied, queries are recollected from layers, increasing the batch_size.
            assert num_preds % num_targets == 0
            repeats = num_preds // num_targets
            batch_gt_instances = batch_gt_instances * repeats
            batch_img_metas = batch_img_metas * repeats

        cls_reg_targets = self.get_dn_targets(batch_gt_instances,
                                              batch_img_metas, dn_meta)
        (labels_list, label_weights_list, bbox_targets_list, bbox_weights_list,
         num_total_pos, num_total_neg) = cls_reg_targets

        labels = torch.cat(labels_list, 0)
        label_weights = torch.cat(label_weights_list, 0)
        bbox_targets = torch.cat(bbox_targets_list, 0)
        bbox_weights = torch.cat(bbox_weights_list, 0)

        # classification loss
        cls_scores = dn_cls_scores.reshape(-1, self.cls_out_channels)
        # construct weighted avg_factor to match with the official DETR repo
        cls_avg_factor = \
            num_total_pos * 1.0 + num_total_neg * self.bg_cls_weight
        if self.sync_cls_avg_factor:
            cls_avg_factor = reduce_mean(
                cls_scores.new_tensor([cls_avg_factor]))
        cls_avg_factor = max(cls_avg_factor, 1)

        # Compute the average number of gt boxes across all gpus, for
        # normalization purposes
        num_total_pos = cls_avg_factor.new_tensor([num_total_pos])
        num_total_pos = torch.clamp(reduce_mean(num_total_pos), min=1).item()

        # construct factors used for rescale bboxes
        factors = []
        for img_meta, bbox_pred in zip(batch_img_metas, dn_bbox_preds):
            img_h, img_w = img_meta['img_shape']
            factor = bbox_pred.new_tensor([img_w, img_h, img_w,
                                           img_h]).unsqueeze(0).repeat(
                bbox_pred.size(0), 1)
            factors.append(factor)
        factors = torch.cat(factors)

        # DETR regress the relative position of boxes (cxcywh) in the image,
        # thus the learning target is normalized by the image size. So here
        # we need to re-scale them for calculating IoU loss
        bbox_preds = dn_bbox_preds.reshape(-1, 4)
        bboxes = bbox_cxcywh_to_xyxy(bbox_preds) * factors
        bboxes_gt = bbox_cxcywh_to_xyxy(bbox_targets) * factors

        # regression IoU loss, defaultly GIoU loss
        loss_iou = self.loss_iou(
            bboxes, bboxes_gt, bbox_weights, avg_factor=num_total_pos)

        # regression L1 loss
        loss_bbox = self.loss_bbox(
            bbox_preds, bbox_targets, bbox_weights, avg_factor=num_total_pos)

        if len(cls_scores) > 0:
            if isinstance(self.loss_cls, PositionFocalLoss):
                bboxes = bboxes.detach()
                bboxes[:, 1::2] = bboxes_gt[:, 1::2]
                ious = bbox_overlaps(bboxes, bboxes_gt, is_aligned=True).clamp(min=1e-6).unsqueeze(-1)
                loss_cls = self.loss_cls(
                    cls_scores, labels, ious, label_weights, avg_factor=cls_avg_factor)
            else:
                loss_cls = self.loss_cls(
                    cls_scores, labels, label_weights, avg_factor=cls_avg_factor)
        else:
            loss_cls = torch.zeros(
                1, dtype=cls_scores.dtype, device=cls_scores.device)
        return loss_cls, loss_bbox, loss_iou

    @staticmethod
    def split_outputs(all_layers_cls_scores: Tensor,
                      all_layers_bbox_preds: Tensor,
                      dn_meta: Dict[str, int]) -> Tuple[Tensor]:
        if dn_meta is not None:
            num_denoising_queries = dn_meta['num_denoising_queries']
            if isinstance(all_layers_cls_scores, list): # when SQR applied, the number of queries in layers are different
                batch_size, _, num_classes = all_layers_cls_scores[0].shape
                all_layers_denoising_cls_scores = [layer_scores[:, :num_denoising_queries, :] for layer_scores in all_layers_cls_scores]
                all_layers_matching_cls_scores = [layer_scores[:, num_denoising_queries:, :] for layer_scores in all_layers_cls_scores]
                all_layers_denoising_bbox_preds = [layer_box[:, :num_denoising_queries, :] for layer_box in all_layers_bbox_preds]
                all_layers_matching_bbox_preds = [layer_box[:, num_denoising_queries:, :] for layer_box in all_layers_bbox_preds]

                # # Roll-back the batch size increased by recollection. Note that you cannot use reshape because it's interleaved
                # all_layers_denoising_cls_scores = [torch.cat(layer_dn_scores.split(batch_size), dim=1) for layer_dn_scores in all_layers_denoising_cls_scores]
                # all_layers_matching_cls_scores = [torch.cat(layer_scores.split(batch_size), dim=1) for layer_scores in all_layers_matching_cls_scores ]
                # all_layers_denoising_bbox_preds = [torch.cat(layer_dn_box.split(batch_size), dim=1)for layer_dn_box in all_layers_denoising_bbox_preds]
                # all_layers_matching_bbox_preds = [torch.cat(layer_box.split(batch_size), dim=1) for layer_box in all_layers_matching_bbox_preds]
            else:
                all_layers_denoising_cls_scores = \
                    all_layers_cls_scores[:, :, : num_denoising_queries, :]
                all_layers_denoising_bbox_preds = \
                    all_layers_bbox_preds[:, :, : num_denoising_queries, :]
                all_layers_matching_cls_scores = \
                    all_layers_cls_scores[:, :, num_denoising_queries:, :]
                all_layers_matching_bbox_preds = \
                    all_layers_bbox_preds[:, :, num_denoising_queries:, :]
        else:
            all_layers_denoising_cls_scores = None
            all_layers_denoising_bbox_preds = None
            all_layers_matching_cls_scores = all_layers_cls_scores
            all_layers_matching_bbox_preds = all_layers_bbox_preds
        return (all_layers_matching_cls_scores, all_layers_matching_bbox_preds,
                all_layers_denoising_cls_scores,
                all_layers_denoising_bbox_preds)
