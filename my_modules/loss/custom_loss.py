from typing import Optional

import torch
import torch.nn.functional as F
from mmdet.models.losses import L1Loss, GIoULoss, DIoULoss, IoULoss, FocalLoss, weight_reduce_loss
from mmdet.models.task_modules import IoUCost, BBoxL1Cost, FocalLossCost
from mmdet.registry import MODELS, TASK_UTILS
from mmengine.structures import InstanceData
from torch import Tensor


def zero_out_loss_coordinates_decorator(forward_method):
    def wrapper(self, pred: Tensor, target: Tensor, *args, **kwargs):
        pred = pred.clone()
        pred[:, 1] = pred[:, 1] * 0 + target[:, 1]
        pred[:, 3] = pred[:, 3] * 0 + target[:, 3]
        return forward_method(self, pred, target, *args, **kwargs)

    return wrapper


def zero_out_pred_coordinates_decorator(forward_method):
    def wrapper(self, pred_instances: InstanceData, gt_instances: InstanceData, *args, **kwargs):
        pred_instances.bboxes[:, 1] = gt_instances.bboxes[0, 1]
        pred_instances.bboxes[:, 3] = gt_instances.bboxes[0, 3]
        return forward_method(self, pred_instances, gt_instances, *args, **kwargs)

    return wrapper


def py_sigmoid_focal_loss(pred,
                          target,
                          iou,
                          weight=None,
                          gamma=2.0,
                          alpha=0.25,
                          reduction='mean',
                          avg_factor=None):
    """PyTorch version of `Focal Loss <https://arxiv.org/abs/1708.02002>`_.

    Args:
        pred (torch.Tensor): The prediction with shape (N, C), C is the
            number of classes
        target (torch.Tensor): The learning label of the prediction.
        weight (torch.Tensor, optional): Sample-wise loss weight.
        gamma (float, optional): The gamma for calculating the modulating
            factor. Defaults to 2.0.
        alpha (float, optional): A balanced form for Focal Loss.
            Defaults to 0.25.
        reduction (str, optional): The method used to reduce the loss into
            a scalar. Defaults to 'mean'.
        avg_factor (int, optional): Average factor that is used to average
            the loss. Defaults to None.
    """
    pred_sigmoid = pred.sigmoid()
    target = target.type_as(pred)
    f = iou ** 2
    f = f / f.max()
    pt = (f - pred_sigmoid) * target * f + pred_sigmoid * (1 - target)
    focal_weight = (alpha * target + (1 - alpha) *
                    (1 - target)) * pt.pow(gamma)
    loss = F.binary_cross_entropy_with_logits(
        pred, target, reduction='none') * focal_weight
    if weight is not None:
        if weight.shape != loss.shape:
            if weight.size(0) == loss.size(0):
                # For most cases, weight is of shape (num_priors, ),
                #  which means it does not have the second axis num_class
                weight = weight.view(-1, 1)
            else:
                # Sometimes, weight per anchor per class is also needed. e.g.
                #  in FSAF. But it may be flattened of shape
                #  (num_priors x num_class, ), while loss is still of shape
                #  (num_priors, num_class).
                assert weight.numel() == loss.numel()
                weight = weight.view(loss.size(0), -1)
        assert weight.ndim == loss.ndim
    loss = weight_reduce_loss(loss, weight, reduction, avg_factor)
    return loss


@MODELS.register_module()
class PositionFocalLoss(FocalLoss):
    def forward(self,
                pred,
                target,
                iou,
                weight=None,
                avg_factor=None,
                reduction_override=None):
        """Forward function.

        Args:
            pred (torch.Tensor): The prediction.
            target (torch.Tensor): The learning label of the prediction.
                The target shape support (N,C) or (N,), (N,C) means
                one-hot form.
            weight (torch.Tensor, optional): The weight of loss for each
                prediction. Defaults to None.
            avg_factor (int, optional): Average factor that is used to average
                the loss. Defaults to None.
            reduction_override (str, optional): The reduction method used to
                override the original reduction method of the loss.
                Options are "none", "mean" and "sum".

        Returns:
            torch.Tensor: The calculated loss
        """
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)
        if self.use_sigmoid:
            if pred.dim() == target.dim():
                # this means that target is already in One-Hot form.
                calculate_loss_func = py_sigmoid_focal_loss
            # elif torch.cuda.is_available() and pred.is_cuda:
            #     calculate_loss_func = sigmoid_focal_loss
            else:
                num_classes = pred.size(1)
                target = F.one_hot(target, num_classes=num_classes + 1)
                target = target[:, :num_classes]
                calculate_loss_func = py_sigmoid_focal_loss

            loss_cls = self.loss_weight * calculate_loss_func(
                pred,
                target,
                iou,
                weight,
                gamma=self.gamma,
                alpha=self.alpha,
                reduction=reduction,
                avg_factor=avg_factor)

        else:
            raise NotImplementedError
        return loss_cls


@MODELS.register_module()
class PositionFocalLossCost(FocalLossCost):

    def _focal_loss_cost(self, cls_pred: Tensor, gt_labels: Tensor, giou: Tensor) -> Tensor:
        """
        Args:
            cls_pred (Tensor): Predicted classification logits, shape
                (num_queries, num_class).
            gt_labels (Tensor): Label of `gt_bboxes`, shape (num_gt,).

        Returns:
            torch.Tensor: cls_cost value with weight
        """
        f = (giou + 1) / 2
        f = f ** 0.5
        cls_pred = cls_pred[:, gt_labels].sigmoid() * f
        neg_cost = -(1 - cls_pred + self.eps).log() * (
                1 - self.alpha) * cls_pred.pow(self.gamma)
        pos_cost = -(cls_pred + self.eps).log() * self.alpha * (
                1 - cls_pred).pow(self.gamma)

        # cls_cost = pos_cost[:, gt_labels] - neg_cost[:, gt_labels]
        cls_cost = pos_cost - neg_cost
        return cls_cost * self.weight

    def __call__(self,
                 pred_instances: InstanceData,
                 gt_instances: InstanceData,
                 gious,
                 img_meta: Optional[dict] = None,
                 **kwargs) -> Tensor:
        """Compute match cost.

        Args:
            pred_instances (:obj:`InstanceData`): Predicted instances which
                must contain ``scores`` or ``masks``.
            gt_instances (:obj:`InstanceData`): Ground truth which must contain
                ``labels`` or ``mask``.
            img_meta (Optional[dict]): Image information. Defaults to None.

        Returns:
            Tensor: Match Cost matrix of shape (num_preds, num_gts).
        """
        if self.binary_input:
            pred_masks = pred_instances.masks
            gt_masks = gt_instances.masks
            return self._mask_focal_loss_cost(pred_masks, gt_masks)
        else:
            pred_scores = pred_instances.scores
            gt_labels = gt_instances.labels
            return self._focal_loss_cost(pred_scores, gt_labels, gious)


@MODELS.register_module()
class DIoU1dLoss(DIoULoss):
    @zero_out_loss_coordinates_decorator
    def forward(self, pred: Tensor, target: Tensor, *args, **kwargs) -> Tensor:
        return super().forward(pred, target, *args, **kwargs)


@MODELS.register_module(force=True)
class L11dLoss(L1Loss):
    """Custom L1 loss so that y1, y2 don't contribute to the loss by multiplying them with zeros."""

    @zero_out_loss_coordinates_decorator
    def forward(self, pred: Tensor, target: Tensor, *args, **kwargs):
        return super().forward(pred, target, *args, **kwargs)


@MODELS.register_module(force=True)
class IoU1dLoss(IoULoss):
    """Custom IoU loss so that y1, y2 don't contribute to the loss by multiplying them with zeros."""

    @zero_out_loss_coordinates_decorator
    def forward(self, pred: Tensor, target: Tensor, *args, **kwargs):
        return super().forward(pred, target, *args, **kwargs)


@MODELS.register_module(force=True)
class GIoU1dLoss(GIoULoss):
    """Custom GIoU loss so that y1, y2 don't contribute to the loss by multiplying them with zeros
    """

    @zero_out_loss_coordinates_decorator
    def forward(self, pred: Tensor, target: Tensor, *args, **kwargs):
        return super().forward(pred, target, *args, **kwargs)


@TASK_UTILS.register_module(force=True)
class IoU1dCost(IoUCost):
    @zero_out_pred_coordinates_decorator
    def __call__(self, pred_instances: InstanceData, gt_instances: InstanceData, *args, **kwargs):
        return super().__call__(pred_instances, gt_instances, *args, **kwargs)


@TASK_UTILS.register_module(force=True)
class BBox1dL1Cost(BBoxL1Cost):
    @zero_out_pred_coordinates_decorator
    def __call__(self, pred_instances: InstanceData, gt_instances: InstanceData, *args, **kwargs):
        return super().__call__(pred_instances, gt_instances, *args, **kwargs)
