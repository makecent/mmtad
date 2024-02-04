import numpy as np
import torch
from mmcv.ops import batched_nms
from mmdet.structures.bbox import bbox_overlaps
from torch import Tensor


def convert_1d_to_2d_bboxes(bboxes_1d, y1=0.1, y2=0.9):
    num_bboxes = bboxes_1d.shape[0]
    # Initialize the 2D bounding boxes tensor
    bboxes_2d = torch.zeros((num_bboxes, 4), device=bboxes_1d.device, dtype=bboxes_1d.dtype)

    # Set the fixed dimension value for ymin and ymax
    bboxes_2d[:, 1] = y1
    bboxes_2d[:, 3] = y2

    # Copy the 1D intervals (xmin and xmax) to the 2D bounding boxes
    bboxes_2d[:, 0::2] = bboxes_1d

    return bboxes_2d


def convert_2d_to_1d_bboxes(bboxes_2d):
    """
    Convert pseudo 2D bounding boxes back to 1D bounding boxes by extracting xmin and xmax.

    Args:
        bboxes_2d (torch.Tensor): Pseudo 2D bounding boxes tensor of shape (N, 4)

    Returns:
        torch.Tensor: 1D bounding boxes tensor of shape (N, 2)
    """
    # Extract xmin and xmax (first and third columns) from the 2D bounding boxes
    bboxes_1d = bboxes_2d[:, 0::2]

    return bboxes_1d


def batched_nms1d(bboxes_1d, *args, **kwargs):
    """
    Apply Non-Maximum Suppression (NMS) on 1D bounding boxes by converting them to pseudo 2D bounding boxes,
    using a 2D NMS function, and converting the results back to 1D bounding boxes.

    Args:
        bboxes_1d (torch.Tensor): 1D bounding boxes tensor of shape (N, 2)
        *args: Additional arguments to pass to the batched_nms function
        **kwargs: Additional keyword arguments to pass to the batched_nms function

    Returns:
        tuple: A tuple containing:
            - torch.Tensor: The 1D bounding boxes after NMS, with shape (N', 2)
            - torch.Tensor: The indices of the kept bounding boxes, with shape (N',)
    """
    bboxes_2d = convert_1d_to_2d_bboxes(bboxes_1d)
    boxes, keep = batched_nms(bboxes_2d, *args, **kwargs)
    return convert_2d_to_1d_bboxes(boxes), keep


# Below all adapted from basicTAD
def segment_overlaps(segments1,
                     segments2,
                     mode='iou',
                     is_aligned=False,
                     eps=1e-6,
                     detect_overlap_edge=False):
    """Calculate overlap between two set of segments.
    If ``is_aligned`` is ``False``, then calculate the ious between each
    segment of segments1 and segments2, otherwise the ious between each aligned
     pair of segments1 and segments2.
    Args:
        segments1 (Tensor): shape (m, 2) in <t1, t2> format or empty.
        segments2 (Tensor): shape (n, 2) in <t1, t2> format or empty.
            If is_aligned is ``True``, then m and n must be equal.
        mode (str): "iou" (intersection over union) or iof (intersection over
            foreground).
    Returns:
        ious(Tensor): shape (m, n) if is_aligned == False else shape (m, 1)
    Example:
        >>> segments1 = torch.FloatTensor([
        >>>     [0, 10],
        >>>     [10, 20],
        >>>     [32, 38],
        >>> ])
        >>> segments2 = torch.FloatTensor([
        >>>     [0, 20],
        >>>     [0, 19],
        >>>     [10, 20],
        >>> ])
        >>> segment_overlaps(segments1, segments2)
        tensor([[0.5000, 0.5263, 0.0000],
                [0.0000, 0.4500, 1.0000],
                [0.0000, 0.0000, 0.0000]])
    Example:
        >>> empty = torch.FloatTensor([])
        >>> nonempty = torch.FloatTensor([
        >>>     [0, 9],
        >>> ])
        >>> assert tuple(segment_overlaps(empty, nonempty).shape) == (0, 1)
        >>> assert tuple(segment_overlaps(nonempty, empty).shape) == (1, 0)
        >>> assert tuple(segment_overlaps(empty, empty).shape) == (0, 0)
    """

    is_numpy = False
    if isinstance(segments1, np.ndarray):
        segments1 = torch.from_numpy(segments1)
        is_numpy = True
    if isinstance(segments2, np.ndarray):
        segments2 = torch.from_numpy(segments2)
        is_numpy = True

    segments1, segments2 = segments1.float(), segments2.float()

    assert mode in ['iou', 'iof']
    # Either the segments are empty or the length of segments's last dimenstion
    # is 2
    assert (segments1.size(-1) == 2 or segments1.size(0) == 0)
    assert (segments2.size(-1) == 2 or segments2.size(0) == 0)

    rows = segments1.size(0)
    cols = segments2.size(0)
    if is_aligned:
        assert rows == cols

    if rows * cols == 0:
        return segments1.new(rows, 1) if is_aligned else segments2.new(
            rows, cols)

    if is_aligned:
        start = torch.max(segments1[:, 0], segments2[:, 0])  # [rows]
        end = torch.min(segments1[:, 1], segments2[:, 1])  # [rows]

        overlap = end - start
        if detect_overlap_edge:
            overlap[overlap == 0] += eps
        overlap = overlap.clamp(min=0)  # [rows, 2]
        area1 = segments1[:, 1] - segments1[:, 0]

        if mode == 'iou':
            area2 = segments2[:, 1] - segments2[:, 0]
            union = area1 + area2 - overlap
        else:
            union = area1
    else:
        start = torch.max(segments1[:, None, 0], segments2[:,
                                                 0])  # [rows, cols]
        end = torch.min(segments1[:, None, 1], segments2[:, 1])  # [rows, cols]

        overlap = end - start
        if detect_overlap_edge:
            overlap[overlap == 0] += eps
        overlap = overlap.clamp(min=0)  # [rows, 2]
        area1 = segments1[:, 1] - segments1[:, 0]

        if mode == 'iou':
            area2 = segments2[:, 1] - segments2[:, 0]
            union = area1[:, None] + area2 - overlap
        else:
            union = area1[:, None]

    eps = union.new_tensor([eps])
    union = torch.max(union, eps)
    ious = overlap / union

    if is_numpy:
        ious = ious.numpy()

    return ious


def bbox_voting(det_bboxes: Tensor, det_labels: Tensor,
                all_bboxes: Tensor, all_scores: Tensor,
                num_classes: int = None, all_labels: Tensor = None,
                iou_thr: float = 0.01, score_thr: float = 0.0,
                class_agnostic: bool = False) -> Tensor:
    """Implementation of score voting method works on each remaining boxes
    after NMS procedure. If voting.iou_thr == NMS.iou_thr, then NMS + voting = NMW.

    Args:
        det_bboxes (Tensor): Remaining boxes after NMS procedure,
            with shape (k, 4).
        det_labels (Tensor): The label of remaining boxes, with shape
            (k, 1),Labels are 0-based.
        all_bboxes (Tensor): All boxes before the NMS procedure,
            with shape (num_anchors,4).
        all_scores (Tensor): The scores of all boxes which is used
            in the NMS procedure, with shape (num_anchors, num_class)
        num_classes: The number of classes, i.e., the max labels.
        all_labels (Tensor): The label of all boxes, with shape
            (k, 1). Labels are 0-based.
        iou_thr (float): The IoU threshold of bboxes for voting.
        score_thr (float): The score threshold of bboxes for voting.
        class_agnostic (bool): Perform voting class-wisely if False,
        otherwise perform voting across all bboxes.

    Returns:
        det_bboxes_voted (Tensor): Re-weighted det_boxes after
                score voting procedure, with shape (k, 4).
    """

    def _bbox_voting(bboxes, voting_bboxes, voting_bbox_scores):
        # IoU tables of shape [num_after_nms, num_before_nms]
        iou_table = bbox_overlaps(bboxes, voting_bboxes)
        pos_scores = voting_bbox_scores * (iou_table > iou_thr).type_as(voting_bbox_scores)
        weights = pos_scores * iou_table
        weights = weights / weights.sum(dim=1, keepdim=True)
        voted_bboxes = weights @ voting_bboxes
        return voted_bboxes

    # %% Filter all boxes that have score less than score_thr
    pos_score_mask = all_scores > score_thr

    all_bboxes = all_bboxes[pos_score_mask]
    all_scores = all_scores[pos_score_mask]
    all_labels = all_labels[pos_score_mask]

    if class_agnostic:
        det_bboxes_voted = _bbox_voting(det_bboxes, all_bboxes, all_scores)
    else:
        # %% Perform score voting class-wisely
        det_bboxes_voted = []
        for cls in range(num_classes):
            pos_cls_mask_all = (all_labels == cls)
            pos_cls_mask_det = (det_labels == cls)
            if not pos_cls_mask_all.any():
                continue
            _all_scores = all_scores[pos_cls_mask_all]
            _all_bboxes = all_bboxes[pos_cls_mask_all]
            _det_bboxes = det_bboxes[pos_cls_mask_det]

            _det_bboxes_voted = _bbox_voting(_det_bboxes, _all_bboxes, _all_scores)
            det_bboxes_voted.append(_det_bboxes_voted)
        det_bboxes_voted = torch.cat(det_bboxes_voted, dim=0)
    return det_bboxes_voted


def batched_nmw(bboxes,
                scores,
                labels,
                nms_cfg):
    """Non-Maximum Weighting for multi-class segments.

    Args:
        multi_segments (Tensor): shape (n, #class*2) or (n, 2)
        multi_scores (Tensor): shape (n, #class), where the last column
            contains scores of the background class, but this will be ignored.
        score_thr (float): segment threshold, segments with scores lower than
            it will not be considered.
        nms_cfg (dict): NMS cfg.
        max_num (int): if there are more than max_num segments after NMS,
            only top max_num will be kept.
        score_factors (Tensor): The factors multiplied to scores before
            applying NMS

    Returns:
        tuple: (segments, labels), tensors of shape (k, 3) and (k, 1). Labels
            are 0-based.
    """

    def _nmw1d(bboxes, scores, labels, remainings):
        from mmdet.structures.bbox import bbox_overlaps
        # select the best prediction (with the highest score) from the remaining.
        keep_idx = remainings[0]
        # collect predictions that have the same labels with the best.
        mask = labels[remainings] == labels[keep_idx]
        bboxes = bboxes[remainings][mask]
        scores = scores[remainings][mask]
        labels = labels[remainings][mask]

        # NMS output the best prediction and delete predictions that intersect with it with IoU >= iou_thr,
        # While NMW aggregates the best prediction and the predictions that intersect with it with IoU >= iou_thr and
        # outputs the aggregated prediction. The aggregation is based on the scores of the predictions.
        ious = bbox_overlaps(bboxes[:1], bboxes, mode='iou')[0]
        ious[0] = 1.0
        iou_mask = ious >= iou_thr
        aggregate_bboxes = bboxes[iou_mask]
        accu_weights = scores[iou_mask] * ious[iou_mask]
        accu_weights /= accu_weights.sum()
        bbox = (accu_weights[:, None] * aggregate_bboxes).sum(dim=0)
        score = scores[0]
        label = labels[0]

        # delete the aggregated predictions from the remaining.
        inds = torch.nonzero(mask)[:, 0]
        mask[inds[~iou_mask]] = False
        remainings = remainings[~mask]

        return bbox, score, label, remainings, keep_idx

    score_factors = nms_cfg.pop('score_factor', None)
    score_thr = nms_cfg.pop('score_threshold', 0)
    # skip nms when nms_cfg is None
    if nms_cfg is None:
        scores, inds = scores.sort(descending=True)
        bboxes = bboxes[inds]
        return torch.cat([bboxes, scores[:, None]], -1), inds

    # num_classes = scores.size(1)
    # # exclude background category
    # if bboxes.shape[1] > 2:
    #     bboxes = bboxes.view(scores.size(0), -1, 2)
    # else:
    #     bboxes = bboxes[:, None].expand(-1, num_classes, 2)

    # filter out segments with low scores
    if score_factors is not None:
        scores = scores * score_factors[:, None]
    valid_mask = scores > score_thr
    bboxes = bboxes[valid_mask]
    scores = scores[valid_mask]
    # labels = valid_mask.nonzero(as_tuple=False)[:, 1]
    labels = labels[valid_mask]

    if bboxes.numel() == 0:
        bboxes = bboxes.new_zeros((0, 3))
        labels = bboxes.new_zeros((0,), dtype=torch.long)
        return bboxes, labels

    remainings = scores.argsort(descending=True)

    max_num = nms_cfg.get('max_num', -1)
    iou_thr = nms_cfg.get('iou_thr')

    results = []
    while remainings.numel() > 0:
        bbox, score, label, remainings, keep_idx = _nmw1d(bboxes, scores, labels, remainings)
        results.append([bbox, score, label, keep_idx])
        if max_num > 0 and len(results) == max_num:
            break

    if len(results) == 0:
        bboxes = bboxes.new_zeros((0, 2))
        scores = scores.new_zeros((0,))
        labels = labels.new_zeros((0,))
        keep = labels.new_zeros((0,))
    else:
        bboxes, scores, labels, keep = list(zip(*results))
        bboxes = torch.stack(bboxes)
        scores = torch.stack(scores)
        labels = torch.stack(labels)
        keep = torch.stack(keep)
    dets = torch.cat([bboxes, scores[:, None]], dim=-1)

    return dets, keep
