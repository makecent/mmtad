import copy
from collections import OrderedDict
from typing import Optional, Sequence, Union, List

import numpy as np
import torch
from mmdet.registry import METRICS
from mmcv.ops import batched_nms
from mmdet.evaluation.functional import eval_map
from mmdet.structures.bbox import bbox_overlaps
from mmengine.evaluator import BaseMetric
from mmengine.logging import MMLogger
from mmengine.structures import InstanceData

from my_modules.task_modules.segments_ops import bbox_voting


@METRICS.register_module()
class TadMetric(BaseMetric):
    """
    This metric differ from the MMDetection metrics: this metric could perform post-processing.
    Specifically, the MMDetection metrics take as input the processed results, by assuming that the post-processing
    was already done in the model's detection head during inference.
    We support post-processing in the Metric because TAD models often split the test video into overlapped sub-videos
    and process them separately.
    Thus, we merge the detection results of sub-videos from the same video and then perform post-processing globally.
    Consequently, this metric receive arguments about post-processing, include:
        (a) the config of NMS (nms_cfg),
        (b) the score threshold for filtering poor detections (score_thr),
        (c) the duration threshold for filtering short detections (duration_thr),
        (d) the maximum number of detections in each video (max_per_video),
        (e) whether merge the results of windows, i.e., sub-videos of the same video, (merge_windows)
    In addition to the above common post-processing arguments, we add two extra arguments:
        (d) the duration threshold for filtering short detections (duration_thr),
        (e) the flag indicating whether to perform NMS on overlapped regions in testing videos (nms_in_overlap),
        (f) the config of segment voting (voting_cfg)
    Example usage:
    For ActionFormer, as it input the entire test video features to the model, no post-processing is needed here, but
    all post-processing should be conducted in the model's detection head. merge_windows=False
    For BasicTAD, as it process cropped windows of test videos, merge_windows=True and NMS config should be set.
    """
    default_prefix: Optional[str] = 'tad'

    def __init__(self,
                 iou_thrs: Union[float, List[float]] = 0.5,
                 nms_cfg=None,
                 max_per_video: int = -1,
                 score_thr=0.0,
                 duration_thr=0.0,
                 nms_in_overlap=False,
                 voting_cfg=None,
                 merge_windows: bool = False,
                 **kwargs):
        super().__init__(**kwargs)
        self.iou_thrs = [iou_thrs] if isinstance(iou_thrs, float) \
            else iou_thrs
        self.nms_cfg = nms_cfg
        self.max_per_video = max_per_video
        self.score_thr = score_thr
        self.duration_thr = duration_thr
        self.nms_in_overlap = nms_in_overlap
        self.voting_cfg = voting_cfg
        self.merge_windows = merge_windows

    def process(self, data_batch: dict, data_samples: Sequence[dict]) -> None:
        """Process one batch of data samples and predictions. The processed
        results should be stored in ``self.results``, which will be used to
        compute the metrics when all batches have been processed.

        Args:
            data_batch (dict): A batch of data from the dataloader.
            data_samples (Sequence[dict]): A batch of data samples that
                contain annotations and predictions.
        """
        for data_sample in data_samples:
            data = copy.deepcopy(data_sample)
            gts, dets = data['gt_instances'], data['pred_instances']
            gts_ignore = data.get('ignored_instances', dict())
            ann = dict(
                video_name=data['video_name'],  # for the purpose of future grouping detections of same video.
                labels=gts['labels'].cpu().numpy(),
                bboxes=gts['bboxes'].cpu().numpy(),
                bboxes_ignore=gts_ignore.get('bboxes', torch.empty((0, 4))).cpu().numpy(),
                labels_ignore=gts_ignore.get('labels', torch.empty(0, )).cpu().numpy())

            ann['overlap'] = data.get('overlap', np.empty((0, 4)))  # for the purpose of NMS on overlapped region

            # Convert the format of segment predictions
            # 1. Add window-offset back to convert the results from window-based to video(frames/features)-based
            # 2. Multiply the frame intervals between adjacent data points so convert detection results to frame-unit.
            # 3. Multiply the FPS to convert detection results to second-unit.
            if 'feat_stride' in data:
                dets['bboxes'] *= data['feat_stride']
            else:
                dets['bboxes'] *= data['frame_interval']
            dets['bboxes'] /= data['fps']
            if 'window_offset' in data:
                dets['bboxes'] += data['window_offset']
            # Set y1, y2 of predictions the fixed value.
            dets['bboxes'][:, 1] = 0.1
            dets['bboxes'][:, 3] = 0.9

            # Filter out predictions with low scores
            valid_inds = dets['scores'] > self.score_thr

            # Filter out predictions with short duration
            valid_inds &= (dets['bboxes'][:, 2] - dets['bboxes'][:, 0]) > self.duration_thr

            dets['bboxes'] = dets['bboxes'][valid_inds].cpu()
            dets['scores'] = dets['scores'][valid_inds].cpu()
            dets['labels'] = dets['labels'][valid_inds].cpu()

            # Format predictions to InstanceData
            dets = InstanceData(**dets)

            self.results.append((ann, dets))

    def compute_metrics(self, results: list) -> dict:
        """Compute the metrics from processed results.

        Args:
            results (list): The processed results of each batch.

        Returns:
            dict: The computed metrics. The keys are the names of the metrics,
            and the values are corresponding results.
        """
        logger: MMLogger = MMLogger.get_current_instance()
        gts, preds = zip(*results)

        if self.merge_windows:
            logger.info(f'\n Concatenating the testing results ...')
            gts, preds = self.merge_results_of_same_video(gts, preds)
        if self.nms_cfg is not None:
            logger.info(f'\n Performing NMS ...')
            preds = self.non_maximum_suppression(preds)

        preds = self.prepare_for_eval(preds)

        eval_results = OrderedDict()
        mean_aps = []
        for iou_thr in self.iou_thrs:
            logger.info(f'\n{"-" * 15}iou_thr: {iou_thr}{"-" * 15}')
            mean_ap, _ = eval_map(
                preds,
                gts,
                iou_thr=iou_thr,
                dataset=self.dataset_meta['classes'],
                logger=logger)
            mean_aps.append(mean_ap)
            eval_results[f'AP{int(iou_thr * 100):02d}'] = round(mean_ap, 3)
        eval_results['mAP'] = sum(mean_aps) / len(mean_aps)
        eval_results.move_to_end('mAP', last=False)
        return eval_results

    @staticmethod
    def merge_results_of_same_video(gts, preds):
        # Merge prediction results from the same videos because we use sliding windows to crop the testing videos
        # Also known as the Cross-Window Fusion (CWF)
        video_names = list(dict.fromkeys([gt['video_name'] for gt in gts]))

        # list of dict to dict of list
        merged_gts_dict = dict()
        merged_preds_dict = dict()
        for this_gt, this_pred in zip(gts, preds):
            video_name = this_gt.pop('video_name')
            # Computer the mask indicating that if a prediction is in the overlapped regions.
            overlap_regions = this_gt.pop('overlap')
            if overlap_regions.size == 0:
                this_pred.in_overlap = np.zeros(this_pred.bboxes.shape[0], dtype=bool)
            else:
                this_pred.in_overlap = bbox_overlaps(this_pred.bboxes, torch.from_numpy(overlap_regions)) > 0

            merged_preds_dict.setdefault(video_name, []).append(this_pred)
            merged_gts_dict.setdefault(video_name, this_gt)  # the gt is video-wise thus no need concatenation

        # dict of list to list of dict
        merged_gts = []
        merged_preds = []
        for video_name in video_names:
            merged_gts.append(merged_gts_dict[video_name])
            # Concatenate detection in windows of the same video
            merged_preds.append(InstanceData.cat(merged_preds_dict[video_name]))
        return merged_gts, merged_preds

    def non_maximum_suppression(self, preds):
        preds_nms = []
        for pred_v in preds:
            if self.nms_cfg is not None:
                if self.nms_in_overlap:
                    if pred_v.in_overlap.sum() > 1:
                        # Perform NMS on predictions inside overlapped regions of windows
                        # in_overlap is a binary matrix of shape [num_of_preds, num_of_overlap_regions]
                        pred_in_overlaps = []
                        for i in range(pred_v.in_overlap.shape[1]):
                            _pred = pred_v[pred_v.in_overlap[:, i]]
                            if len(_pred) == 0:
                                continue
                            bboxes_scores, keep_idxs = batched_nms(_pred.bboxes,
                                                                   _pred.scores,
                                                                   _pred.labels,
                                                                   nms_cfg=self.nms_cfg)
                            bboxes = bboxes_scores[:, :-1]
                            scores = bboxes_scores[:, -1]
                            labels = _pred.labels[keep_idxs]
                            if self.voting_cfg is not None and len(bboxes) > 0:
                                bboxes = bbox_voting(bboxes, labels, _pred.bboxes, _pred.scores,
                                                     len(self.dataset_meta['classes']), _pred.labels,
                                                     iou_thr=self.voting_cfg.get('iou_thr', 0.01),
                                                     score_thr=self.voting_cfg.get('score_thr', 0))
                            _pred = InstanceData(bboxes=bboxes, scores=scores, labels=labels)
                            pred_in_overlaps.append(_pred)
                        pred_not_in_overlaps = pred_v[~pred_v.in_overlap.max(-1)[0]]
                        pred_not_in_overlaps.pop('in_overlap')
                        pred_v = InstanceData.cat(pred_in_overlaps + [pred_not_in_overlaps])
                else:
                    bboxes_scores, keep_idxs = batched_nms(pred_v.bboxes,
                                                           pred_v.scores,
                                                           pred_v.labels,
                                                           nms_cfg=self.nms_cfg)
                    bboxes = bboxes_scores[:, :-1]
                    scores = bboxes_scores[:, -1]
                    labels = pred_v.labels[keep_idxs]
                    if self.voting_cfg is not None and len(bboxes) > 0:
                        bboxes = bbox_voting(bboxes, labels, pred_v.bboxes, pred_v.scores,
                                             len(self.dataset_meta['classes']), pred_v.labels,
                                             iou_thr=self.voting_cfg.get('iou_thr', 0.01),
                                             score_thr=self.voting_cfg.get('score_thr', 0))
                    pred_v = InstanceData(bboxes=bboxes, scores=scores, labels=labels)
            sort_idxs = pred_v.scores.argsort(descending=True)
            pred_v = pred_v[sort_idxs]
            # keep top-k predictions
            if self.max_per_video > 0:
                pred_v = pred_v[:self.max_per_video]
            preds_nms.append(pred_v)
        return preds_nms

    def prepare_for_eval(self, preds):
        """Reformat predictions to meet the requirement of eval_map function: VideoList[ClassList[PredictionArray]]"""
        out_preds = []
        for pred_v in preds:
            dets = []
            for label in range(len(self.dataset_meta['classes'])):
                index = np.where(pred_v.labels == label)[0]
                pred_bbox_with_scores = np.hstack(
                    [pred_v[index].bboxes, pred_v[index].scores.reshape((-1, 1))])
                dets.append(pred_bbox_with_scores)
            out_preds.append(dets)
        return out_preds
