# Adapted from https://github.com/MCG-NJU/BasicTAD/
# https://github.com/open-mmlab/mmcv or
# https://github.com/open-mmlab/mmdetection

import torch.nn as nn
from mmdet.registry import MODELS
from mmdet.models.dense_heads import RetinaHead
from typing import Tuple

import torch
from torch import Tensor

from mmdet.structures import SampleList
from mmdet.utils import InstanceList


@MODELS.register_module()
class RetinaHead1D(RetinaHead):
    r"""Modified RetinaHead to support 1D
    """

    def _init_layers(self):
        super()._init_layers()
        # Change the cls head and reg head to be based on Conv1d instead of Conv2d
        self.retina_cls = nn.Conv1d(
            self.feat_channels,
            self.num_base_priors * self.cls_out_channels,
            3,
            padding=1)
        self.retina_reg = nn.Conv1d(
            self.feat_channels, self.num_base_priors * 2, 3, padding=1)

    def forward_single(self, x):
        cls_score, bbox_pred = super().forward_single(x)
        # add pseudo H dimension
        cls_score, bbox_pred = cls_score.unsqueeze(-2), bbox_pred.unsqueeze(-2)
        # bbox_pred = [N, 2], where 2 is the x, w. Now adding pseudo y, h
        bbox_pred = bbox_pred.unflatten(1, (self.num_base_priors, -1))
        y, h = torch.split(torch.zeros_like(bbox_pred), 1, dim=2)
        bbox_pred = torch.cat((bbox_pred[:, :, :1, :, :], y, bbox_pred[:, :, 1:, :, :], h), dim=2)
        bbox_pred = bbox_pred.flatten(start_dim=1, end_dim=2)
        return cls_score, bbox_pred

    def predict(self,
                x: Tuple[Tensor],
                batch_data_samples: SampleList,
                rescale: bool = False) -> InstanceList:
        """Perform forward propagation of the detection head and predict
        detection results on the features of the upstream network.

        Args:
            x (tuple[Tensor]): Multi-level features from the
                upstream network, each is a 4D-tensor.
            batch_data_samples (List[:obj:`DetDataSample`]): The Data
                Samples. It usually includes information such as
                `gt_instance`, `gt_panoptic_seg` and `gt_sem_seg`.
            rescale (bool, optional): Whether to rescale the results.
                Defaults to False.

        Returns:
            list[obj:`InstanceData`]: Detection results of each image
            after the post process.
        """
        batch_img_metas = [
            data_samples.metainfo for data_samples in batch_data_samples
        ]

        outs = self(x)
        with_nms = True if 'nms' in self.test_cfg else False
        predictions = self.predict_by_feat(*outs,
                                           batch_img_metas=batch_img_metas,
                                           with_nms=with_nms,
                                           rescale=rescale)
        return predictions
