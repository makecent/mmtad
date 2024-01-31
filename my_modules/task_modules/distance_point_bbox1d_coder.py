# Copyright (c) OpenMMLab. All rights reserved.
from typing import Optional, Sequence, Union

from mmdet.models.task_modules.coders import DistancePointBBoxCoder
from mmdet.registry import TASK_UTILS
from mmdet.structures.bbox import (BaseBoxes, HorizontalBoxes, bbox2distance,
                                   distance2bbox, get_box_tensor)
from torch import Tensor


@TASK_UTILS.register_module()
class DistancePointBBox1dCoder(DistancePointBBoxCoder):
    """Distance Point BBox coder that always set y1 and y2 to 0.1 and 0.9 when decoding.
    """

    def decode(self, *args, **kwargs):
        bboxes = super().decode(*args, **kwargs)
        if isinstance(bboxes, Tensor):
            bboxes[:, 1] = 0.1
            bboxes[:, 3] = 0.9
        else:
            assert isinstance(bboxes, BaseBoxes)
            bboxes.tensor[:, 1] = 0.1
            bboxes.tensor[:, 3] = 0.9
        return bboxes