import torch
import torch.nn as nn
from mmdet.models.dense_heads import DETRHead
from mmdet.registry import MODELS, TASK_UTILS
from mmdet.utils import (ConfigType, OptMultiConfig)
from my_modules.layers.pseudo_layers import Pseudo4DRegLinear
from mmengine.model import bias_init_with_prob, constant_init

@MODELS.register_module()
class DETR_TADHead(DETRHead):
    """
    TadTR head.
    We modify the regression branches to output 2 (x1, x2) rather than 4 (x1, y1, x2, y2).
    """

    def __init__(
            self,
            num_classes: int,
            embed_dims: int = 256,
            num_reg_fcs: int = 2,
            sync_cls_avg_factor: bool = False,
            loss_cls: ConfigType = dict(
                type='CrossEntropyLoss',
                bg_cls_weight=0.1,
                use_sigmoid=False,
                loss_weight=1.0,
                class_weight=1.0),
            loss_bbox: ConfigType = dict(type='L1Loss', loss_weight=5.0),
            loss_iou: ConfigType = dict(type='GIoU1dLoss', loss_weight=2.0),
            train_cfg: ConfigType = dict(
                assigner=dict(
                    type='HungarianAssigner',
                    match_costs=[
                        dict(type='ClassificationCost', weight=1.),
                        dict(type='BBox1dL1Cost', weight=5.0, box_format='xywh'),
                        dict(type='IoU1dCost', iou_mode='giou', weight=2.0)
                    ])),
            test_cfg: ConfigType = dict(max_per_img=200),
            init_cfg: OptMultiConfig = None) -> None:
        super(DETRHead, self).__init__(init_cfg=init_cfg)
        self.bg_cls_weight = 0
        self.sync_cls_avg_factor = sync_cls_avg_factor
        class_weight = loss_cls.get('class_weight', None)
        if class_weight is not None and (self.__class__ is DETR_TADHead):
            assert isinstance(class_weight, float), 'Expected ' \
                                                    'class_weight to have type float. Found ' \
                                                    f'{type(class_weight)}.'
            # NOTE following the official DETR repo, bg_cls_weight means
            # relative classification weight of the no-object class.
            bg_cls_weight = loss_cls.get('bg_cls_weight', class_weight)
            assert isinstance(bg_cls_weight, float), 'Expected ' \
                                                     'bg_cls_weight to have type float. Found ' \
                                                     f'{type(bg_cls_weight)}.'
            class_weight = torch.ones(num_classes + 1) * class_weight
            # set background class as the last indice
            class_weight[num_classes] = bg_cls_weight
            loss_cls.update({'class_weight': class_weight})
            if 'bg_cls_weight' in loss_cls:
                loss_cls.pop('bg_cls_weight')
            self.bg_cls_weight = bg_cls_weight

        if train_cfg:
            assert 'assigner' in train_cfg, 'assigner should be provided ' \
                                            'when train_cfg is set.'
            assigner = train_cfg['assigner']
            self.assigner = TASK_UTILS.build(assigner)
            if train_cfg.get('sampler', None) is not None:
                raise RuntimeError('DETR do not build sampler.')
        self.num_classes = num_classes
        self.embed_dims = embed_dims
        self.num_reg_fcs = num_reg_fcs
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self.loss_cls = MODELS.build(loss_cls)
        self.loss_bbox = MODELS.build(loss_bbox)
        self.loss_iou = MODELS.build(loss_iou)

        if self.loss_cls.use_sigmoid:
            self.cls_out_channels = num_classes
        else:
            self.cls_out_channels = num_classes + 1

        self._init_layers()

    def _init_layers(self) -> None:
        """Initialize layers of the transformer head."""
        super()._init_layers()
        self.fc_reg = Pseudo4DRegLinear(self.embed_dims, delta=False)

    # def init_weights(self) -> None:
    #     """Initialize weights of the Deformable DETR head."""
    #     if self.loss_cls.use_sigmoid:
    #         bias_init = bias_init_with_prob(0.01)
    #         nn.init.constant_(self.fc_cls.bias, bias_init)
    #     constant_init(self.fc_reg, 0, bias=0)
