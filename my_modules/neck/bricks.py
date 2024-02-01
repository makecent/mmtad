from functools import wraps

import torch.nn as nn
from mmcv.cnn import ConvModule
from mmdet.registry import MODELS


def enable_batch_processing(module_cls):
    original_forward = module_cls.forward

    @wraps(original_forward)
    def modified_forward(self, x):
        if isinstance(x, (list, tuple)):
            return [original_forward(self, item) for item in x]
        else:
            return original_forward(self, x)

    module_cls.forward = modified_forward
    return module_cls


MODELS.register_module(module=enable_batch_processing(nn.AdaptiveAvgPool3d))
MODELS.register_module(module=enable_batch_processing(nn.Flatten))
MODELS.register_module(module=enable_batch_processing(nn.Unflatten))


@MODELS.register_module()
class TemporalDownSampler(nn.Module):
    """Temporal Down-Sampling Module."""

    def __init__(self,
                 num_levels=4,
                 in_channels=2048,
                 out_channels=512,
                 conv_type='Conv3d',
                 kernel_sizes=(3, 3, 3),
                 strides=(2, 1, 1),
                 paddings=(1, 1, 1),
                 out_indices=(0, 1, 2, 3)):
        super(TemporalDownSampler, self).__init__()

        self.in_channels = in_channels
        self.num_levels = num_levels
        self.kernel_sizes = kernel_sizes
        self.strides = strides
        self.paddings = paddings
        self.out_channels = out_channels
        self.out_indices = out_indices
        self.conv_type = conv_type

        td_layers = []
        for i in range(self.num_levels - 1):
            td_layers.append(ConvModule(in_channels,
                                        out_channels,
                                        kernel_sizes,
                                        strides,
                                        paddings,
                                        conv_cfg=dict(type=conv_type),
                                        norm_cfg=dict(type='SyncBN'),
                                        act_cfg=dict(type='ReLU')))
            in_channels = out_channels
        self.td_layers = nn.Sequential(*td_layers)

    def forward(self, x):
        assert x.size(-1) >= 2 ** self.num_levels, (f"The temporal length of input {x.size(-1)} is too short"
                                                    f" for {self.num_levels} levels of down-sampling")
        outs = []
        if 0 in self.out_indices:
            outs.append(x)

        for i, layer_name in enumerate(self.td_layers):
            x = self.td_layers[i](x)
            if (i + 1) in self.out_indices:
                outs.append(x)

        return tuple(outs)
