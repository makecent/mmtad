import torch.nn as nn
from mmcv.cnn import ConvModule
from mmdet.registry import MODELS
from torch.nn import functional as F


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
                 out_indices=(0, 1, 2, 3),
                 pool_position=None,
                 ):
        super(TemporalDownSampler, self).__init__()

        self.in_channels = in_channels
        self.num_levels = num_levels
        self.kernel_sizes = kernel_sizes
        self.strides = strides
        self.paddings = paddings
        self.out_channels = out_channels
        self.out_indices = out_indices
        self.conv_type = conv_type
        self.pool_position = pool_position

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
        if self.pool_position is not None:
            self.pool = nn.AdaptiveAvgPool3d((None, 1, 1))

    def forward(self, x):
        if self.pool_position == 'before':
            # x: N, C, T, H, W -> N, C, T
            x = self.pool(x).flatten(start_dim=2)

        outs = []
        if 0 in self.out_indices:
            outs.append(x)

        for i, layer_name in enumerate(self.td_layers):
            x = self.td_layers[i](x)
            if (i + 1) in self.out_indices:
                outs.append(x)
        if self.pool_position == 'after':
            outs = [self.pool(i).flatten(start_dim=2) for i in outs]
        # N, C, T -> N, C, 1, T (mimic the NCHW)
        outs = [i.unsqueeze(-2) for i in outs]
        return tuple(outs)
