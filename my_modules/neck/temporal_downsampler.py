import torch.nn as nn
from mmcv.cnn import ConvModule
from mmdet.registry import MODELS
from torch.nn import functional as F


@MODELS.register_module()
class DownSampler1D(nn.Module):
    """Temporal Down-Sampling Module."""

    def __init__(self,
                 num_levels=4,
                 in_channels=2048,
                 out_channels=512,
                 kernel_sizes=(1, 3),
                 strides=(1, 2),
                 paddings=(0, 1),
                 out_indices=(0, 1, 2, 3),
                 mask=False,
                 ):
        super(DownSampler1D, self).__init__()

        self.in_channels = in_channels
        self.num_levels = num_levels
        self.kernel_sizes = kernel_sizes
        self.strides = strides
        self.paddings = paddings
        self.out_channels = out_channels
        self.out_indices = out_indices
        self.mask = mask

        td_layers = []
        for i in range(self.num_levels - 1):
            td_layers.append(ConvModule(in_channels,
                                        out_channels,
                                        kernel_sizes,
                                        strides,
                                        paddings,
                                        conv_cfg=dict(type='Conv2d'),
                                        norm_cfg=dict(type='SyncBN'),
                                        act_cfg=dict(type='ReLU')))
            in_channels = out_channels
        self.td_layers = nn.Sequential(*td_layers)

    # def init_weights(self):
    #     """Initiate the parameters."""
    #     for m in self.modules():
    #         if isinstance(m, _ConvNd):
    #             kaiming_init(m)
    #         elif isinstance(m, _BatchNorm):
    #             constant_init(m, 1)

    # def train(self, mode=True):
    #     """Set the optimization status when training."""
    #     super().train(mode)
    #
    #     if mode:
    #         for m in self.modules():
    #             if isinstance(m, _BatchNorm):
    #                 m.eval()

    def forward(self, x):
        # x: N, C, 1, T

        if self.mask:
            B, C, _, T = x.size()
            mask = x.new_zeros((B, 1, T))
            for feat_id, feat in enumerate(x):
                # find valid feature length for each sample in the batch
                # feat is of shape (C, T), we find the tails that are all zeros and take it as the padding
                valid_feat_len = T - (feat == 0).all(dim=0).sum()
                mask[feat_id, :, :, :valid_feat_len] = 1

        outs = []
        if 0 in self.out_indices:
            outs.append(x)

        for i, layer_name in enumerate(self.td_layers):
            x = self.td_layers[i](x)
            if self.mask:
                if self.td_layers[i].stride[0] > 1:
                    # downsample the mask using nearest neighbor
                    mask = F.interpolate(mask, size=x.size(-1), mode='nearest')

                # masking out the features, stop grad to mask
                x = x * mask.detach()
            if (i + 1) in self.out_indices:
                outs.append(x)
        return tuple(outs)
