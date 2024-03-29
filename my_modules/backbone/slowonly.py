import torch
import torch.nn as nn
from einops import rearrange
from mmdet.registry import MODELS

torch.fx.wrap('rearrange')


@MODELS.register_module()
class SlowOnly(nn.Module):
    """
    The SlowFast model (The Slow part only) from https://arxiv.org/abs/1812.03982.
    The pre-training image size is (4x224x224), the frame interval is 16. Pre-trained on Kinetics-400.
    The output feature dimension is 4x7x7x2048.
    """

    def __init__(self,
                 out_indices=(4,),
                 freeze_bn=True,
                 freeze_bn_affine=True,
                 freeze=False,
                 ):
        super(SlowOnly, self).__init__()
        model = torch.hub.load("facebookresearch/pytorchvideo", model='slow_r50', pretrained=True)
        self.blocks = model.blocks[:-1]  # exclude the last HEAD block
        self.out_indices = out_indices
        self._freeze_bn = freeze_bn
        self._freeze_bn_affine = freeze_bn_affine
        self._freeze = freeze
        if freeze:
            # Freeze all parameters
            for param in model.parameters():
                param.requires_grad = False
        elif freeze_bn_affine:
            # Freeze only normalization layers
            for module in model.modules():
                if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
                    for param in module.parameters():
                        param.requires_grad = False

    def forward(self, x):
        # The inputs should be (N, M, C, T, H, W), N is the batch size and M = num_crops x num_clips.
        n, m, c, t, h, w = x.shape
        num_crops = 1  # TODO: compatible with dynamic num_crops, e.g. num_crops=3 when ThreeCrop as test augmentation
        num_clips = m // num_crops
        # x: (N, M, C, T, H, W) -> (NxM, C, T, H, W)
        x = rearrange(x, 'n m c t h w -> (n m) c t h w')

        outs = []
        for i, block in enumerate(self.blocks):
            x = block(x)
            if i in self.out_indices:
                # x: (NxM, C', T', H', W') -> (N x num_crops, C', num_clips x T', H', W')
                x_ = rearrange(x, '(n crops clips) c t h w -> (n crops) c (clips t) h w',
                               n=n, crops=num_crops, clips=num_clips)
                outs.append(x_)
        if len(outs) == 1:
            return outs[0]
        return x

    def train(self, mode=True):
        super(SlowOnly, self).train(mode)
        if self._freeze_bn and mode:
            for name, m in self.named_modules():
                if isinstance(m, (nn.BatchNorm3d, nn.BatchNorm2d, nn.BatchNorm1d)):
                    m.eval()

# from fvcore.nn import FlopCountAnalysis, flop_count_table
# from torch.profiler import profile, ProfilerActivity, record_function
#
# imgs = torch.randn(1, 3, 16, 160, 160)
# model = SlowOnly()
# model.eval()
# x3d = torch.hub.load("facebookresearch/pytorchvideo", model='x3d_s', pretrained=True)
# x3d.eval()
#
# flops1 = FlopCountAnalysis(model, imgs)
# flops2 = FlopCountAnalysis(x3d, imgs)
#
# print(flop_count_table(flops1, max_depth=3))
# print(flop_count_table(flops2, max_depth=3))
#
# with profile(activities=[ProfilerActivity.CPU], profile_memory=True, record_shapes=True) as prof1:
#     with record_function("model_inference"):
#         model(torch.randn(1, 3, 16, 112, 112))
# with profile(activities=[ProfilerActivity.CPU], profile_memory=True, record_shapes=True) as prof2:
#     with record_function("x3d_inference"):
#         x3d(torch.randn(1, 3, 16, 160, 160))
#
# print(prof1.key_averages().table(sort_by="cpu_time_total", row_limit=10))
# print(prof2.key_averages().table(sort_by="cpu_time_total", row_limit=10))
