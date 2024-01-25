import torch
import torch.nn as nn

from mmdet.registry import MODELS


def crops_to_batch(forward_methods):
    def wrapper(self, inputs, *args, **kwargs):
        if inputs.ndim == 6:
            num_crops = inputs.shape[1]
            inputs = inputs.view(-1, *inputs.shape[2:])
        return forward_methods(self, inputs, *args, **kwargs)

    return wrapper


@MODELS.register_module()
class SlowOnly(nn.Module):

    def __init__(self,
                 out_indices=(4,),
                 freeze_bn=True,
                 freeze_bn_affine=True
                 ):
        super(SlowOnly, self).__init__()
        model = torch.hub.load("facebookresearch/pytorchvideo", model='slow_r50', pretrained=True)
        self.blocks = model.blocks[:-1]  # exclude the last HEAD block
        self.out_indices = out_indices
        self._freeze_bn = freeze_bn
        self._freeze_bn_affine = freeze_bn_affine

    @crops_to_batch
    def forward(self, x):
        outs = []
        for i, block in enumerate(self.blocks):
            x = block(x)
            if i in self.out_indices:
                outs.append(x)
        if len(outs) == 1:
            return outs[0]
        return outs

    def train(self, mode=True):
        super(SlowOnly, self).train(mode)
        if self._freeze_bn and mode:
            for name, m in self.named_modules():
                if isinstance(m, (nn.BatchNorm3d, nn.BatchNorm2d, nn.BatchNorm1d)):
                    m.eval()
                    if self._freeze_bn_affine:
                        m.weight.register_hook(lambda grad: torch.zeros_like(grad))
                        m.bias.register_hook(lambda grad: torch.zeros_like(grad))


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
