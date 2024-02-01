import torch.nn
from mmdet.registry import MODELS


@MODELS.register_module()
class PseudoBackbone(torch.nn.Module):

    def __init__(self, multi_scale=True, *args, **kwargs):
        self.multi_scale = multi_scale
        super().__init__(*args, **kwargs)

    def forward(self, x):
        if self.multi_scale:
            return [x]   # mimic the multi-scale feature
        else:
            return x
