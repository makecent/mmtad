import torch.nn as nn
from mmdet.models.dense_heads import DeformableDETRHead
from mmdet.registry import MODELS

from my_modules.layers.pseudo_layers import Pseudo4DRegLinear


@MODELS.register_module()
class TadTRHead(DeformableDETRHead):
    """
    TadTR head.
    We modify the regression branches to output 2 (x1, x2) rather than 4 (x1, y1, x2, y2).
    """

    def _init_layers(self) -> None:
        """Change the regression output dimension from 4 to 2"""
        super()._init_layers()
        for reg_branch in self.reg_branches:
            reg_branch[-1] = Pseudo4DRegLinear(self.embed_dims)

    def init_weights(self) -> None:
        super().init_weights()
        nn.init.constant_(self.reg_branches[0][-1].bias.data[1:], -2.0)  # [2:] -> [1:]
        if self.as_two_stage:
            for m in self.reg_branches:
                nn.init.constant_(m[-1].bias.data[1:], 0.0)  # [2:] -> [1:]
