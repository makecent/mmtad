# from typing import Dict, List
#
# import torch.nn as nn
# from mmdet.registry import MODELS
# from mmdet.structures import SampleList
# from mmdet.models.dense_heads import DeformableDETRHead, DINOHead
# from mmdet.utils import InstanceList, OptInstanceList
# from torch import Tensor
#
# from my_modules.layers.pseudo_layers import Pseudo4DRegLinear
# from my_modules.head import CustomDINOHead
#
#
# @MODELS.register_module()
# class MyDeformableDETRHead(CustomDINOHead):
#     """
#     Customized DINO Head to support Temporal Action Detection.
#     1. We modify the regression branches to remove the unused FC nodes (x1, y1, x2, y2) -> (x1, x2).
#     Note that this modification is optional since we have already modified the loss functions to
#     make sure that the y1, y2 will not contribute to the loss and cost. See my_modules/loss/custom_loss.py
#     2. Modify the loss function to support Position-supervised Focal Loss (Stable-DINO).
#     3. Support de-noising.
#     """
#
#     def _init_layers(self) -> None:
#         """Change the regression output dimension from 4 to 2"""
#         super()._init_layers()
#         for reg_branch in self.reg_branches:
#             reg_branch[-1] = Pseudo4DRegLinear(self.embed_dims)
#
#     def init_weights(self) -> None:
#         super().init_weights()
#         nn.init.constant_(self.reg_branches[0][-1].bias.data[1:], -2.0)  # [2:] -> [1:]
#         if self.as_two_stage:
#             for m in self.reg_branches:
#                 nn.init.constant_(m[-1].bias.data[1:], 0.0)  # [2:] -> [1:]
