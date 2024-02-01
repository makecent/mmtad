from typing import Optional

import torch
from mmdet.models.layers import SinePositionalEncoding
from mmdet.registry import MODELS
from torch import Tensor


@MODELS.register_module()
class SinePositional1dEncoding(SinePositionalEncoding):
    """Customized SinePositionalEncoding to support 1D sequence.
    In 2D case, the dimension of positional encoding (num_feats) must be the half of the feature dimension,
    so that the x and y positional encoding can be concatenated to have the same dimension as the feature.
    In 1D case, to be compatible with the original repository, the self.num_feats is still the half of the
    feature dimension, whereas in the forward() function, the produced num_feats is equal to self.num_feats * 2.
    """

    def forward(self, mask: Tensor, input: Optional[Tensor] = None) -> Tensor:
        """Forward function for `SinePositionalEncoding`.

        Args:
            mask (Tensor): ByteTensor mask. Non-zero values representing
                ignored positions, while zero values means valid positions
                for this image. Shape [bs, h, w].
            input (Tensor, optional): Input image/feature Tensor.
                Shape [bs, c, h, w]


        Returns:
            pos (Tensor): Returned position embedding with shape
                [bs, num_feats*2, h, w].
        """
        assert not (mask is None and input is None)

        if mask is not None:
            B, H, W = mask.size()
            device = mask.device
            # For convenience of exporting to ONNX,
            # it's required to convert
            # `masks` from bool to int.
            mask = mask.to(torch.int)
            not_mask = 1 - mask  # logical_not
            # y_embed = not_mask.cumsum(1, dtype=torch.float32)
            x_embed = not_mask.cumsum(2, dtype=torch.float32)
        else:
            # single image or batch image with no padding
            B, _, H, W = input.shape
            device = input.device
            x_embed = torch.arange(
                1, W + 1, dtype=torch.float32, device=device)
            x_embed = x_embed.view(1, 1, -1).repeat(B, H, 1)
            # y_embed = torch.arange(
            #     1, H + 1, dtype=torch.float32, device=device)
            # y_embed = y_embed.view(1, -1, 1).repeat(B, 1, W)

        if self.normalize:
            # y_embed = (y_embed + self.offset) / \
            #           (y_embed[:, -1:, :] + self.eps) * self.scale
            x_embed = (x_embed + self.offset) / \
                      (x_embed[:, :, -1:] + self.eps) * self.scale
        dim_t = torch.arange(
            self.num_feats, dtype=torch.float32, device=device)
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_feats)
        pos_x = x_embed[:, :, :, None] / dim_t
        # pos_y = y_embed[:, :, :, None] / dim_t
        # use `view` instead of `flatten` for dynamically exporting to ONNX

        pos_x = torch.stack(
            (pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()),
            dim=4).view(B, H, W, -1)
        # pos_y = torch.stack(
        #     (pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()),
        #     dim=4).view(B, H, W, -1)
        pos = pos_x.permute(0, 3, 1, 2)
        return pos
