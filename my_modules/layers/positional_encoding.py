import torch
from mmdet.models.layers import SinePositionalEncoding
from mmdet.registry import MODELS
from torch import Tensor


# class PositionEmbeddingSine(nn.Module):
#     """
#     This is a more standard version of the position embedding, very similar to the one
#     used by the Attention is all you need paper, generalized to work on videos.
#     """
#
#     def __init__(self, num_feats=128, temperature=10000, normalize=False, offset=0, scale=None):
#         super().__init__()
#         self.num_feats = num_feats
#         self.temperature = temperature
#         self.normalize = normalize
#         if scale is not None and normalize is False:
#             raise ValueError("normalize should be True if scale is passed")
#         if scale is None:
#             scale = 2 * math.pi
#         self.scale = scale
#
#     def forward(self, mask):
#         # bs, 1, w -> bs, w
#         mask = mask.squeeze(dim=1)
#         not_mask = ~mask
#         x_embed = not_mask.cumsum(1, dtype=torch.float32)  # N x T
#         if self.normalize:
#             eps = 1e-6
#             x_embed = x_embed / (x_embed[:, -1:] + eps) * self.scale
#
#         dim_t = torch.arange(self.num_feats * 2, dtype=torch.float32, device=mask.device)
#         dim_t = self.temperature ** (2 * (dim_t // 2) / (self.num_feats * 2))
#
#         pos_x = x_embed[:, :, None] / dim_t  # N x T x C
#         # n,c,t
#         pos_x = torch.stack((pos_x[:, :, 0::2].sin(), pos_x[:, :, 1::2].cos()), dim=3).flatten(2)
#         pos = pos_x.permute(0, 2, 1)  # N x C x T
#         return pos.unsqueeze(dim=2)
#
#
# def build_position_encoding(args):
#     feat_dim = args.hidden_dim
#     if args.position_embedding in ('v2', 'sine'):
#         position_embedding = PositionEmbeddingSine(feat_dim, normalize=True)
#     else:
#         raise ValueError(f"not supported {args.position_embedding}")
#
#     return position_embedding
# class SinePositionalEncoding(BaseModule):
@MODELS.register_module()
class CustomSinePositionalEncoding(SinePositionalEncoding):
    """Customized SinePositionalEncoding to support 1D sequence.
    In 2D case, the dimension of positional encoding (num_feats) must be the half of the feature dimension,
    so that the x and y positional encoding can be concatenated to have the same dimension as the feature.
    In 1D case, to be compatible with the original repository, the self.num_feats is still the half of the
    feature dimenstion, whereas in the forward() function, the real num_feats is equal to self.num_feats * 2.
    """

    def forward(self, mask: Tensor) -> Tensor:
        """Forward function for `SinePositionalEncoding`.

        Args:
            mask (Tensor): ByteTensor mask. Non-zero values representing
                ignored positions, while zero values means valid positions
                for this image. Shape [bs, h, w].

        Returns:
            pos (Tensor): Returned position embedding with shape
                [bs, num_feats*2, h, w].
        """
        # For convenience of exporting to ONNX, it's required to convert
        # `masks` from bool to int.
        # Since our SinePositionalEncoding is used for 1D sequence, we multiply the number of feats by two
        num_feats = self.num_feats * 2
        mask = mask.to(torch.int)
        not_mask = 1 - mask  # logical_not
        # y_embed = not_mask.cumsum(1, dtype=torch.float32)
        x_embed = not_mask.cumsum(2, dtype=torch.float32)
        if self.normalize:
            # y_embed = (y_embed + self.offset) / \
            #           (y_embed[:, -1:, :] + self.eps) * self.scale
            x_embed = (x_embed + self.offset) / \
                      (x_embed[:, :, -1:] + self.eps) * self.scale
        dim_t = torch.arange(
            num_feats, dtype=torch.float32, device=mask.device)
        dim_t = self.temperature ** (2 * (dim_t // 2) / num_feats)
        pos_x = x_embed[:, :, :, None] / dim_t
        # pos_y = y_embed[:, :, :, None] / dim_t
        # use `view` instead of `flatten` for dynamically exporting to ONNX
        B, H, W = mask.size()  # H=1
        pos_x = torch.stack(
            (pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()),
            dim=4).view(B, H, W, -1)
        # pos_y = torch.stack(
        #     (pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()),
        #     dim=4).view(B, H, W, -1)
        pos = pos_x.permute(0, 3, 1, 2)
        return pos
