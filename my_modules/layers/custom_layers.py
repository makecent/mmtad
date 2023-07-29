import math
import warnings
from typing import Optional

import mmengine
import torch
import torch.nn as nn
from mmcv.cnn import (build_norm_layer)
from mmcv.cnn.bricks.transformer import FFN, MultiheadAttention
from mmcv.ops import MultiScaleDeformableAttention
from mmcv.ops.multi_scale_deform_attn import MultiScaleDeformableAttnFunction, multi_scale_deformable_attn_pytorch
from mmcv.utils import IS_CUDA_AVAILABLE, IS_MLU_AVAILABLE
from mmdet.models.layers import DeformableDetrTransformerEncoder, DeformableDetrTransformerDecoder, \
    DeformableDetrTransformerDecoderLayer, DeformableDetrTransformerEncoderLayer, DinoTransformerDecoder
from mmdet.models.layers import MLP
from mmengine.model import BaseModule, ModuleList
from mmengine.model import constant_init, xavier_init
from mmengine.utils import deprecated_api_warning
from torch import Tensor

from my_modules.layers.pseudo_layers import Pseudo2DLinear


def zero_y_reference_points(forward_method):
    def wrapper(self, *args, **kwargs):
        if 'reference_points' in kwargs:
            reference_points = kwargs['reference_points'].clone()
            if reference_points.shape[-1] == 2:
                reference_points[..., 1] = 0.5
            elif reference_points.shape[-1] == 4:
                reference_points[..., 1] = 0.5
                reference_points[..., 3] = 0.
            kwargs['reference_points'] = reference_points
        return forward_method(self, *args, **kwargs)

    return wrapper


class CustomMultiScaleDeformableAttention(MultiScaleDeformableAttention):
    """
    Customized DeformableAttention:
    a. sampling_offsets linear layer is changed to output only x offsets and y offsets are fixed to be zeros.
    b. Init the sampling_offsets bias with in 1D style
    c. decorate the forward() function to fix the reference point on y-axis to be zeros.
    """

    def __init__(self,
                 embed_dims: int = 256,
                 num_heads: int = 8,
                 num_levels: int = 4,
                 num_points: int = 4,
                 im2col_step: int = 64,
                 dropout: float = 0.1,
                 batch_first: bool = False,
                 norm_cfg: Optional[dict] = None,
                 init_cfg: Optional[mmengine.ConfigDict] = None,
                 value_proj_ratio: float = 1.0):
        super(MultiScaleDeformableAttention, self).__init__(init_cfg)
        if embed_dims % num_heads != 0:
            raise ValueError(f'embed_dims must be divisible by num_heads, '
                             f'but got {embed_dims} and {num_heads}')
        dim_per_head = embed_dims // num_heads
        self.norm_cfg = norm_cfg
        self.dropout = nn.Dropout(dropout)
        self.batch_first = batch_first

        # you'd better set dim_per_head to a power of 2
        # which is more efficient in the CUDA implementation
        def _is_power_of_2(n):
            if (not isinstance(n, int)) or (n < 0):
                raise ValueError(
                    'invalid input for _is_power_of_2: {} (type: {})'.format(
                        n, type(n)))
            return (n & (n - 1) == 0) and n != 0

        if not _is_power_of_2(dim_per_head):
            warnings.warn(
                "You'd better set embed_dims in "
                'MultiScaleDeformAttention to make '
                'the dimension of each attention head a power of 2 '
                'which is more efficient in our CUDA implementation.')

        self.im2col_step = im2col_step
        self.embed_dims = embed_dims
        self.num_levels = num_levels
        self.num_heads = num_heads
        self.num_points = num_points
        # Change the sampling_offsets layer to only output x offsets. The y offsets are fixed to be zeros.
        self.sampling_offsets = Pseudo2DLinear(
            self.embed_dims, self.num_heads * self.num_levels * self.num_points)
        self.attention_weights = nn.Linear(embed_dims,
                                           num_heads * num_levels * num_points)
        value_proj_size = int(embed_dims * value_proj_ratio)
        self.value_proj = nn.Linear(embed_dims, value_proj_size)
        self.output_proj = nn.Linear(value_proj_size, embed_dims)
        self.init_weights()

    def init_weights(self) -> None:
        """Default initialization for Parameters of Module."""
        constant_init(self.sampling_offsets, 0.)
        device = next(self.parameters()).device
        # Re-init the sampling_offsets bias in 1D style
        thetas = torch.arange(self.num_heads,
                              device=device) * (4 * math.pi / self.num_heads)
        grid_init = thetas.cos()[:, None]
        grid_init = grid_init.view(self.num_heads, 1, 1, 1).repeat(
            1, self.num_levels, self.num_points, 1)
        for i in range(self.num_points):
            grid_init[:, :, i, :] *= i + 1

        self.sampling_offsets.bias.data = grid_init.view(-1)
        constant_init(self.attention_weights, val=0., bias=0.)
        xavier_init(self.value_proj, distribution='uniform', bias=0.)
        xavier_init(self.output_proj, distribution='uniform', bias=0.)
        self._is_init = True

    @zero_y_reference_points
    def forward(self, *args, **kwargs):
        return super().forward(*args, **kwargs)
    # @zero_y_reference_points
    # def forward(self,
    #             query: torch.Tensor,
    #             key: Optional[torch.Tensor] = None,
    #             value: Optional[torch.Tensor] = None,
    #             identity: Optional[torch.Tensor] = None,
    #             query_pos: Optional[torch.Tensor] = None,
    #             key_padding_mask: Optional[torch.Tensor] = None,
    #             reference_points: Optional[torch.Tensor] = None,
    #             spatial_shapes: Optional[torch.Tensor] = None,
    #             level_start_index: Optional[torch.Tensor] = None,
    #             **kwargs) -> torch.Tensor:
    #     """Forward Function of MultiScaleDeformAttention.
    #
    #     Args:
    #         query (torch.Tensor): Query of Transformer with shape
    #             (num_query, bs, embed_dims).
    #         key (torch.Tensor): The key tensor with shape
    #             `(num_key, bs, embed_dims)`.
    #         value (torch.Tensor): The value tensor with shape
    #             `(num_key, bs, embed_dims)`.
    #         identity (torch.Tensor): The tensor used for addition, with the
    #             same shape as `query`. Default None. If None,
    #             `query` will be used.
    #         query_pos (torch.Tensor): The positional encoding for `query`.
    #             Default: None.
    #         key_padding_mask (torch.Tensor): ByteTensor for `query`, with
    #             shape [bs, num_key].
    #         reference_points (torch.Tensor):  The normalized reference
    #             points with shape (bs, num_query, num_levels, 2),
    #             all elements is range in [0, 1], top-left (0,0),
    #             bottom-right (1, 1), including padding area.
    #             or (N, Length_{query}, num_levels, 4), add
    #             additional two dimensions is (w, h) to
    #             form reference boxes.
    #         spatial_shapes (torch.Tensor): Spatial shape of features in
    #             different levels. With shape (num_levels, 2),
    #             last dimension represents (h, w).
    #         level_start_index (torch.Tensor): The start index of each level.
    #             A tensor has shape ``(num_levels, )`` and can be represented
    #             as [0, h_0*w_0, h_0*w_0+h_1*w_1, ...].
    #
    #     Returns:
    #         torch.Tensor: forwarded results with shape
    #         [num_query, bs, embed_dims].
    #     """
    #
    #     if value is None:
    #         value = query
    #
    #     if identity is None:
    #         identity = query
    #     if query_pos is not None:
    #         query[:, -query_pos.shape[1]:, :] += query_pos
    #     if not self.batch_first:
    #         # change to (bs, num_query ,embed_dims)
    #         query = query.permute(1, 0, 2)
    #         value = value.permute(1, 0, 2)
    #
    #     bs, num_query, _ = query.shape
    #     bs, num_value, _ = value.shape
    #     assert (spatial_shapes[:, 0] * spatial_shapes[:, 1]).sum() == num_value
    #
    #     value = self.value_proj(value)
    #     if key_padding_mask is not None:
    #         value = value.masked_fill(key_padding_mask[..., None], 0.0)
    #     value = value.view(bs, num_value, self.num_heads, -1)
    #     sampling_offsets = self.sampling_offsets(query).view(
    #         bs, num_query, self.num_heads, self.num_levels, self.num_points, 2)
    #     attention_weights = self.attention_weights(query).view(
    #         bs, num_query, self.num_heads, self.num_levels * self.num_points)
    #     attention_weights = attention_weights.softmax(-1)
    #
    #     attention_weights = attention_weights.view(bs, num_query,
    #                                                self.num_heads,
    #                                                self.num_levels,
    #                                                self.num_points)
    #     if reference_points.shape[-1] == 2:
    #         offset_normalizer = torch.stack(
    #             [spatial_shapes[..., 1], spatial_shapes[..., 0]], -1)
    #         sampling_locations = reference_points[:, :, None, :, None, :] \
    #                              + sampling_offsets \
    #                              / offset_normalizer[None, None, None, :, None, :]
    #     elif reference_points.shape[-1] == 4:
    #         sampling_locations = reference_points[:, :, None, :, None, :2] \
    #                              + sampling_offsets / self.num_points \
    #                              * reference_points[:, :, None, :, None, 2:] \
    #                              * 0.5
    #     else:
    #         raise ValueError(
    #             f'Last dim of reference_points must be'
    #             f' 2 or 4, but get {reference_points.shape[-1]} instead.')
    #     if ((IS_CUDA_AVAILABLE and value.is_cuda)
    #             or (IS_MLU_AVAILABLE and value.is_mlu)):
    #         output = MultiScaleDeformableAttnFunction.apply(
    #             value, spatial_shapes, level_start_index, sampling_locations,
    #             attention_weights, self.im2col_step)
    #     else:
    #         output = multi_scale_deformable_attn_pytorch(
    #             value, spatial_shapes, sampling_locations, attention_weights)
    #
    #     output = self.output_proj(output)
    #
    #     if not self.batch_first:
    #         # (num_query, bs ,embed_dims)
    #         output = output.permute(1, 0, 2)
    #
    #     return self.dropout(output) + identity


# Below part is to replace the original Deformable Attention with our customized Attention (to support 1D),
# You do NOT have to read them at all.


class CustomDeformableDetrTransformerEncoder(DeformableDetrTransformerEncoder):

    def __init__(self, memory_fuse=False, *args, **kwargs) -> None:
        self.memory_fuse = memory_fuse
        super().__init__(*args, **kwargs)

    def _init_layers(self) -> None:
        """Initialize encoder layers."""
        self.layers = ModuleList([
            CustomDeformableDetrTransformerEncoderLayer(**self.layer_cfg)
            for _ in range(self.num_layers)
        ])
        self.embed_dims = self.layers[0].embed_dims
        # if self.memory_fuse:
        #     self.proj = nn.Linear(self.embed_dims * (self.num_layers + 1), self.embed_dims)
        #     self.norm = nn.LayerNorm(self.embed_dims)

    def forward(self, query: Tensor, query_pos: Tensor,
                key_padding_mask: Tensor, spatial_shapes: Tensor,
                level_start_index: Tensor, valid_ratios: Tensor,
                **kwargs) -> Tensor:
        """Forward function of Transformer encoder.

        Args:
            query (Tensor): The input query, has shape (bs, num_queries, dim).
            query_pos (Tensor): The positional encoding for query, has shape
                (bs, num_queries, dim).
            key_padding_mask (Tensor): The `key_padding_mask` of `self_attn`
                input. ByteTensor, has shape (bs, num_queries).
            spatial_shapes (Tensor): Spatial shapes of features in all levels,
                has shape (num_levels, 2), last dimension represents (h, w).
            level_start_index (Tensor): The start index of each level.
                A tensor has shape (num_levels, ) and can be represented
                as [0, h_0*w_0, h_0*w_0+h_1*w_1, ...].
            valid_ratios (Tensor): The ratios of the valid width and the valid
                height relative to the width and the height of features in all
                levels, has shape (bs, num_levels, 2).

        Returns:
            Tensor: Output queries of Transformer encoder, which is also
            called 'encoder output embeddings' or 'memory', has shape
            (bs, num_queries, dim)
        """
        reference_points = self.get_encoder_reference_points(
            spatial_shapes, valid_ratios, device=query.device)
        intermidiate_query = [query]
        for layer in self.layers:
            query = layer(
                query=query,
                query_pos=query_pos,
                key_padding_mask=key_padding_mask,
                spatial_shapes=spatial_shapes,
                level_start_index=level_start_index,
                valid_ratios=valid_ratios,
                reference_points=reference_points,
                **kwargs)
            intermidiate_query.append(query)
        if self.memory_fuse:
            query = torch.sum(torch.stack(intermidiate_query), dim=0)
            # query = torch.cat(intermidiate_query, dim=-1)
            # query = self.proj(query)
            # query = self.norm(query)
        return query


class CustomDeformableDetrTransformerDecoder(DeformableDetrTransformerDecoder):

    def _init_layers(self) -> None:
        """Initialize decoder layers."""
        self.layers = ModuleList([
            CustomDeformableDetrTransformerDecoderLayer(**self.layer_cfg)
            for _ in range(self.num_layers)
        ])
        self.embed_dims = self.layers[0].embed_dims
        if self.post_norm_cfg is not None:
            raise ValueError('There is not post_norm in '
                             f'{self._get_name()}')


class CustomDinoTransformerDecoder(DinoTransformerDecoder):

    def _init_layers(self) -> None:
        """Initialize decoder layers."""
        self.layers = ModuleList([
            CustomDeformableDetrTransformerDecoderLayer(**self.layer_cfg)
            for _ in range(self.num_layers)
        ])
        self.embed_dims = self.layers[0].embed_dims
        if self.post_norm_cfg is not None:
            raise ValueError('There is not post_norm in '
                             f'{self._get_name()}')
        self.ref_point_head = MLP(self.embed_dims * 2, self.embed_dims,
                                  self.embed_dims, 2)
        self.norm = nn.LayerNorm(self.embed_dims)


# class CustomMultiheadAttention(MultiheadAttention):
#
#     def forward(self,
#                 query,
#                 key=None,
#                 value=None,
#                 identity=None,
#                 query_pos=None,
#                 key_pos=None,
#                 attn_mask=None,
#                 key_padding_mask=None,
#                 **kwargs):
#         """Forward function for `MultiheadAttention`.
#
#         **kwargs allow passing a more general data flow when combining
#         with other operations in `transformerlayer`.
#
#         Args:
#             query (Tensor): The input query with shape [num_queries, bs,
#                 embed_dims] if self.batch_first is False, else
#                 [bs, num_queries embed_dims].
#             key (Tensor): The key tensor with shape [num_keys, bs,
#                 embed_dims] if self.batch_first is False, else
#                 [bs, num_keys, embed_dims] .
#                 If None, the ``query`` will be used. Defaults to None.
#             value (Tensor): The value tensor with same shape as `key`.
#                 Same in `nn.MultiheadAttention.forward`. Defaults to None.
#                 If None, the `key` will be used.
#             identity (Tensor): This tensor, with the same shape as x,
#                 will be used for the identity link.
#                 If None, `x` will be used. Defaults to None.
#             query_pos (Tensor): The positional encoding for query, with
#                 the same shape as `x`. If not None, it will
#                 be added to `x` before forward function. Defaults to None.
#             key_pos (Tensor): The positional encoding for `key`, with the
#                 same shape as `key`. Defaults to None. If not None, it will
#                 be added to `key` before forward function. If None, and
#                 `query_pos` has the same shape as `key`, then `query_pos`
#                 will be used for `key_pos`. Defaults to None.
#             attn_mask (Tensor): ByteTensor mask with shape [num_queries,
#                 num_keys]. Same in `nn.MultiheadAttention.forward`.
#                 Defaults to None.
#             key_padding_mask (Tensor): ByteTensor with shape [bs, num_keys].
#                 Defaults to None.
#
#         Returns:
#             Tensor: forwarded results with shape
#             [num_queries, bs, embed_dims]
#             if self.batch_first is False, else
#             [bs, num_queries embed_dims].
#         """
#
#         if key is None:
#             key = query
#         if value is None:
#             value = key
#         if identity is None:
#             identity = query
#         if key_pos is None:
#             if query_pos is not None:
#                 # use query_pos if key_pos is not available
#                 if query_pos.shape == key.shape:
#                     key_pos = query_pos
#                 else:
#                     warnings.warn(f'position encoding of key is'
#                                   f'missing in {self.__class__.__name__}.')
#         if query_pos is not None:
#             # query = query[:, -query_pos.shape[1]:, :] + query_pos
#             query[:, -query_pos.shape[1]:, :] += query_pos
#         if key_pos is not None:
#             # key = key[:, -key_pos.shape[1]:, :] + key_pos
#             key[:, -key_pos.shape[1]:, :] += key_pos
#
#         # Because the dataflow('key', 'query', 'value') of
#         # ``torch.nn.MultiheadAttention`` is (num_query, batch,
#         # embed_dims), We should adjust the shape of dataflow from
#         # batch_first (batch, num_query, embed_dims) to num_query_first
#         # (num_query ,batch, embed_dims), and recover ``attn_output``
#         # from num_query_first to batch_first.
#         if self.batch_first:
#             query = query.transpose(0, 1)
#             key = key.transpose(0, 1)
#             value = value.transpose(0, 1)
#
#         out = self.attn(
#             query=query,
#             key=key,
#             value=value,
#             attn_mask=attn_mask,
#             key_padding_mask=key_padding_mask)[0]
#
#         if self.batch_first:
#             out = out.transpose(0, 1)
#
#         return identity + self.dropout_layer(self.proj_drop(out))


class CustomDeformableDetrTransformerDecoderLayer(DeformableDetrTransformerDecoderLayer):
    def _init_layers(self) -> None:
        """Initialize self_attn, cross-attn, ffn, and norms."""
        self.self_attn = MultiheadAttention(**self.self_attn_cfg)
        self.cross_attn = CustomMultiScaleDeformableAttention(**self.cross_attn_cfg)
        self.embed_dims = self.self_attn.embed_dims
        self.ffn = FFN(**self.ffn_cfg)
        norms_list = [
            build_norm_layer(self.norm_cfg, self.embed_dims)[1]
            for _ in range(3)
        ]
        self.norms = ModuleList(norms_list)

class CustomDeformableDetrTransformerEncoderLayer(DeformableDetrTransformerEncoderLayer):

    def _init_layers(self) -> None:
        """Initialize self_attn, ffn, and norms."""
        self.self_attn = CustomMultiScaleDeformableAttention(**self.self_attn_cfg)
        self.embed_dims = self.self_attn.embed_dims
        self.ffn = FFN(**self.ffn_cfg)
        norms_list = [
            build_norm_layer(self.norm_cfg, self.embed_dims)[1]
            for _ in range(2)
        ]
        self.norms = ModuleList(norms_list)
