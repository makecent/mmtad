import math
import warnings
from typing import Optional, no_type_check
from typing import Union, Tuple

import mmengine
import torch
from mmcv.cnn import build_norm_layer
from mmcv.cnn.bricks.transformer import FFN, MultiheadAttention
from mmcv.ops.multi_scale_deform_attn import MultiScaleDeformableAttnFunction, multi_scale_deformable_attn_pytorch
from mmcv.utils import IS_CUDA_AVAILABLE, IS_MLU_AVAILABLE
from mmcv.utils import ext_loader
from mmdet.models.layers import DeformableDetrTransformerDecoder, \
    DeformableDetrTransformerDecoderLayer, DeformableDetrTransformerEncoderLayer, DinoTransformerDecoder
from mmdet.models.layers import MLP
from mmdet.models.layers.transformer.utils import inverse_sigmoid
from mmdet.utils import ConfigType, OptConfigType
from mmengine import ConfigDict
from mmengine.model import BaseModule, ModuleList
from mmengine.model import constant_init, xavier_init
from mmengine.utils import deprecated_api_warning
from torch import Tensor
from torch import nn


ext_module = ext_loader.load_ext(
    '_ext', ['ms_deform_attn_backward', 'ms_deform_attn_forward'])

from my_modules.layers.pseudo_layers import Pseudo2DLinear

try:
    from fairscale.nn.checkpoint import checkpoint_wrapper
except Exception:
    checkpoint_wrapper = None


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


# Below part is to replace the original Deformable Attention with our customized Attention (to support 1D),
# You do NOT have to read them at all.


class DitaTransformerEncoder(BaseModule):

    def __init__(self,
                 num_layers: int,
                 layer_cfg: ConfigType,
                 memory_fuse=False,
                 num_cp: int = -1,
                 init_cfg: OptConfigType = None) -> None:

        super().__init__(init_cfg=init_cfg)
        self.num_layers = num_layers
        self.layer_cfg = layer_cfg
        self.memory_fuse = memory_fuse
        self.num_cp = num_cp
        assert self.num_cp <= self.num_layers
        self._init_layers()

    def _init_layers(self) -> None:
        """Initialize encoder layers."""
        self.layers = ModuleList([
            DitaTransformerEncoderLayer(**self.layer_cfg)
            for _ in range(self.num_layers)
        ])

        if self.num_cp > 0:
            if checkpoint_wrapper is None:
                raise NotImplementedError(
                    'If you want to reduce GPU memory usage, \
                    please install fairscale by executing the \
                    following command: pip install fairscale.')
            for i in range(self.num_cp):
                self.layers[i] = checkpoint_wrapper(self.layers[i])

        self.embed_dims = self.layers[0].embed_dims

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
        all_layers_query = [query]
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
            all_layers_query.append(query)
        if self.memory_fuse:
            query = torch.sum(torch.stack(all_layers_query), dim=0)
        return query

    @staticmethod
    def get_encoder_reference_points(
            spatial_shapes: Tensor, valid_ratios: Tensor,
            device: Union[torch.device, str]) -> Tensor:
        """Get the reference points used in encoder.

        Args:
            spatial_shapes (Tensor): Spatial shapes of features in all levels,
                has shape (num_levels, 2), last dimension represents (h, w).
            valid_ratios (Tensor): The ratios of the valid width and the valid
                height relative to the width and the height of features in all
                levels, has shape (bs, num_levels, 2).
            device (obj:`device` or str): The device acquired by the
                `reference_points`.

        Returns:
            Tensor: Reference points used in decoder, has shape (bs, length,
            num_levels, 2).
        """

        reference_points_list = []
        for lvl, (H, W) in enumerate(spatial_shapes):
            ref_y, ref_x = torch.meshgrid(
                torch.linspace(
                    0.5, H - 0.5, H, dtype=torch.float32, device=device),
                torch.linspace(
                    0.5, W - 0.5, W, dtype=torch.float32, device=device))
            ref_y = ref_y.reshape(-1)[None] / (
                valid_ratios[:, None, lvl, 1] * H)
            ref_x = ref_x.reshape(-1)[None] / (
                valid_ratios[:, None, lvl, 0] * W)
            ref = torch.stack((ref_x, ref_y), -1)
            reference_points_list.append(ref)
        reference_points = torch.cat(reference_points_list, 1)
        # [bs, sum(hw), num_level, 2]
        reference_points = reference_points[:, :, None] * valid_ratios[:, None]
        return reference_points


class DitaTransformerDecoder(BaseModule):
    """Transformer encoder of DITA, support selective_query_recollection (SQR) https://arxiv.org/abs/2212.07593
     and dynamic query_pos where the query_pos are computed by reference points in each layer via projection"""

    def __init__(self,
                 num_layers: int,
                 layer_cfg: ConfigType,
                 post_norm_cfg: OptConfigType = dict(type='LN'),
                 return_intermediate: bool = True,
                 dynamic_query_pos=False,
                 selective_query_recollection=False,
                 init_cfg: Union[dict, ConfigDict] = None) -> None:
        super().__init__(init_cfg=init_cfg)
        self.layer_cfg = layer_cfg
        self.num_layers = num_layers
        self.post_norm_cfg = post_norm_cfg
        self.return_intermediate = return_intermediate
        self.dynamic_query_pos = dynamic_query_pos
        self.sqr = selective_query_recollection
        self._init_layers()

    def _init_layers(self) -> None:
        """Initialize decoder layers."""
        self.layers = nn.ModuleList([
            DitaTransformerDecoderLayer(**self.layer_cfg)
            for _ in range(self.num_layers)
        ])
        self.embed_dims = self.layers[0].embed_dims
        if self.post_norm_cfg is not None:
            self.post_norm = build_norm_layer(self.post_norm_cfg,
                                              self.embed_dims)[1]
        else:
            self.post_norm = nn.Identity()
        if self.dynamic_query_pos:
            self.ref_point_head = MLP(self.embed_dims * 2, self.embed_dims,
                                      self.embed_dims, 2)

    def forward(self,
                query: Tensor,
                query_pos: Tensor,
                value: Tensor,
                key_padding_mask: Tensor,
                self_attn_mask: Tensor,
                reference_points: Tensor,
                spatial_shapes: Tensor,
                level_start_index: Tensor,
                valid_ratios: Tensor,
                reg_branches: nn.ModuleList,
                **kwargs) -> Tuple[Tensor]:
        intermediate = []
        intermediate_reference_points = [reference_points]
        if self.sqr and self.training:
            batch_size, query_dim, emd_dim = query.shape
            selective_num = batch_size
            query_pos_reserve, value_reserve, key_padding_mask_reserve, valid_ratios_reserve \
                = query_pos, value, key_padding_mask, valid_ratios
        for lid, layer in enumerate(self.layers):
            if self.sqr and self.training:
                query_reserve = query
                repeats = query.shape[0] // batch_size
                query_pos = query_pos_reserve.repeat(repeats, 1, 1)
                value = value_reserve.repeat(repeats, 1, 1)
                key_padding_mask = key_padding_mask_reserve.repeat(repeats, 1)
                valid_ratios = valid_ratios_reserve.repeat(repeats, 1, 1)
            if reference_points.shape[-1] == 4:
                reference_points_input = \
                    reference_points[:, :, None] * torch.cat(
                        [valid_ratios, valid_ratios], -1)[:, None]
            else:
                assert reference_points.shape[-1] == 2
                reference_points_input = \
                    reference_points[:, :, None] * valid_ratios[:, None]
            if self.dynamic_query_pos:
                # compute query_pos based on each layer's reference points
                query_sine_embed = self.coordinate_to_encoding(
                    reference_points_input[:, :, 0, :], num_feats=self.embed_dims)
                query_pos = self.ref_point_head(query_sine_embed)

            query = layer(
                query,
                query_pos=query_pos,
                value=value,
                key_padding_mask=key_padding_mask,
                self_attn_mask=self_attn_mask,
                spatial_shapes=spatial_shapes,
                level_start_index=level_start_index,
                valid_ratios=valid_ratios,
                reference_points=reference_points_input,
                **kwargs)

            if reg_branches is not None:
                tmp = reg_branches[lid](query)
                if reference_points.shape[-1] == 4:
                    new_reference_points = tmp + inverse_sigmoid(reference_points, eps=1e-5)  # DINO use eps=1e-3
                    new_reference_points = new_reference_points.sigmoid()
                else:
                    assert reference_points.shape[-1] == 2
                    new_reference_points = tmp
                    new_reference_points[..., :2] = tmp[..., :2] + inverse_sigmoid(reference_points, eps=1e-5)
                    new_reference_points = new_reference_points.sigmoid()
            else:
                new_reference_points = reference_points

            # Query Recollection
            if self.training and self.sqr:
                intermediate.append(query)
                new_reference_points = torch.cat(
                    [new_reference_points, intermediate_reference_points[-1][:selective_num]])
                intermediate_reference_points.append(new_reference_points)
                # # Dense Query Recollection (currently not support because the reference points are hard to handle,
                # I have tried repeating the reference points to next layer but the performance is not good.
                # query_rec = torch.cat([query, pre_query], dim=1)
                # Selective Query Recollection
                query_rec = torch.cat([query, query_reserve[:selective_num]])
                selective_num = query.shape[0]
                query = query_rec

            elif self.return_intermediate:
                intermediate.append(self.post_norm(query))  # DINO perform post-norm while DeformableDETR does not
                # intermediate_reference_points.append(reference_points)  # look forward once
                intermediate_reference_points.append(new_reference_points)  # look forward twice used in DINO

            reference_points = new_reference_points.detach()

        if self.return_intermediate:
            return intermediate, intermediate_reference_points

        return query, reference_points

    @staticmethod
    def coordinate_to_encoding(coord_tensor: Tensor,
                               num_feats: int = 256,
                               temperature: int = 10000,
                               scale: float = 2 * math.pi):
        """Convert coordinate tensor to positional encoding.
        Ignore the y-axis for temporal action detection
        """
        dim_t = torch.arange(
            num_feats, dtype=torch.float32, device=coord_tensor.device)
        dim_t = temperature ** (2 * (dim_t // 2) / num_feats)
        x_embed = coord_tensor[..., 0] * scale
        # y_embed = coord_tensor[..., 1] * scale
        pos_x = x_embed[..., None] / dim_t
        # pos_y = y_embed[..., None] / dim_t
        pos_x = torch.stack((pos_x[..., 0::2].sin(), pos_x[..., 1::2].cos()),
                            dim=-1).flatten(2)
        # pos_y = torch.stack((pos_y[..., 0::2].sin(), pos_y[..., 1::2].cos()),
        #                     dim=-1).flatten(2)
        if coord_tensor.size(-1) == 2:
            # pos = torch.cat((pos_y, pos_x), dim=-1)
            pos = pos_x
        elif coord_tensor.size(-1) == 4:
            w_embed = coord_tensor[..., 2] * scale
            pos_w = w_embed[..., None] / dim_t
            pos_w = torch.stack((pos_w[..., 0::2].sin(), pos_w[..., 1::2].cos()),
                                dim=-1).flatten(2)

            # h_embed = coord_tensor[..., 3] * scale
            # pos_h = h_embed[..., None] / dim_t
            # pos_h = torch.stack((pos_h[..., 0::2].sin(), pos_h[..., 1::2].cos()),
            #                     dim=-1).flatten(2)

            # pos = torch.cat((pos_y, pos_x, pos_w, pos_h), dim=-1)
            pos = torch.cat((pos_x, pos_w), dim=-1)
        else:
            raise ValueError('Unknown pos_tensor shape(-1):{}'.format(
                coord_tensor.size(-1)))
        return pos


class DitaTransformerEncoderLayer(DeformableDetrTransformerEncoderLayer):

    def _init_layers(self) -> None:
        """Initialize self_attn, ffn, and norms."""
        self.self_attn = DitaMultiScaleDeformableAttention(**self.self_attn_cfg)
        self.embed_dims = self.self_attn.embed_dims
        self.ffn = FFN(**self.ffn_cfg)
        norms_list = [
            build_norm_layer(self.norm_cfg, self.embed_dims)[1]
            for _ in range(2)
        ]
        self.norms = ModuleList(norms_list)


class DitaTransformerDecoderLayer(DeformableDetrTransformerDecoderLayer):
    def _init_layers(self) -> None:
        """Initialize self_attn, cross-attn, ffn, and norms."""
        self.self_attn = MultiheadAttention(**self.self_attn_cfg)
        self.cross_attn = DitaMultiScaleDeformableAttention(**self.cross_attn_cfg)
        self.embed_dims = self.self_attn.embed_dims
        self.ffn = FFN(**self.ffn_cfg)
        norms_list = [
            build_norm_layer(self.norm_cfg, self.embed_dims)[1]
            for _ in range(3)
        ]
        self.norms = ModuleList(norms_list)


class DitaMultiScaleDeformableAttention(BaseModule):
    """
    DITA DeformableAttention, compared with normal DeformableAttention:
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
        super().__init__(init_cfg)
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
        # the sampling_offsets layer only output x offsets. The y offsets are fixed to be zeros.
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
        thetas = torch.arange(
            self.num_heads, dtype=torch.float32,
            device=device) * (4.0 * math.pi / self.num_heads)
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
    @no_type_check
    @deprecated_api_warning({'residual': 'identity'},
                            cls_name='MultiScaleDeformableAttention')
    def forward(self,
                query: torch.Tensor,
                key: Optional[torch.Tensor] = None,
                value: Optional[torch.Tensor] = None,
                identity: Optional[torch.Tensor] = None,
                query_pos: Optional[torch.Tensor] = None,
                key_padding_mask: Optional[torch.Tensor] = None,
                reference_points: Optional[torch.Tensor] = None,
                spatial_shapes: Optional[torch.Tensor] = None,
                level_start_index: Optional[torch.Tensor] = None,
                **kwargs) -> torch.Tensor:
        """Forward Function of MultiScaleDeformAttention.

        Args:
            query (torch.Tensor): Query of Transformer with shape
                (num_query, bs, embed_dims).
            key (torch.Tensor): The key tensor with shape
                `(num_key, bs, embed_dims)`.
            value (torch.Tensor): The value tensor with shape
                `(num_key, bs, embed_dims)`.
            identity (torch.Tensor): The tensor used for addition, with the
                same shape as `query`. Default None. If None,
                `query` will be used.
            query_pos (torch.Tensor): The positional encoding for `query`.
                Default: None.
            key_padding_mask (torch.Tensor): ByteTensor for `query`, with
                shape [bs, num_key].
            reference_points (torch.Tensor):  The normalized reference
                points with shape (bs, num_query, num_levels, 2),
                all elements is range in [0, 1], top-left (0,0),
                bottom-right (1, 1), including padding area.
                or (N, Length_{query}, num_levels, 4), add
                additional two dimensions is (w, h) to
                form reference boxes.
            spatial_shapes (torch.Tensor): Spatial shape of features in
                different levels. With shape (num_levels, 2),
                last dimension represents (h, w).
            level_start_index (torch.Tensor): The start index of each level.
                A tensor has shape ``(num_levels, )`` and can be represented
                as [0, h_0*w_0, h_0*w_0+h_1*w_1, ...].

        Returns:
            torch.Tensor: forwarded results with shape
            [num_query, bs, embed_dims].
        """

        if value is None:
            value = query

        if identity is None:
            identity = query
        if query_pos is not None:
            query = query + query_pos
        if not self.batch_first:
            # change to (bs, num_query ,embed_dims)
            query = query.permute(1, 0, 2)
            value = value.permute(1, 0, 2)

        bs, num_query, _ = query.shape
        bs, num_value, _ = value.shape
        assert (spatial_shapes[:, 0] * spatial_shapes[:, 1]).sum() == num_value

        value = self.value_proj(value)
        if key_padding_mask is not None:
            value = value.masked_fill(key_padding_mask[..., None], 0.0)
        value = value.view(bs, num_value, self.num_heads, -1)
        sampling_offsets = self.sampling_offsets(query).view(
            bs, num_query, self.num_heads, self.num_levels, self.num_points, 2)
        attention_weights = self.attention_weights(query).view(
            bs, num_query, self.num_heads, self.num_levels * self.num_points)
        attention_weights = attention_weights.softmax(-1)

        attention_weights = attention_weights.view(bs, num_query,
                                                   self.num_heads,
                                                   self.num_levels,
                                                   self.num_points)
        if reference_points.shape[-1] == 2:
            offset_normalizer = torch.stack(
                [spatial_shapes[..., 1], spatial_shapes[..., 0]], -1)
            sampling_locations = reference_points[:, :, None, :, None, :] \
                                 + sampling_offsets \
                                 / offset_normalizer[None, None, None, :, None, :]
        elif reference_points.shape[-1] == 4:
            sampling_locations = reference_points[:, :, None, :, None, :2] \
                                 + sampling_offsets / self.num_points \
                                 * reference_points[:, :, None, :, None, 2:] \
                                 * 0.5
        else:
            raise ValueError(
                f'Last dim of reference_points must be'
                f' 2 or 4, but get {reference_points.shape[-1]} instead.')
        if ((IS_CUDA_AVAILABLE and value.is_cuda)
                or (IS_MLU_AVAILABLE and value.is_mlu)):
            output = MultiScaleDeformableAttnFunction.apply(
                value, spatial_shapes, level_start_index, sampling_locations,
                attention_weights, self.im2col_step)
        else:
            output = multi_scale_deformable_attn_pytorch(
                value, spatial_shapes, sampling_locations, attention_weights)

        output = self.output_proj(output)

        if not self.batch_first:
            # (num_query, bs ,embed_dims)
            output = output.permute(1, 0, 2)

        return self.dropout(output) + identity


class Deformable1dDetrTransformerDecoder(DeformableDetrTransformerDecoder):

    def _init_layers(self) -> None:
        """Initialize decoder layers."""
        self.layers = ModuleList([
            DitaTransformerDecoderLayer(**self.layer_cfg)
            for _ in range(self.num_layers)
        ])
        self.embed_dims = self.layers[0].embed_dims
        if self.post_norm_cfg is not None:
            raise ValueError('There is not post_norm in '
                             f'{self._get_name()}')


class Dino1dTransformerDecoder(DinoTransformerDecoder):

    def _init_layers(self) -> None:
        """Initialize decoder layers."""
        self.layers = ModuleList([
            DitaTransformerDecoderLayer(**self.layer_cfg)
            for _ in range(self.num_layers)
        ])
        self.embed_dims = self.layers[0].embed_dims
        if self.post_norm_cfg is not None:
            raise ValueError('There is not post_norm in '
                             f'{self._get_name()}')
        self.ref_point_head = MLP(self.embed_dims * 2, self.embed_dims,
                                  self.embed_dims, 2)
        self.norm = nn.LayerNorm(self.embed_dims)
