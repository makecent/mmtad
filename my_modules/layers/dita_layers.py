import math
from typing import Tuple, Union, Optional

import torch
from mmcv.cnn import build_norm_layer
from mmcv.cnn.bricks.transformer import FFN, MultiheadAttention
from mmcv.ops import MultiScaleDeformableAttention
from mmcv.utils import ext_loader
from mmdet.models.layers import DeformableDetrTransformerDecoderLayer, DetrTransformerEncoderLayer, \
    DinoTransformerDecoder, DeformableDetrTransformerEncoder, DeformableDetrTransformerDecoder
from mmdet.models.layers.transformer.utils import MLP, coordinate_to_encoding, inverse_sigmoid
from mmdet.utils import ConfigType
from mmdet.utils import OptConfigType
from mmengine import ConfigDict
from mmengine.model import BaseModule
from mmengine.model import ModuleList
from torch import Tensor, nn

ext_module = ext_loader.load_ext(
    '_ext', ['ms_deform_attn_backward', 'ms_deform_attn_forward'])

from my_modules.layers.pseudo_layers import Pseudo2DLinear

try:
    from fairscale.nn.checkpoint import checkpoint_wrapper
except Exception:
    checkpoint_wrapper = None


def constant_y_reference_points(forward_method):
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


class TdtrTransformerEncoder(DeformableDetrTransformerEncoder):
    """
    Modifications:
    a. Use Deformable1D attention module
    b. Support memory fusion to sum all layers' output.
    """

    def __init__(self, deformable=True, memory_fuse=False, *args, **kwargs):
        self.deformable = deformable
        self.memory_fuse = memory_fuse
        if not deformable:
            kwargs['layer_cfg']['self_attn_cfg'].pop('num_levels')
        super().__init__(*args, **kwargs)

    def _init_layers(self) -> None:
        """Initialize encoder layers."""
        self.layers = ModuleList([
            TdtrTransformerEncoderLayer(**self.layer_cfg) if self.deformable else DetrTransformerEncoderLayer(
                **self.layer_cfg)
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
        all_layers_query = [query]
        if self.deformable:
            reference_points = self.get_encoder_reference_points(
                spatial_shapes, valid_ratios, device=query.device)
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
        else:
            for layer in self.layers:
                query = layer(query, query_pos, key_padding_mask, **kwargs)
                all_layers_query.append(query)
        if self.memory_fuse:
            query = torch.sum(torch.stack(all_layers_query), dim=0)
        return query


class TdtrTransformerDecoder(DeformableDetrTransformerDecoder):
    """
    Modifications:
    a. Use Deformable1D attention module
    b. Support dynamic query pos, i.e., updated as the change of reference points across decoder layers
    """

    def __init__(self, dynamic_query_pos, *args, **kwargs):
        self.dynamic_query_pos = dynamic_query_pos
        super().__init__(*args, **kwargs)

    def _init_layers(self) -> None:
        """Initialize decoder layers."""
        self.layers = ModuleList([
            TdtrTransformerDecoderLayer(**self.layer_cfg)
            for _ in range(self.num_layers)
        ])
        self.embed_dims = self.layers[0].embed_dims
        if self.post_norm_cfg is not None:
            self.post_norm = build_norm_layer(self.post_norm_cfg, self.embed_dims)[1]
        else:
            self.post_norm = nn.Identity()
        if self.dynamic_query_pos:
            self.ref_point_head = MLP(self.embed_dims * 2, self.embed_dims, self.embed_dims, 2)

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
                reg_branches: Optional[nn.Module] = None,
                **kwargs):

        intermediate = []
        intermediate_reference_points = [reference_points]
        for layer_id, layer in enumerate(self.layers):
            if reference_points.shape[-1] == 4:
                reference_points_input = \
                    reference_points[:, :, None] * torch.cat(
                        [valid_ratios, valid_ratios], -1)[:, None]
            else:
                assert reference_points.shape[-1] == 2
                reference_points_input = \
                    reference_points[:, :, None] * valid_ratios[:, None]

            if self.dynamic_query_pos:
                query_sine_embed = coordinate_to_encoding(
                    reference_points_input[:, :, 0, :])
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
                tmp_reg_preds = reg_branches[layer_id](query)
                if reference_points.shape[-1] == 4:
                    new_reference_points = tmp_reg_preds + inverse_sigmoid(reference_points)
                    new_reference_points = new_reference_points.sigmoid()
                else:
                    assert reference_points.shape[-1] == 2
                    new_reference_points = tmp_reg_preds
                    new_reference_points[..., :2] = tmp_reg_preds[..., :2] + inverse_sigmoid(reference_points)
                    new_reference_points = new_reference_points.sigmoid()
                reference_points = new_reference_points.detach()

            if self.return_intermediate:
                intermediate.append(self.post_norm(query))
                intermediate_reference_points.append(reference_points)
                # in the DINO, new_reference_points was appended ("Look Forward Twice").

        if self.return_intermediate:
            return torch.stack(intermediate), torch.stack(intermediate_reference_points)
        return query.unsqueeze(0), reference_points.unsqueeze(0)


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
            TdtrTransformerDecoderLayer(**self.layer_cfg)
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


class TdtrTransformerEncoderLayer(DetrTransformerEncoderLayer):

    def _init_layers(self) -> None:
        """Initialize self_attn, ffn, and norms."""
        self.self_attn = MultiScaleDeformableAttention1d(**self.self_attn_cfg)
        self.embed_dims = self.self_attn.embed_dims
        self.ffn = FFN(**self.ffn_cfg)
        norms_list = [
            build_norm_layer(self.norm_cfg, self.embed_dims)[1]
            for _ in range(2)
        ]
        self.norms = ModuleList(norms_list)


class TdtrTransformerDecoderLayer(DeformableDetrTransformerDecoderLayer):

    def _init_layers(self) -> None:
        """Initialize self_attn, cross-attn, ffn, and norms."""
        self.self_attn = MultiheadAttention(**self.self_attn_cfg)
        self.cross_attn = MultiScaleDeformableAttention1d(**self.cross_attn_cfg)
        self.embed_dims = self.self_attn.embed_dims
        self.ffn = FFN(**self.ffn_cfg)
        norms_list = [
            build_norm_layer(self.norm_cfg, self.embed_dims)[1]
            for _ in range(3)
        ]
        self.norms = ModuleList(norms_list)


class MultiScaleDeformableAttention1d(MultiScaleDeformableAttention):
    """
    DITA DeformableAttention, compared with normal DeformableAttention:
    a. sampling_offsets linear layer is changed to output only x offsets, and y offsets are fixed to be zeros.
    b. Init the sampling_offsets bias with in 1D style
    c. decorate the forward() function to fix the input reference point on y-axis to be zeros.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # the sampling_offsets layer only predict x offsets. The y offsets are fixed to be zeros.
        self.sampling_offsets = Pseudo2DLinear(self.embed_dims, self.num_heads * self.num_levels * self.num_points)

    def init_weights(self) -> None:
        """Default initialization for Parameters of Module."""
        super().init_weights()
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

    @constant_y_reference_points
    def forward(self, *args, **kwargs):
        return super().forward(*args, **kwargs)


class Dino1dTransformerDecoder(DinoTransformerDecoder):

    def _init_layers(self) -> None:
        """Initialize decoder layers."""
        self.layers = ModuleList([
            TdtrTransformerDecoderLayer(**self.layer_cfg)
            for _ in range(self.num_layers)
        ])
        self.embed_dims = self.layers[0].embed_dims
        if self.post_norm_cfg is not None:
            raise ValueError('There is not post_norm in '
                             f'{self._get_name()}')
        self.ref_point_head = MLP(self.embed_dims * 2, self.embed_dims,
                                  self.embed_dims, 2)
        self.norm = nn.LayerNorm(self.embed_dims)
