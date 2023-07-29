import math

import torch
from mmdet.models.layers import DetrTransformerDecoder
from mmdet.models.layers.transformer.utils import MLP, coordinate_to_encoding, inverse_sigmoid
from torch import Tensor, nn

from my_modules.layers import CustomDeformableDetrTransformerDecoderLayer


class MyTransformerDecoder(DetrTransformerDecoder):
    """Transformer encoder of DINO."""

    def __init__(self, dynamic_pos=False, *args, **kwargs):
        self.dynamic_pos = dynamic_pos
        super().__init__(*args, **kwargs)
        self.sqr = True

    def _init_layers(self) -> None:
        """Initialize decoder layers."""
        self.layers = nn.ModuleList([
            CustomDeformableDetrTransformerDecoderLayer(**self.layer_cfg)
            for _ in range(self.num_layers)
        ])
        self.embed_dims = self.layers[0].embed_dims
        if self.post_norm_cfg is not None:
            raise ValueError('There is not post_norm in '
                             f'{self._get_name()}')
        if self.dynamic_pos:
            self.ref_point_head = MLP(self.embed_dims * 2, self.embed_dims,
                                      self.embed_dims, 2)
        # self.norm = nn.LayerNorm(self.embed_dims)

    def forward(self, query: Tensor, query_pos: Tensor, value: Tensor, key_padding_mask: Tensor,
                self_attn_mask: Tensor, reference_points: Tensor,
                spatial_shapes: Tensor, level_start_index: Tensor,
                valid_ratios: Tensor, reg_branches: nn.ModuleList,
                **kwargs) -> Tensor:
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
            if self.dynamic_pos:
                query_sine_embed = self.coordinate_to_encoding(
                    # DINO compute query_pos based on each layer's referece_points
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
                # intermediate.append(self.norm(query))   # DINO add apply LayerNorm on each intermediate output
                intermediate.append(query)
                intermediate_reference_points.append(new_reference_points) # look forward twice

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
