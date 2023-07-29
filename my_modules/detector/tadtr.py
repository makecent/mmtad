import torch
from mmdet.models import DeformableDETR
from mmdet.registry import MODELS
from torch import nn

from my_modules.layers.custom_layers import CustomDeformableDetrTransformerEncoder, \
    CustomDeformableDetrTransformerDecoder
from my_modules.layers.positional_encoding import CustomSinePositionalEncoding


@MODELS.register_module()
class TadTR(DeformableDETR):
    """
    The TadTR implementation based on Deformable DETR. We replace the MultiScaleDeformableAttention
    to support temporal attention.
    """

    def _init_layers(self) -> None:
        # Replace the MultiScaleDeformableAttention to support temporal attention.
        self.encoder = CustomDeformableDetrTransformerEncoder(**self.encoder)
        self.decoder = CustomDeformableDetrTransformerDecoder(**self.decoder)
        self.positional_encoding = CustomSinePositionalEncoding(**self.positional_encoding)

        # Below are the same with DeformableDETR
        self.embed_dims = self.encoder.embed_dims
        if not self.as_two_stage:
            self.query_embedding = nn.Embedding(self.num_queries,
                                                self.embed_dims * 2)
            # NOTE The query_embedding will be split into query and query_pos
            # in self.pre_decoder, hence, the embed_dims are doubled.

        num_feats = self.positional_encoding.num_feats
        assert num_feats * 2 == self.embed_dims, \
            'embed_dims should be exactly 2 times of num_feats. ' \
            f'Found {self.embed_dims} and {num_feats}.'

        self.level_embed = nn.Parameter(
            torch.Tensor(self.num_feature_levels, self.embed_dims))

        if self.as_two_stage:
            self.memory_trans_fc = nn.Linear(self.embed_dims, self.embed_dims)
            self.memory_trans_norm = nn.LayerNorm(self.embed_dims)
            self.pos_trans_fc = nn.Linear(self.embed_dims * 2,
                                          self.embed_dims * 2)
            self.pos_trans_norm = nn.LayerNorm(self.embed_dims * 2)
        else:
            self.reference_points_fc = nn.Linear(self.embed_dims, 2)
