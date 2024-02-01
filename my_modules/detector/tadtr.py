import torch
from mmdet.models import DeformableDETR
from mmdet.registry import MODELS
from torch import nn

from my_modules.layers.dita_layers import DitaTransformerEncoder, \
    Deformable1dDetrTransformerDecoder
from my_modules.layers.positional_encoding import SinePositional1dEncoding


@MODELS.register_module()
class TadTR(DeformableDETR):
    """
    The TadTR implementation based on Deformable DETR. We replace the MultiScaleDeformableAttention
    to support temporal attention.
    """

    def _init_layers(self) -> None:
        # Replace the MultiScaleDeformableAttention to support temporal attention.
        self.encoder = DitaTransformerEncoder(**self.encoder)
        self.decoder = Deformable1dDetrTransformerDecoder(**self.decoder)
        self.positional_encoding = SinePositional1dEncoding(**self.positional_encoding)

        # Below are the same with DeformableDETR
        self.embed_dims = self.encoder.embed_dims
        if not self.as_two_stage:
            self.query_embedding = nn.Embedding(self.num_queries,
                                                self.embed_dims * 2)
            # NOTE The query_embedding will be split into query and query_pos
            # in self.pre_decoder, hence, the embed_dims are doubled.

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
