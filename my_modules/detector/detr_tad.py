from mmdet.models.detectors import DETR
from mmdet.models.layers import DetrTransformerDecoder, DetrTransformerEncoder
from mmdet.registry import MODELS
from torch import nn

from my_modules.layers.positional_encoding import SinePositional1dEncoding


@MODELS.register_module()
class DETR_TAD(DETR):
    """
    DETR for temporal action detection
    1. Modify the positional encoding from 2D to 1D to support temporal positional encoding.
    """

    def _init_layers(self) -> None:
        """Initialize layers except for backbone, neck and bbox_head."""
        self.positional_encoding = SinePositional1dEncoding(
            **self.positional_encoding)
        if self.encoder.get('num_layers', 0) == 0:
            self.encoder = PseudoEncoder()
            self.embed_dims = 256
        else:
            self.encoder = DetrTransformerEncoder(**self.encoder)
            self.embed_dims = self.encoder.embed_dims
        self.decoder = DetrTransformerDecoder(**self.decoder)
        # NOTE The embed_dims is typically passed from the inside out.
        # For example in DETR, The embed_dims is passed as
        # self_attn -> the first encoder layer -> encoder -> detector.
        self.query_embedding = nn.Embedding(self.num_queries, self.embed_dims)


class PseudoEncoder(nn.Module):

    def forward(self, query, *args, **kwargs):
        return query
