from typing import Dict, Tuple

import torch
from mmdet.models.detectors import DINO
from mmdet.models.detectors.deformable_detr import DeformableDETR, MultiScaleDeformableAttention
from mmdet.models.layers import CdnQueryGenerator
from mmdet.registry import MODELS
from mmdet.structures import OptSampleList
from mmdet.utils import OptConfigType
from mmengine.model import xavier_init
from torch import Tensor, nn
from torch.nn.init import normal_

from my_modules.layers.custom_layers import CustomDeformableDetrTransformerEncoder, CustomDinoTransformerDecoder
from my_modules.layers.positional_encoding import CustomSinePositionalEncoding
from my_modules.layers.pseudo_layers import Pseudo4DRegLinear


@MODELS.register_module()
class CustomDINO(DINO):
    """
    Customized DINO to support Temporal Action Detection.
    1. The MultiScaleDeformableAttention is modified to support temporal attention.
    2. Now it supports one-stage.
    """

    def __init__(self, *args, dn_cfg: OptConfigType = None, **kwargs) -> None:
        super(DINO, self).__init__(*args, **kwargs)
        assert self.with_box_refine, 'with_box_refine must be True for DINO'
        self.dn_cfg = dn_cfg
        if dn_cfg is not None:
            assert 'num_classes' not in dn_cfg and \
                   'num_queries' not in dn_cfg and \
                   'hidden_dim' not in dn_cfg, \
                'The three keyword args `num_classes`, `embed_dims`, and ' \
                '`num_matching_queries` are set in `detector.__init__()`, ' \
                'users should not set them in `dn_cfg` config.'
            dn_cfg['num_classes'] = self.bbox_head.num_classes
            dn_cfg['embed_dims'] = self.embed_dims
            dn_cfg['num_matching_queries'] = self.num_queries
            self.dn_query_generator = CdnQueryGenerator(**dn_cfg)

    def _init_layers(self) -> None:
        """Initialize layers except for backbone, neck and bbox_head."""
        self.positional_encoding = CustomSinePositionalEncoding(
            **self.positional_encoding)
        self.encoder = CustomDeformableDetrTransformerEncoder(**self.encoder)
        self.decoder = CustomDinoTransformerDecoder(**self.decoder)
        self.embed_dims = self.encoder.embed_dims
        self.query_embedding = nn.Embedding(self.num_queries, self.embed_dims)
        # NOTE In DINO, the query_embedding only contains content
        # queries, while in Deformable DETR, the query_embedding
        # contains both content and spatial queries, and in DETR,
        # it only contains spatial queries.

        num_feats = self.positional_encoding.num_feats
        assert num_feats * 2 == self.embed_dims, \
            f'embed_dims should be exactly 2 times of num_feats. ' \
            f'Found {self.embed_dims} and {num_feats}.'

        self.level_embed = nn.Parameter(
            torch.Tensor(self.num_feature_levels, self.embed_dims))
        if self.as_two_stage:
            self.memory_trans_fc = nn.Linear(self.embed_dims, self.embed_dims)
            self.memory_trans_norm = nn.LayerNorm(self.embed_dims)
        else:
            self.reference_points_fc = Pseudo4DRegLinear(self.embed_dims, delta=False)

    def init_weights(self) -> None:
        """Initialize weights for Transformer and other components."""
        super(DeformableDETR, self).init_weights()
        for coder in self.encoder, self.decoder:
            for p in coder.parameters():
                if p.dim() > 1:
                    nn.init.xavier_uniform_(p)
        for m in self.modules():
            if isinstance(m, MultiScaleDeformableAttention):
                m.init_weights()
        if self.as_two_stage:
            nn.init.xavier_uniform_(self.memory_trans_fc.weight)
            nn.init.xavier_uniform_(self.query_embedding.weight)
        else:
            xavier_init(self.reference_points_fc, distribution='uniform', bias=0)
        normal_(self.level_embed)

    def pre_decoder(
            self,
            memory: Tensor,
            memory_mask: Tensor,
            spatial_shapes: Tensor,
            batch_data_samples: OptSampleList = None,
    ) -> Tuple[Dict]:
        """Prepare intermediate variables before entering Transformer decoder,
        such as `query`, `query_pos`, and `reference_points`.

        Args:
            memory (Tensor): The output embeddings of the Transformer encoder,
                has shape (bs, num_feat_points, dim).
            memory_mask (Tensor): ByteTensor, the padding mask of the memory,
                has shape (bs, num_feat_points). Will only be used when
                `as_two_stage` is `True`.
            spatial_shapes (Tensor): Spatial shapes of features in all levels.
                With shape (num_levels, 2), last dimension represents (h, w).
                Will only be used when `as_two_stage` is `True`.
            batch_data_samples (list[:obj:`DetDataSample`]): The batch
                data samples. It usually includes information such
                as `gt_instance` or `gt_panoptic_seg` or `gt_sem_seg`.
                Defaults to None.

        Returns:
            tuple[dict]: The decoder_inputs_dict and head_inputs_dict.

            - decoder_inputs_dict (dict): The keyword dictionary args of
              `self.forward_decoder()`, which includes 'query', 'memory',
              `reference_points`, and `dn_mask`. The reference points of
              decoder input here are 4D boxes, although it has `points`
              in its name.
            - head_inputs_dict (dict): The keyword dictionary args of the
              bbox_head functions, which includes `topk_score`, `topk_coords`,
              and `dn_meta` when `self.training` is `True`, else is empty.
        """
        bs, _, c = memory.shape
        if self.as_two_stage:
            cls_out_features = self.bbox_head.cls_branches[
                self.decoder.num_layers].out_features

            output_memory, output_proposals = self.gen_encoder_output_proposals(
                memory, memory_mask, spatial_shapes)
            enc_outputs_class = self.bbox_head.cls_branches[
                self.decoder.num_layers](
                output_memory)
            enc_outputs_coord_unact = self.bbox_head.reg_branches[
                                          self.decoder.num_layers](output_memory) + output_proposals

            # NOTE The DINO selects top-k proposals according to scores of
            # multi-class classification, while DeformDETR, where the input
            # is `enc_outputs_class[..., 0]` selects according to scores of
            # binary classification.
            topk_indices = torch.topk(
                enc_outputs_class.max(-1)[0], k=self.num_queries, dim=1)[1]
            topk_score = torch.gather(
                enc_outputs_class, 1,
                topk_indices.unsqueeze(-1).repeat(1, 1, cls_out_features))
            topk_coords_unact = torch.gather(
                enc_outputs_coord_unact, 1,
                topk_indices.unsqueeze(-1).repeat(1, 1, 4))
            topk_coords = topk_coords_unact.sigmoid()
            reference_points_unact = topk_coords_unact.detach()

            query = self.query_embedding.weight[:, None, :]
            query = query.repeat(1, bs, 1).transpose(0, 1)
        else:
            topk_score, topk_coords = None, None
            query = self.query_embedding.weight.unsqueeze(0).expand(bs, -1, -1)
            reference_points_unact = self.reference_points_fc(query)

        if self.training and (self.dn_cfg is not None):
            dn_label_query, dn_bbox_query, dn_mask, dn_meta = \
                self.dn_query_generator(batch_data_samples)
            query = torch.cat([dn_label_query, query], dim=1)
            reference_points_unact = torch.cat([dn_bbox_query, reference_points_unact],
                                               dim=1)
        else:
            reference_points_unact = reference_points_unact
            dn_mask, dn_meta = None, None
        reference_points = reference_points_unact.sigmoid()

        decoder_inputs_dict = dict(
            query=query,
            memory=memory,
            reference_points=reference_points,
            dn_mask=dn_mask)
        # NOTE DINO calculates encoder losses on scores and coordinates
        # of selected top-k encoder queries, while DeformDETR is of all
        # encoder queries.
        head_inputs_dict = dict(
            enc_outputs_class=topk_score,
            enc_outputs_coord=topk_coords,
            dn_meta=dn_meta) if self.training else dict()
        return decoder_inputs_dict, head_inputs_dict
