from typing import Dict, Optional, Tuple

import torch
from mmcv.cnn.bricks.transformer import MultiScaleDeformableAttention
from mmdet.models import DetectionTransformer, DINO
from mmdet.models.layers import CdnQueryGenerator
from mmdet.registry import MODELS
from mmdet.structures import OptSampleList
from mmdet.utils import OptConfigType
from mmengine.model import xavier_init
from torch import Tensor, nn
from torch.nn.init import normal_

from my_modules.detector.my_decoder import MyTransformerDecoder
from my_modules.layers.custom_layers import CustomDeformableDetrTransformerEncoder
from my_modules.layers.positional_encoding import CustomSinePositionalEncoding
from my_modules.layers.pseudo_layers import Pseudo2DLinear, Pseudo4DRegLinear


@MODELS.register_module()
class MyDeformableDETR(DINO):
    """
    The Customized DeformableDETR that:
    1. Replace MultiScaleDeformableAttention with Customized MultiScaleDeformableAttention
    so that the y-axis do NOT function to support Temporal Action Detection (1D).
    2. Replace SingleScaleDeformableAttention with Customized SingleScaleDeformableAttention
    to avoid the meaningless static y-axis position embeddings.
    3. Change implementation details in two-stage:
        3.1: The object queries now comes from a static embedding weight (like in one-stage), rather than inferred
         from encoder's outputs. While the reference points are still inferred from encoder's outputs.
        3.2: The encoder loss now are computed only on the top-k proposals (rather than all proposals),
         and its classification loss is in a multi-class manner (rather than binary classification).
    4. Support de-noising queries.
    """

    def __init__(self,
                 *args,
                 decoder: OptConfigType = None,
                 bbox_head: OptConfigType = None,
                 with_box_refine: bool = False,
                 as_two_stage: bool = False,
                 num_feature_levels: int = 4,
                 dn_cfg: OptConfigType = None,
                 **kwargs) -> None:
        self.with_box_refine = with_box_refine
        self.as_two_stage = as_two_stage
        self.num_feature_levels = num_feature_levels
        self.dn_cfg = dn_cfg
        self.with_dn = dn_cfg is not None
        self.dynamic_pos = False

        if bbox_head is not None:
            assert 'share_pred_layer' not in bbox_head and \
                   'num_pred_layer' not in bbox_head and \
                   'as_two_stage' not in bbox_head, \
                'The two keyword args `share_pred_layer`, `num_pred_layer`, ' \
                'and `as_two_stage are set in `detector.__init__()`, users ' \
                'should not set them in `bbox_head` config.'
            # The last prediction layer is used to generate proposal
            # from encode feature map when `as_two_stage` is `True`.
            # And all the prediction layers should share parameters
            # when `with_box_refine` is `True`.
            bbox_head['share_pred_layer'] = not with_box_refine
            bbox_head['num_pred_layer'] = (decoder['num_layers'] + 1) \
                if self.as_two_stage else decoder['num_layers']
            bbox_head['as_two_stage'] = as_two_stage

        DetectionTransformer.__init__(self, *args, decoder=decoder, bbox_head=bbox_head, **kwargs)

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
        self.encoder = CustomDeformableDetrTransformerEncoder(**self.encoder)
        self.decoder = MyTransformerDecoder(dynamic_pos=self.dynamic_pos, **self.decoder)
        self.positional_encoding = CustomSinePositionalEncoding(**self.positional_encoding)
        self.embed_dims = self.encoder.embed_dims
        num_feats = self.positional_encoding.num_feats
        assert num_feats * 2 == self.embed_dims, \
            'embed_dims should be exactly 2 times of num_feats. ' \
            f'Found {self.embed_dims} and {num_feats}.'
        self.level_embed = nn.Parameter(
            torch.Tensor(self.num_feature_levels, self.embed_dims))

        if self.as_two_stage:
            # Our implementation of two-stage only use encoder output as reference point
            # While the initial object queries are static.
            self.memory_trans_fc = nn.Linear(self.embed_dims, self.embed_dims)
            self.memory_trans_norm = nn.LayerNorm(self.embed_dims)
            # self.pos_trans_fc = nn.Linear(self.embed_dims * 2,
            #                               self.embed_dims * 2)
            # self.pos_trans_norm = nn.LayerNorm(self.embed_dims * 2)
        else:
            if self.with_dn or self.dynamic_pos:
                self.reference_points_fc = Pseudo4DRegLinear(self.embed_dims, delta=False)
            else:
                self.reference_points_fc = Pseudo2DLinear(self.embed_dims, 1)
        # NOTE The query_embedding will be split into query and query_pos
        # in self.pre_decoder, hence, the embed_dims are doubled.
        if self.dynamic_pos:
            self.query_embedding = nn.Embedding(self.num_queries, self.embed_dims)
        else:
            self.query_embedding = nn.Embedding(self.num_queries, self.embed_dims * 2)

    def init_weights(self) -> None:
        """Initialize weights for Transformer and other components."""
        DetectionTransformer.init_weights(self)
        for coder in self.encoder, self.decoder:
            for p in coder.parameters():
                if p.dim() > 1:
                    nn.init.xavier_uniform_(p)
        for m in self.modules():
            if isinstance(m, MultiScaleDeformableAttention):
                m.init_weights()
        if self.as_two_stage:
            nn.init.xavier_uniform_(self.memory_trans_fc.weight)
            # nn.init.xavier_uniform_(self.pos_trans_fc.weight)
        else:
            xavier_init(
                self.reference_points_fc, distribution='uniform', bias=0.)
        normal_(self.level_embed)

    def pre_decoder(
            self,
            memory: Tensor,
            memory_mask: Tensor,
            spatial_shapes: Tensor,
            batch_data_samples: OptSampleList = None,
    ) -> Tuple[Dict]:
        batch_size, _, c = memory.shape
        # Static object queries
        if self.dynamic_pos:
            query = self.query_embedding.weight.unsqueeze(0).expand(batch_size, -1, -1)
            query_pos = None
        else:
            query_embed = self.query_embedding.weight
            query_pos, query = torch.split(query_embed, c, dim=1)
            query_pos = query_pos.unsqueeze(0).expand(batch_size, -1, -1)
            query = query.unsqueeze(0).expand(batch_size, -1, -1)

        if self.as_two_stage:
            output_memory, output_proposals = \
                self.gen_encoder_output_proposals(
                    memory, memory_mask, spatial_shapes)
            enc_outputs_class = self.bbox_head.cls_branches[
                self.decoder.num_layers](
                output_memory)
            enc_outputs_coord_unact = self.bbox_head.reg_branches[
                                          self.decoder.num_layers](output_memory) + output_proposals
            topk_proposals = torch.topk(
                enc_outputs_class.max(-1)[0], self.num_queries, dim=1)[1]
            cls_out_features = self.bbox_head.cls_branches[self.decoder.num_layers].out_features
            topk_scores = torch.gather(
                enc_outputs_class, 1,
                topk_proposals.unsqueeze(-1).repeat(1, 1, cls_out_features))
            topk_coords_unact = torch.gather(
                enc_outputs_coord_unact, 1,
                topk_proposals.unsqueeze(-1).repeat(1, 1, 4))
            topk_coords = topk_coords_unact.sigmoid()
            topk_coords_unact = topk_coords_unact.detach()
            reference_points_unact = topk_coords_unact
            # pos_trans_out = self.pos_trans_fc(
            #     self.get_proposal_pos_embed(topk_coords_unact[..., ::2], num_pos_feats=256))
            # pos_trans_out = self.pos_trans_norm(pos_trans_out)
            # query_pos, query = torch.split(pos_trans_out, c, dim=2)

            # query = self.query_embedding.weight[:, None, :]
            # query = query.repeat(1, batch_size, 1).transpose(0, 1)
        else:
            topk_scores, topk_coords = None, None
            if self.dynamic_pos:
                reference_points_unact = self.reference_points_fc(query)
            else:
                reference_points_unact = self.reference_points_fc(query_pos)

        if self.training and (self.dn_cfg is not None):
            dn_label_query, dn_bbox_query, dn_mask, dn_meta = \
                self.dn_query_generator(batch_data_samples)
            query = torch.cat([dn_label_query, query], dim=1)
            reference_points_unact = torch.cat([dn_bbox_query, reference_points_unact], dim=1)
            if not self.dynamic_pos:
                dn_pos = MyTransformerDecoder.coordinate_to_encoding(dn_bbox_query.sigmoid(),
                                                                     num_feats=self.embed_dims // 2)
                query_pos = torch.cat([dn_pos, query_pos], dim=1)
        else:
            reference_points_unact = reference_points_unact
            dn_mask, dn_meta = None, None
        reference_points = reference_points_unact.sigmoid()
        decoder_inputs_dict = dict(
            query=query,
            query_pos=query_pos,
            memory=memory,
            reference_points=reference_points,
            dn_mask=dn_mask)
        head_inputs_dict = dict(
            enc_outputs_class=topk_scores,
            enc_outputs_coord=topk_coords,
            dn_meta=dn_meta) if self.training else dict()
        return decoder_inputs_dict, head_inputs_dict

    def forward_decoder(self,
                        query: Tensor,
                        query_pos: Tensor,  # DINO does not use query_pos here.
                        memory: Tensor,
                        memory_mask: Tensor,
                        reference_points: Tensor,
                        spatial_shapes: Tensor,
                        level_start_index: Tensor,
                        valid_ratios: Tensor,
                        dn_mask: Optional[Tensor] = None) -> Dict:
        inter_states, references = self.decoder(
            query=query,
            query_pos=query_pos,
            value=memory,
            key_padding_mask=memory_mask,
            self_attn_mask=dn_mask,
            reference_points=reference_points,
            spatial_shapes=spatial_shapes,
            level_start_index=level_start_index,
            valid_ratios=valid_ratios,
            reg_branches=self.bbox_head.reg_branches
            if self.with_box_refine else None)

        if len(query) == self.num_queries:
            # NOTE: This is to make sure label_embeding can be involved to
            # produce loss even if there is no denoising query (no ground truth
            # target in this GPU), otherwise, this will raise runtime error in
            # distributed training.
            inter_states[0] += \
                self.dn_query_generator.label_embedding.weight[0, 0] * 0.0

        decoder_outputs_dict = dict(
            hidden_states=inter_states, references=list(references))
        return decoder_outputs_dict
