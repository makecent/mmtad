from typing import Tuple, Dict, Optional

import torch
from mmcv.cnn.bricks.transformer import MultiScaleDeformableAttention
from mmdet.models import DeformableDETR
from mmdet.models.layers import CdnQueryGenerator
from mmdet.registry import MODELS
from mmdet.structures import OptSampleList
from mmengine.model import xavier_init
from torch import Tensor, nn
from torch.nn.init import normal_

from my_modules.layers.dita_layers import TdtrTransformerEncoder, TdtrTransformerDecoder
from my_modules.layers.positional_encoding import SinePositional1dEncoding
from my_modules.layers.pseudo_layers import PseudoEncoder, Pseudo4DRegLinear


@MODELS.register_module()
class TDTR(DeformableDETR):
    """
    The TadTR implementation based on Deformable DETR. We replace the MultiScaleDeformableAttention
    to support temporal attention.
    Deformable DETR: query_from_enc=True, query_pos_from_enc=True, dynamic_query_pos = False
    DINO: query_from_enc=False, query_pos_from_enc=True, dynamic_query_pos = True
    """

    def __init__(self,
                 query_from_enc=False,
                 query_pos_from_enc=True,
                 dynamic_query_pos=True,
                 dn_cfg=None,
                 *args, **kwargs) -> None:

        self.query_from_enc = query_from_enc
        self.query_pos_from_enc = query_pos_from_enc
        self.dynamic_query_pos = dynamic_query_pos
        super().__init__(*args, **kwargs)
        if not query_from_enc and not query_pos_from_enc:
            assert not self.as_two_stage, \
                'At least one of the decoder query initialization should be from encoder when as_two_stage is True.'
        self.dn_cfg = dn_cfg
        if dn_cfg is not None and self.decoder.deformable:
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
        # Positional encoding
        self.positional_encoding = SinePositional1dEncoding(**self.positional_encoding)

        # Encoder and Decoder
        if self.encoder.get('num_layers', 0) == 0:
            self.encoder = PseudoEncoder()
            self.embed_dims = 256
        else:
            self.encoder = TdtrTransformerEncoder(**self.encoder)
        self.decoder = TdtrTransformerDecoder(dynamic_query_pos=self.dynamic_query_pos, **self.decoder)
        self.embed_dims = self.decoder.embed_dims

        # The initialization of decoder queries
        if self.query_from_enc:
            self.query_fc = nn.Linear(self.embed_dims, self.embed_dims)
            self.query_norm = nn.LayerNorm(self.embed_dims)
        else:
            self.query_embedding = nn.Embedding(self.num_queries, self.embed_dims)

        # The initialization of positional queries of decoder
        if self.query_pos_from_enc:
            if not self.dynamic_query_pos:
                self.query_pos_fc = nn.Linear(self.embed_dims, self.embed_dims)
                self.query_pos_norm = nn.LayerNorm(self.embed_dims)
        else:
            self.query_pos_embedding = nn.Embedding(self.num_queries, self.embed_dims)

        # Level encoding
        self.level_embed = nn.Parameter(torch.Tensor(self.num_feature_levels, self.embed_dims))
        if isinstance(self.encoder, PseudoEncoder):
            self.level_embed.requires_grad_(False)

        # Others
        if self.as_two_stage:
            # if two_stage, encoder output (memory) are projected to construct the initial states of decoder input,
            self.memory_trans_fc = nn.Linear(self.embed_dims, self.embed_dims)
            self.memory_trans_norm = nn.LayerNorm(self.embed_dims)
            self.proposals = nn.Embedding(480, self.embed_dims)
            self.enc_fc = Pseudo4DRegLinear(self.embed_dims, delta=False)

        elif self.decoder.deformable:
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
        elif self.decoder.deformable:
            xavier_init(self.reference_points_fc, distribution='uniform', bias=0.)
        if self.query_from_enc:
            nn.init.xavier_uniform_(self.query_fc.weight)
        if self.query_pos_from_enc and not self.dynamic_query_pos:
            nn.init.xavier_uniform_(self.query_pos_fc.weight)

        normal_(self.level_embed)

    def pre_transformer(
            self,
            mlvl_feats: Tuple[Tensor],
            batch_data_samples: OptSampleList = None) -> Tuple[Dict]:
        encoder_inputs_dict, decoder_inputs_dict = super().pre_transformer(mlvl_feats, batch_data_samples)
        decoder_inputs_dict['memory_pos'] = encoder_inputs_dict['feat_pos']
        return encoder_inputs_dict, decoder_inputs_dict

    def forward_transformer(self, img_feats: Tuple[Tensor],
                            batch_data_samples: OptSampleList = None) -> Dict:

        encoder_inputs_dict, decoder_inputs_dict = self.pre_transformer(img_feats, batch_data_samples)

        encoder_outputs_dict = self.forward_encoder(**encoder_inputs_dict)

        tmp_dec_in, head_inputs_dict = self.pre_decoder(
            **encoder_outputs_dict, batch_data_samples=batch_data_samples)
        decoder_inputs_dict.update(tmp_dec_in)

        decoder_outputs_dict = self.forward_decoder(**decoder_inputs_dict)
        head_inputs_dict.update(decoder_outputs_dict)
        return head_inputs_dict

    def pre_decoder(self, memory: Tensor, memory_mask: Tensor, spatial_shapes: Tensor,
                    batch_data_samples: OptSampleList = None) -> Tuple[Dict, Dict]:
        batch_size, _, c = memory.shape

        # %% In two_stage, the initial decoder reference points are based on encoder output
        if self.as_two_stage:
            # get the encoder memory and the initial proposals. the memory is the encoder output features, but
            # processed by a projection layer. the proposals are pre-defined anchors depends on the input spatial shapes
            # , and they are inverse-normalized, i.e., proposals.sigmoid() = coordinates_of_the_proposals
            output_memory, output_proposals = self.gen_encoder_output_proposals(memory, memory_mask, spatial_shapes)
            # get the predicted classification scores of the proposals by input memory to the detection head
            enc_outputs_class = self.bbox_head.cls_branches[self.decoder.num_layers](output_memory)
            # get the predicted bbox offsets of the proposals by input memory to the detection head
            enc_outputs_offset_unact = self.bbox_head.reg_branches[self.decoder.num_layers](output_memory)
            # add the predicted bbox offsets and the proposals to get the predicted coordinates (inverse-normalized)
            output_proposals = self.proposals.weight.unsqueeze(0).expand(batch_size, -1, -1)
            output_proposals = self.enc_fc(output_proposals)
            enc_outputs_coord_unact = enc_outputs_offset_unact + output_proposals
            # use the coordinates of proposals of top-k classification scores as the initial decoder reference points
            topk_proposals_indices = torch.topk(enc_outputs_class.max(-1)[0], self.num_queries, dim=1)[1]
            topk_coords_unact = torch.gather(enc_outputs_coord_unact, 1,
                                             topk_proposals_indices.unsqueeze(-1).repeat(1, 1, 4))
            topk_coords = topk_coords_unact.sigmoid()
            topk_coords_unact = topk_coords_unact.detach()
            reference_points_unact = topk_coords_unact
            # the top-k scores and coordinates are packed to be input to the detection head for computing encoder losses
            cls_out_features = self.bbox_head.cls_branches[self.decoder.num_layers].out_features
            topk_scores = torch.gather(enc_outputs_class, 1,
                                       topk_proposals_indices.unsqueeze(-1).repeat(1, 1, cls_out_features))

        # %% Compute initial decoder query and query_pos
        if self.query_from_enc:
            pos_trans_out = self.query_fc(
                self.get_proposal_pos_embed(topk_coords_unact[..., ::2], num_pos_feats=self.embed_dims // 2))
            query = self.query_norm(pos_trans_out)
        else:
            query = self.query_embedding.weight.unsqueeze(0).expand(batch_size, -1, -1)

        if self.query_pos_from_enc:
            if not self.dynamic_query_pos:
                pos_trans_out = self.query_pos_fc(
                    self.get_proposal_pos_embed(topk_coords_unact[..., ::2], num_pos_feats=self.embed_dims // 2))
                query_pos = self.query_pos_norm(pos_trans_out)
        else:
            query_pos = self.query_pos_embedding.weight.unsqueeze(0).expand(batch_size, -1, -1)

        # %% In one stage, the initial reference points of the decoder are derived by projecting the query embeddings.
        if not self.as_two_stage:
            topk_scores, topk_coords = None, None
            if self.decoder.deformable:
                reference_points_unact = self.reference_points_fc(query_pos)

        # %% De-noising training if triggered
        if self.training and (self.dn_cfg is not None) and self.decoder.deformable:
            # %% when de-noising training is enabled, generate dn_label_query and dn_bbox_query (inverse-sigmoid)
            # by adding random noise to ground truth labels and bboxes
            dn_label_query, dn_bbox_query, dn_mask, dn_meta = self.dn_query_generator(batch_data_samples)
            # %% concat the dn_label_query and dn_bbox_query with the decoder query and reference points, respectively,
            # and input them together to the model
            query = torch.cat([dn_label_query, query], dim=1)
            reference_points_unact = torch.cat([dn_bbox_query, reference_points_unact], dim=1)
            # %% if the decoder query_pos is computed based on reference_points, then NO de-noising query_pos is needed
            # as the reference points already contain the dn_bbox_query. Otherwise, compute dn_query_pos based on
            # the dn_bbox_query's coordinates and concat it with decoder query_pos
            if not self.dynamic_query_pos:
                dn_query_pos = TdtrTransformerDecoder.coordinate_to_encoding(dn_bbox_query.sigmoid(),
                                                                             num_feats=self.embed_dims // 2)
                query_pos = torch.cat([dn_query_pos, query_pos], dim=1)
        else:
            dn_mask, dn_meta = None, None

        # %% Convert reference points to coordinates and pack the decoder input
        decoder_inputs_dict = dict(
            query=query,
            query_pos=None if self.dynamic_query_pos else query_pos,
            memory=memory,
            reference_points=reference_points_unact.sigmoid() if self.decoder.deformable else torch.rand(1, 2, 3),
            dn_mask=dn_mask)
        head_inputs_dict = dict(
            enc_outputs_class=topk_scores,
            enc_outputs_coord=topk_coords,
            dn_meta=dn_meta) if self.training else dict()
        return decoder_inputs_dict, head_inputs_dict

    def forward_decoder(self,
                        query: Tensor,
                        query_pos: Tensor,
                        memory: Tensor,
                        memory_pos: Tensor,
                        memory_mask: Tensor,
                        reference_points: Tensor,
                        spatial_shapes: Tensor,
                        level_start_index: Tensor,
                        valid_ratios: Tensor,
                        dn_mask: Optional[Tensor] = None,
                        **kwargs) -> Dict:
        inter_states, references = self.decoder(
            query=query,
            value=memory,
            query_pos=query_pos,
            key_pos=memory_pos,
            key_padding_mask=memory_mask,
            self_attn_mask=dn_mask,
            reference_points=reference_points,
            spatial_shapes=spatial_shapes,
            level_start_index=level_start_index,
            valid_ratios=valid_ratios,
            reg_branches=self.bbox_head.reg_branches if self.with_box_refine else None,
            **kwargs)

        if len(query) == self.num_queries and self.dn_cfg is not None:
            # NOTE: This is to make sure label_embeding can be involved to
            # produce loss even if there is no denoising query (no ground truth
            # target in this GPU), otherwise, this will raise runtime error in
            # distributed training.
            inter_states[0] += \
                self.dn_query_generator.label_embedding.weight[0, 0] * 0.0

        decoder_outputs_dict = dict(
            hidden_states=inter_states, references=list(references))
        return decoder_outputs_dict

    #
    def predict(self,
                batch_inputs: Tensor,
                batch_data_samples=None,
                rescale: bool = True):
        """Modified for computing flops, removing the batch_data_samples from positional arguments"""
        img_feats = self.extract_feat(batch_inputs)
        if batch_data_samples is None:
            from mmdet.structures import DetDataSample
            width = batch_inputs.shape[-1]
            dummy = DetDataSample(metainfo=dict(
                img_shape=(1, width),
                batch_input_shape=(1, width),
                scale_factor=(1.0, 1.0)
            ))
            batch_data_samples = [dummy]
        head_inputs_dict = self.forward_transformer(img_feats,
                                                    batch_data_samples)
        results_list = self.bbox_head.predict(
            **head_inputs_dict,
            rescale=rescale,
            batch_data_samples=batch_data_samples)
        batch_data_samples = self.add_pred_to_datasample(
            batch_data_samples, results_list)
        return batch_data_samples