import math
from typing import Dict, List, Tuple, Union
from typing import Optional

import torch
import torch.nn.functional as F
from mmcv.cnn.bricks.transformer import MultiScaleDeformableAttention
from mmdet.models.layers import CdnQueryGenerator
from mmdet.registry import MODELS
from mmdet.structures import OptSampleList, SampleList
from mmdet.utils import ConfigType, OptConfigType, OptMultiConfig
from mmengine.model import xavier_init
from mmdet.models import BaseDetector
from torch import Tensor
from torch import nn
from torch.nn.init import normal_

from my_modules.layers.dita_layers import TdtrTransformerEncoder, DitaTransformerDecoder
from my_modules.layers.positional_encoding import SinePositional1dEncoding
from my_modules.layers.pseudo_layers import Pseudo2DLinear, Pseudo4DRegLinear


@MODELS.register_module()
class DITA(BaseDetector):
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
                 backbone: ConfigType,
                 neck: OptConfigType = None,
                 encoder: OptConfigType = None,
                 decoder: OptConfigType = None,
                 bbox_head: OptConfigType = None,
                 positional_encoding: OptConfigType = None,
                 num_queries: int = 100,
                 with_box_refine: bool = True,
                 as_two_stage: bool = False,
                 static_query=True,
                 static_query_pos=True,
                 num_feature_levels: int = 4,
                 dn_cfg: OptConfigType = None,
                 train_cfg: OptConfigType = None,
                 test_cfg: OptConfigType = None,
                 data_preprocessor: OptConfigType = None,
                 init_cfg: OptMultiConfig = None) -> None:

        super().__init__(
            data_preprocessor=data_preprocessor, init_cfg=init_cfg)
        self.with_box_refine = with_box_refine
        self.as_two_stage = as_two_stage
        self.static_query = static_query
        self.static_query_pos = static_query_pos
        self.num_feature_levels = num_feature_levels
        self.dn_cfg = dn_cfg
        self.with_dn = dn_cfg is not None
        self.dynamic_query_pos = False

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

        # process args
        bbox_head.update(train_cfg=train_cfg)
        bbox_head.update(test_cfg=test_cfg)
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self.encoder = encoder
        self.decoder = decoder
        self.positional_encoding = positional_encoding
        self.num_queries = num_queries

        # init model layers
        self.backbone = MODELS.build(backbone)
        if neck is not None:
            self.neck = MODELS.build(neck)
        self.bbox_head = MODELS.build(bbox_head)
        self._init_layers()

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
        self.positional_encoding = SinePositional1dEncoding(**self.positional_encoding)
        self.encoder = TdtrTransformerEncoder(**self.encoder)
        self.decoder = DitaTransformerDecoder(dynamic_query_pos=self.dynamic_query_pos, **self.decoder)
        self.embed_dims = self.encoder.embed_dims
        num_feats = self.positional_encoding.num_feats
        assert num_feats == self.embed_dims, \
            'embed_dims should equal to the of num_feats of tje positional_encoding layer. ' \
            f'Found {self.embed_dims} and {num_feats}.'

        self.level_embed = nn.Parameter(torch.Tensor(self.num_feature_levels, self.embed_dims))

        if self.as_two_stage:
            # if two_stage, encoder output (memory) are projected to affect the initial states of decoder input,
            # it could include the initial reference points, the initial decoder query and decoder query_pos.
            self.memory_trans_fc = nn.Linear(self.embed_dims, self.embed_dims)
            self.memory_trans_norm = nn.LayerNorm(self.embed_dims)
        else:
            # In one-stage, the initial decoder query and query_pos are trainable static embeddings,
            # and the initial reference points are projected from initial query embeddings
            assert self.static_query and self.static_query_pos
            if self.with_dn or self.dynamic_query_pos:
                self.reference_points_fc = Pseudo4DRegLinear(self.embed_dims, delta=False)
            else:
                self.reference_points_fc = Pseudo2DLinear(self.embed_dims, 1)

        if self.static_query:
            # if self.static_query, the initial decoder query is a trainable embedding
            # which does NOT refer to the output of encoder.
            self.query_embedding = nn.Embedding(self.num_queries, self.embed_dims)
        else:
            self.query_fc = nn.Linear(self.embed_dims * 2, self.embed_dims * 2)
            self.query_norm = nn.LayerNorm(self.embed_dims * 2)

        if self.static_query_pos:
            # if self.static_query_pos, the initial decoder query_pos is a trainable embedding
            # which does NOT refer to the output of encoder.
            self.query_pos_embedding = nn.Embedding(self.num_queries, self.embed_dims)
        elif not self.dynamic_query_pos:
            self.query_pos_fc = nn.Linear(self.embed_dims, self.embed_dims)
            self.query_pos_norm = nn.LayerNorm(self.embed_dims)
        else:
            # if dynamic_query_pos, the query_pos is projected from reference points in every decoder layer.
            pass

    def init_weights(self) -> None:
        """Initialize weights for Transformer and other components."""
        super().init_weights()
        for coder in self.encoder, self.decoder:
            for p in coder.parameters():
                if p.dim() > 1:
                    nn.init.xavier_uniform_(p)
        for m in self.modules():
            if isinstance(m, MultiScaleDeformableAttention):
                m.init_weights()
        if self.as_two_stage:
            nn.init.xavier_uniform_(self.memory_trans_fc.weight)
            if self.static_query:
                nn.init.xavier_uniform_(self.pos_trans_fc.weight)
        else:
            xavier_init(
                self.reference_points_fc, distribution='uniform', bias=0.)
        normal_(self.level_embed)

    def loss(self, batch_inputs: Tensor,
             batch_data_samples: SampleList) -> Union[dict, list]:
        """Calculate losses from a batch of inputs and data samples.

        Args:
            batch_inputs (Tensor): Input images of shape (bs, dim, H, W).
                These should usually be mean centered and std scaled.
            batch_data_samples (List[:obj:`DetDataSample`]): The batch
                data samples. It usually includes information such
                as `gt_instance` or `gt_panoptic_seg` or `gt_sem_seg`.

        Returns:
            dict: A dictionary of loss components
        """
        img_feats = self.extract_feat(batch_inputs)
        head_inputs_dict = self.forward_transformer(img_feats,
                                                    batch_data_samples)
        losses = self.bbox_head.loss(
            **head_inputs_dict, batch_data_samples=batch_data_samples)

        return losses

    def predict(self,
                batch_inputs: Tensor,
                batch_data_samples: SampleList,
                rescale: bool = True) -> SampleList:
        """Predict results from a batch of inputs and data samples with post-
        processing.

        Args:
            batch_inputs (Tensor): Inputs, has shape (bs, dim, H, W).
            batch_data_samples (List[:obj:`DetDataSample`]): The batch
                data samples. It usually includes information such
                as `gt_instance` or `gt_panoptic_seg` or `gt_sem_seg`.
            rescale (bool): Whether to rescale the results.
                Defaults to True.

        Returns:
            list[:obj:`DetDataSample`]: Detection results of the input images.
            Each DetDataSample usually contain 'pred_instances'. And the
            `pred_instances` usually contains following keys.

            - scores (Tensor): Classification scores, has a shape
              (num_instance, )
            - labels (Tensor): Labels of bboxes, has a shape
              (num_instances, ).
            - bboxes (Tensor): Has a shape (num_instances, 4),
              the last dimension 4 arrange as (x1, y1, x2, y2).
        """
        img_feats = self.extract_feat(batch_inputs)
        head_inputs_dict = self.forward_transformer(img_feats,
                                                    batch_data_samples)
        results_list = self.bbox_head.predict(
            **head_inputs_dict,
            rescale=rescale,
            batch_data_samples=batch_data_samples)
        batch_data_samples = self.add_pred_to_datasample(
            batch_data_samples, results_list)
        return batch_data_samples

    def _forward(
            self,
            batch_inputs: Tensor,
            batch_data_samples: OptSampleList = None) -> Tuple[List[Tensor]]:
        """Network forward process. Usually includes backbone, neck and head
        forward without any post-processing.

         Args:
            batch_inputs (Tensor): Inputs, has shape (bs, dim, H, W).
            batch_data_samples (List[:obj:`DetDataSample`], optional): The
                batch data samples. It usually includes information such
                as `gt_instance` or `gt_panoptic_seg` or `gt_sem_seg`.
                Defaults to None.

        Returns:
            tuple[Tensor]: A tuple of features from ``bbox_head`` forward.
        """
        img_feats = self.extract_feat(batch_inputs)
        head_inputs_dict = self.forward_transformer(img_feats,
                                                    batch_data_samples)
        results = self.bbox_head.forward(**head_inputs_dict)
        return results

    def extract_feat(self, batch_inputs: Tensor) -> Tuple[Tensor]:
        """Extract features.

        Args:
            batch_inputs (Tensor): Image tensor, has shape (bs, dim, H, W).

        Returns:
            tuple[Tensor]: Tuple of feature maps from neck. Each feature map
            has shape (bs, dim, H, W).
        """
        x = self.backbone(batch_inputs)
        if self.with_neck:
            x = self.neck(x)
        return x

    def forward_transformer(self,
                            img_feats: Tuple[Tensor],
                            batch_data_samples: OptSampleList = None) -> Dict:
        """Forward process of Transformer, which includes four steps:
        'pre_transformer' -> 'encoder' -> 'pre_decoder' -> 'decoder'. We
        summarized the parameters flow of the existing DETR-like detector,
        which can be illustrated as follow:

        .. code:: text

                 img_feats & batch_data_samples
                               |
                               V
                      +-----------------+
                      | pre_transformer |
                      +-----------------+
                          |          |
                          |          V
                          |    +-----------------+
                          |    | forward_encoder |
                          |    +-----------------+
                          |             |
                          |             V
                          |     +---------------+
                          |     |  pre_decoder  |
                          |     +---------------+
                          |         |       |
                          V         V       |
                      +-----------------+   |
                      | forward_decoder |   |
                      +-----------------+   |
                                |           |
                                V           V
                               head_inputs_dict

        Args:
            img_feats (tuple[Tensor]): Tuple of feature maps from neck. Each
                    feature map has shape (bs, dim, H, W).
            batch_data_samples (list[:obj:`DetDataSample`], optional): The
                batch data samples. It usually includes information such
                as `gt_instance` or `gt_panoptic_seg` or `gt_sem_seg`.
                Defaults to None.

        Returns:
            dict: The dictionary of bbox_head function inputs, which always
            includes the `hidden_states` of the decoder output and may contain
            `references` including the initial and intermediate references.
        """
        encoder_inputs_dict, decoder_inputs_dict = self.pre_transformer(
            img_feats, batch_data_samples)

        encoder_outputs_dict = self.forward_encoder(**encoder_inputs_dict)

        tmp_dec_in, head_inputs_dict = self.pre_decoder(**encoder_outputs_dict, batch_data_samples=batch_data_samples)
        decoder_inputs_dict.update(tmp_dec_in)

        decoder_outputs_dict = self.forward_decoder(**decoder_inputs_dict)
        head_inputs_dict.update(decoder_outputs_dict)
        return head_inputs_dict

    def pre_transformer(
            self,
            mlvl_feats: Tuple[Tensor],
            batch_data_samples: OptSampleList = None) -> Tuple[Dict, Dict]:
        """Process image features before feeding them to the transformer.

        The forward procedure of the transformer is defined as:
        'pre_transformer' -> 'encoder' -> 'pre_decoder' -> 'decoder'
        More details can be found at `TransformerDetector.forward_transformer`
        in `mmdet/detector/base_detr.py`.

        Args:
            mlvl_feats (tuple[Tensor]): Multi-level features that may have
                different resolutions, output from neck. Each feature has
                shape (bs, dim, h_lvl, w_lvl), where 'lvl' means 'layer'.
            batch_data_samples (list[:obj:`DetDataSample`], optional): The
                batch data samples. It usually includes information such
                as `gt_instance` or `gt_panoptic_seg` or `gt_sem_seg`.
                Defaults to None.

        Returns:
            tuple[dict]: The first dict contains the inputs of encoder and the
            second dict contains the inputs of decoder.

            - encoder_inputs_dict (dict): The keyword args dictionary of
              `self.forward_encoder()`, which includes 'feat', 'feat_mask',
              and 'feat_pos'.
            - decoder_inputs_dict (dict): The keyword args dictionary of
              `self.forward_decoder()`, which includes 'memory_mask'.
        """
        batch_size = mlvl_feats[0].size(0)

        # %% construct binary masks and the corresponding masked positional embeddings
        # to nullify padded positions
        assert batch_data_samples is not None
        batch_input_shape = batch_data_samples[0].batch_input_shape
        input_img_h, input_img_w = batch_input_shape
        img_shape_list = [sample.img_shape for sample in batch_data_samples]
        same_shape_flag = all([
            s[0] == input_img_h and s[1] == input_img_w for s in img_shape_list
        ])
        # support torch2onnx without feeding masks
        if torch.onnx.is_in_onnx_export() or same_shape_flag:
            mlvl_masks = []
            mlvl_pos_embeds = []
            for feat in mlvl_feats:
                mlvl_masks.append(None)
                mlvl_pos_embeds.append(
                    self.positional_encoding(None, input=feat))
        else:
            masks = mlvl_feats[0].new_ones(
                (batch_size, input_img_h, input_img_w))
            for img_id in range(batch_size):
                img_h, img_w = img_shape_list[img_id]
                masks[img_id, :img_h, :img_w] = 0
            # NOTE following the official DETR repo, non-zero
            # values representing ignored positions, while
            # zero values means valid positions.

            mlvl_masks = []
            mlvl_pos_embeds = []
            for feat in mlvl_feats:
                mlvl_masks.append(
                    F.interpolate(masks[None], size=feat.shape[-2:]).to(
                        torch.bool).squeeze(0))
                mlvl_pos_embeds.append(
                    self.positional_encoding(mlvl_masks[-1]))
        # %% flat the multi-level features, positional embeddings, and masks.
        feat_flatten = []
        lvl_pos_embed_flatten = []
        mask_flatten = []
        spatial_shapes = []
        for lvl, (feat, mask, pos_embed) in enumerate(
                zip(mlvl_feats, mlvl_masks, mlvl_pos_embeds)):
            batch_size, c, h, w = feat.shape
            spatial_shape = torch._shape_as_tensor(feat)[2:].to(feat.device)
            # [bs, c, h_lvl, w_lvl] -> [bs, h_lvl*w_lvl, c]
            feat = feat.view(batch_size, c, -1).permute(0, 2, 1)
            pos_embed = pos_embed.view(batch_size, c, -1).permute(0, 2, 1)
            lvl_pos_embed = pos_embed + self.level_embed[lvl].view(1, 1, -1)
            # [bs, h_lvl, w_lvl] -> [bs, h_lvl*w_lvl]
            if mask is not None:
                mask = mask.flatten(1)

            feat_flatten.append(feat)
            lvl_pos_embed_flatten.append(lvl_pos_embed)
            mask_flatten.append(mask)
            spatial_shapes.append(spatial_shape)

        # (bs, num_feat_points, dim)
        feat_flatten = torch.cat(feat_flatten, 1)
        lvl_pos_embed_flatten = torch.cat(lvl_pos_embed_flatten, 1)
        # (bs, num_feat_points), where num_feat_points = sum_lvl(h_lvl*w_lvl)
        if mask_flatten[0] is not None:
            mask_flatten = torch.cat(mask_flatten, 1)
        else:
            mask_flatten = None

        # %% record the level_start_index, spatial shapes and valid_ratios for the convenience of future process.
        # (num_level, 2)
        spatial_shapes = torch.cat(spatial_shapes).view(-1, 2)
        level_start_index = torch.cat((
            spatial_shapes.new_zeros((1,)),  # (num_level)
            spatial_shapes.prod(1).cumsum(0)[:-1]))
        if mlvl_masks[0] is not None:
            valid_ratios = torch.stack(  # (bs, num_level, 2)
                [self.get_valid_ratio(m) for m in mlvl_masks], 1)
        else:
            valid_ratios = mlvl_feats[0].new_ones(batch_size, len(mlvl_feats),
                                                  2)

        encoder_inputs_dict = dict(
            feat=feat_flatten,
            feat_mask=mask_flatten,
            feat_pos=lvl_pos_embed_flatten,
            spatial_shapes=spatial_shapes,
            level_start_index=level_start_index,
            valid_ratios=valid_ratios)
        decoder_inputs_dict = dict(
            memory_mask=mask_flatten,
            spatial_shapes=spatial_shapes,
            level_start_index=level_start_index,
            valid_ratios=valid_ratios)
        return encoder_inputs_dict, decoder_inputs_dict

    def forward_encoder(self, feat: Tensor, feat_mask: Tensor,
                        feat_pos: Tensor, spatial_shapes: Tensor,
                        level_start_index: Tensor,
                        valid_ratios: Tensor) -> Dict:
        """Forward with Transformer encoder.

        The forward procedure of the transformer is defined as:
        'pre_transformer' -> 'encoder' -> 'pre_decoder' -> 'decoder'
        More details can be found at `TransformerDetector.forward_transformer`
        in `mmdet/detector/base_detr.py`.

        Args:
            feat (Tensor): Sequential features, has shape (bs, num_feat_points,
                dim).
            feat_mask (Tensor): ByteTensor, the padding mask of the features,
                has shape (bs, num_feat_points).
            feat_pos (Tensor): The positional embeddings of the features, has
                shape (bs, num_feat_points, dim).
            spatial_shapes (Tensor): Spatial shapes of features in all levels,
                has shape (num_levels, 2), last dimension represents (h, w).
            level_start_index (Tensor): The start index of each level.
                A tensor has shape (num_levels, ) and can be represented
                as [0, h_0*w_0, h_0*w_0+h_1*w_1, ...].
            valid_ratios (Tensor): The ratios of the valid width and the valid
                height relative to the width and the height of features in all
                levels, has shape (bs, num_levels, 2).

        Returns:
            dict: The dictionary of encoder outputs, which includes the
            `memory` of the encoder output.
        """
        memory = self.encoder(
            query=feat,
            query_pos=feat_pos,
            key_padding_mask=feat_mask,  # for self_attn
            spatial_shapes=spatial_shapes,
            level_start_index=level_start_index,
            valid_ratios=valid_ratios)
        encoder_outputs_dict = dict(
            memory=memory,
            memory_mask=feat_mask,
            spatial_shapes=spatial_shapes)
        return encoder_outputs_dict

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
            enc_outputs_coord_unact = enc_outputs_offset_unact + output_proposals
            # use the coordinates of proposals of top-k classification scores as the initial decoder reference points
            topk_proposals_indices = torch.topk(enc_outputs_class.max(-1)[0], self.num_queries, dim=1)[1]
            topk_coords_unact = torch.gather(enc_outputs_coord_unact, 1,
                                             topk_proposals_indices.unsqueeze(-1).repeat(1, 1, 4))
            topk_coords_unact = topk_coords_unact.detach()
            reference_points_unact = topk_coords_unact
            # the top-k scores and coordinates are packed to be input to the detection head for computing encoder losses
            cls_out_features = self.bbox_head.cls_branches[self.decoder.num_layers].out_features
            topk_scores = torch.gather(enc_outputs_class, 1,
                                       topk_proposals_indices.unsqueeze(-1).repeat(1, 1, cls_out_features))
            topk_coords = topk_coords_unact.sigmoid()

        # %% Compute initial decoder query and query_pos
        if self.static_query:
            query = self.query_embedding.weight.unsqueeze(0).expand(batch_size, -1, -1)
        else:
            pos_trans_out = self.query_fc(
                self.get_proposal_pos_embed(topk_coords_unact[..., ::2], num_pos_feats=256))
            query = self.query_norm(pos_trans_out)

        if self.static_query_pos:
            query_pos = self.query_pos_embedding.weight.unsqueeze(0).expand(batch_size, -1, -1)
        elif not self.dynamic_query_pos:
            pos_trans_out = self.query_pos_fc(
                self.get_proposal_pos_embed(topk_coords_unact[..., ::2], num_pos_feats=256))
            query_pos = self.query_pos_norm(pos_trans_out)
        else:
            query_pos = None

        # %% In one stage, the initial reference points of the decoder are derived by projecting the query embeddings.
        if not self.as_two_stage:
            topk_scores, topk_coords = None, None
            if self.dynamic_query_pos:
                reference_points_unact = self.reference_points_fc(query)
            else:
                reference_points_unact = self.reference_points_fc(query_pos)

        # %% De-noising training if triggered
        if self.training and (self.dn_cfg is not None):
            # %% when de-noising training is enabled, generate dn_label_query and dn_bbox_query (inverse-normalized)
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
                dn_query_pos = DitaTransformerDecoder.coordinate_to_encoding(dn_bbox_query.sigmoid(),
                                                                             num_feats=self.embed_dims // 2)
                query_pos = torch.cat([dn_query_pos, query_pos], dim=1)
        else:
            reference_points_unact = reference_points_unact
            dn_mask, dn_meta = None, None

        # %% Convert reference points to coordinates and pack the decoder input
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
                        query_pos: Tensor,
                        memory: Tensor,
                        memory_mask: Tensor,
                        reference_points: Tensor,
                        spatial_shapes: Tensor,
                        level_start_index: Tensor,
                        valid_ratios: Tensor,
                        dn_mask: Optional[Tensor] = None,
                        **kwargs) -> Dict:
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
            if self.with_box_refine else None,
            **kwargs)

        if len(query) == self.num_queries:
            # NOTE: This is to make sure label_embeding can be involved to
            # produce loss even if there is no denoising query (no ground truth
            # target in this GPU), otherwise, this will raise runtime error in
            # distributed training.
            inter_states[0] += self.dn_query_generator.label_embedding.weight[0, 0] * 0.0

        decoder_outputs_dict = dict(hidden_states=inter_states, references=list(references))
        return decoder_outputs_dict

    def gen_encoder_output_proposals(
            self, memory: Tensor, memory_mask: Tensor,
            spatial_shapes: Tensor) -> Tuple[Tensor, Tensor]:
        """Generate proposals from encoded memory. The function will only be
        used when `as_two_stage` is `True`.

        Args:
            memory (Tensor): The output embeddings of the Transformer encoder,
                has shape (bs, num_feat_points, dim).
            memory_mask (Tensor): ByteTensor, the padding mask of the memory,
                has shape (bs, num_feat_points).
            spatial_shapes (Tensor): Spatial shapes of features in all levels,
                has shape (num_levels, 2), last dimension represents (h, w).

        Returns:
            tuple: A tuple of transformed memory and proposals.

            - output_memory (Tensor): The transformed memory for obtaining
              top-k proposals, has shape (bs, num_feat_points, dim).
            - output_proposals (Tensor): The inverse-normalized proposal, has
              shape (batch_size, num_keys, 4) with the last dimension arranged
              as (cx, cy, w, h).
        """

        bs = memory.size(0)
        proposals = []
        _cur = 0  # start index in the sequence of the current level
        for lvl, HW in enumerate(spatial_shapes):
            H, W = HW

            if memory_mask is not None:
                mask_flatten_ = memory_mask[:, _cur:(_cur + H * W)].view(
                    bs, H, W, 1)
                valid_H = torch.sum(~mask_flatten_[:, :, 0, 0],
                                    1).unsqueeze(-1)
                valid_W = torch.sum(~mask_flatten_[:, 0, :, 0],
                                    1).unsqueeze(-1)
                scale = torch.cat([valid_W, valid_H], 1).view(bs, 1, 1, 2)
            else:
                if not isinstance(HW, torch.Tensor):
                    HW = memory.new_tensor(HW)
                scale = HW.unsqueeze(0).flip(dims=[0, 1]).view(1, 1, 1, 2)
            grid_y, grid_x = torch.meshgrid(
                torch.linspace(
                    0, H - 1, H, dtype=torch.float32, device=memory.device),
                torch.linspace(
                    0, W - 1, W, dtype=torch.float32, device=memory.device))
            grid = torch.cat([grid_x.unsqueeze(-1), grid_y.unsqueeze(-1)], -1)
            grid = (grid.unsqueeze(0).expand(bs, -1, -1, -1) + 0.5) / scale
            wh = torch.ones_like(grid) * 0.05 * (2.0 ** lvl)
            proposal = torch.cat((grid, wh), -1).view(bs, -1, 4)
            proposals.append(proposal)
            _cur += (H * W)
        output_proposals = torch.cat(proposals, 1)
        # do not use `all` to make it exportable to onnx
        output_proposals_valid = (
                                         (output_proposals > 0.01) & (output_proposals < 0.99)).sum(
            -1, keepdim=True) == output_proposals.shape[-1]
        # inverse_sigmoid
        output_proposals = torch.log(output_proposals / (1 - output_proposals))
        if memory_mask is not None:
            output_proposals = output_proposals.masked_fill(
                memory_mask.unsqueeze(-1), float('inf'))
        output_proposals = output_proposals.masked_fill(
            ~output_proposals_valid, float('inf'))

        output_memory = memory
        if memory_mask is not None:
            output_memory = output_memory.masked_fill(
                memory_mask.unsqueeze(-1), float(0))
        output_memory = output_memory.masked_fill(~output_proposals_valid,
                                                  float(0))
        output_memory = self.memory_trans_fc(output_memory)
        output_memory = self.memory_trans_norm(output_memory)
        # [bs, sum(hw), 2]
        return output_memory, output_proposals

    @staticmethod
    def get_proposal_pos_embed(proposals: Tensor,
                               num_pos_feats: int = 128,
                               temperature: int = 10000) -> Tensor:
        """Get the position embedding of the proposal.

        Args:
            proposals (Tensor): Not normalized proposals, has shape
                (bs, num_queries, 4) with the last dimension arranged as
                (cx, cy, w, h).
            num_pos_feats (int, optional): The feature dimension for each
                position along x, y, w, and h-axis. Note the final returned
                dimension for each position is 4 times of num_pos_feats.
                Default to 128.
            temperature (int, optional): The temperature used for scaling the
                position embedding. Defaults to 10000.

        Returns:
            Tensor: The position embedding of proposal, has shape
            (bs, num_queries, num_pos_feats * 4), with the last dimension
            arranged as (cx, cy, w, h)
        """
        scale = 2 * math.pi
        dim_t = torch.arange(
            num_pos_feats, dtype=torch.float32, device=proposals.device)
        dim_t = temperature ** (2 * (dim_t // 2) / num_pos_feats)
        # N, L, 4
        proposals = proposals.sigmoid() * scale
        # N, L, 4, 128
        pos = proposals[:, :, :, None] / dim_t
        # N, L, 4, 64, 2
        pos = torch.stack((pos[:, :, :, 0::2].sin(), pos[:, :, :, 1::2].cos()),
                          dim=4).flatten(2)
        return pos
