from mmdet.models.dense_heads import DETRHead
from mmdet.registry import MODELS

from my_modules.layers.pseudo_layers import Pseudo4DRegLinear


@MODELS.register_module()
class DETR_TADHead(DETRHead):
    """
    TadTR head.
    We modify the regression branches to output 2 (x1, x2) rather than 4 (x1, y1, x2, y2).
    """

    def __init__(self,
                 share_pred_layer=False,
                 num_pred_layer=1,
                 as_two_stage=False,
                 *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _init_layers(self) -> None:
        """Initialize layers of the transformer head."""
        super()._init_layers()
        self.fc_reg = Pseudo4DRegLinear(self.embed_dims, delta=False)

    def loss(self, hidden_states, references,
             enc_outputs_class, enc_outputs_coord,
             batch_data_samples, dn_meta) -> dict:
        batch_gt_instances = []
        batch_img_metas = []
        for data_sample in batch_data_samples:
            batch_img_metas.append(data_sample.metainfo)
            batch_gt_instances.append(data_sample.gt_instances)

        outs = self(hidden_states, references)
        loss_inputs = outs + (batch_gt_instances, batch_img_metas)
        losses = self.loss_by_feat(*loss_inputs)
        return losses

    def predict(self,
                hidden_states,
                references,
                batch_data_samples,
                rescale: bool = True) :
        batch_img_metas = [
            data_samples.metainfo for data_samples in batch_data_samples
        ]

        last_layer_hidden_state = hidden_states[-1].unsqueeze(0)
        outs = self(last_layer_hidden_state, references)

        predictions = self.predict_by_feat(
            *outs, batch_img_metas=batch_img_metas, rescale=rescale)

        return predictions
    def forward(self, hidden_states, references):
        layers_cls_scores = self.fc_cls(hidden_states)
        layers_bbox_preds = self.fc_reg(
            self.activate(self.reg_ffn(hidden_states))).sigmoid()
        return layers_cls_scores, layers_bbox_preds