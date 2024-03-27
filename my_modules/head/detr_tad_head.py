from mmdet.models.dense_heads import DETRHead
from mmdet.registry import MODELS

from my_modules.layers.pseudo_layers import Pseudo4DRegLinear


@MODELS.register_module()
class DETR_TADHead(DETRHead):
    """
    TadTR head.
    We modify the regression branches to output 2 (x1, x2) rather than 4 (x1, y1, x2, y2).
    """

    def _init_layers(self) -> None:
        """Initialize layers of the transformer head."""
        super()._init_layers()
        self.fc_reg = Pseudo4DRegLinear(self.embed_dims, delta=False)
