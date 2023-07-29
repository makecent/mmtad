import torch
from torch import nn
from mmdet.models.layers.transformer.utils import inverse_sigmoid

class Pseudo4DRegLinear(nn.Linear):
    """
    This is a pseudo regression linear layer which is used to replace the regression linear layer in
    DETR-like model to support 1D cases (Temporal Action Detection).

    It still works like a Linear(in_dim, 4) to regress the localization information (cxcywh or x1y1x2y2).
    However, it uses inside sa Linear(in_dim, 2) to regress the (cx, w)/(x1, x2) and the y-axis information
    is constant. That why it's called "pseudo4D".

    Such layer can make you use the original bbox-head that is designed for 2D bboxes regressions on
    1D interval regressions without compatibility problem, meanwhile keep the logic right.

    Note that our computation is based on that the default y1, y2 are 0.1 and 0.9, respectively.
    "delta" indicate if the regression target is delta-bbox or bbox. For example, the original DETR bbox-head
    output bbox while the DeformableDETR output delta-bbox which will be added onto a reference-bbox to form
    the final bbox.
    """

    def __init__(self, in_dim, format='cxcywh', delta=True):
        super().__init__(in_dim, 2)
        assert format in ['cxcywh', 'xyxy']
        self.format = format
        self.delta = delta

    def forward(self, x: torch.Tensor):
        x = super().forward(x)
        if self.format == 'cxcywh':
            if self.delta:  # DeformableDETR
                cy = torch.full_like(x[..., :1], 0.)
                h = torch.full_like(x[..., :1], 0.)
            else:   # DETR
                cy = torch.full_like(x[..., :1], 0.)    # equal to 0.5 after sigmoid
                h = torch.full_like(x[..., :1], 1.3863)    # close to 0.8 after sigmoid
            x = torch.cat((x[..., :1], cy, x[..., 1:], h), dim=-1)
        else:
            if self.delta:  # DeformableDETR
                y1 = torch.full_like(x[..., :1], 0.)
                y2 = torch.full_like(x[..., :1], 0.)
            else:   # DETR
                y1 = torch.full_like(x[..., :1], -2.1972)   # close to 0.1 after sigmoid
                y2 = torch.full_like(x[..., :1], 2.1972)    # close to 0.9 after sigmoid
            x = torch.cat((x[..., :1], y1, x[..., 1:], y2), dim=-1)
        return x


class Pseudo2DLinear(nn.Linear):
    """
    This is a pseudo linear layer which is used to replace the linear layer in DETR-like that
    regress the reference-points or sampling-offsets to support 1D cases (Temporal Action Detection).

    It still works like a Linear(in_dim, 2) to regress the localization information (x, y or delta-x, delta-y).
    However, it uses inside sa Linear(in_dim, 1) to regress the (x)/(delta-x) and the y-axis information
    is constant. That why it's called "pseudo2D".

    Such layer can make you replace the reference-point-fc and sampling-offset-fc in DETR-like model
    to 1D without compatibility problem.
    """

    def forward(self, x: torch.Tensor):
        x = super().forward(x)

        # Interleave the output values with zeros (the offsets on y-axis)
        reshaped_output = x.view(-1, 1)
        y = torch.zeros_like(reshaped_output)
        interleaved_output = torch.cat((reshaped_output, y), dim=1)
        # when applied for sampling-offset, since no sigmoid will be applied, the sampling offset on y-axis
        # will always be zeros, which is expected.
        # when applied for initial reference-point, since sigmoid will be applied, the reference-point on y-axis
        # will always be 0.5, which is also expected.
        x = interleaved_output.view(*x.shape[:-1], x.shape[-1]*2)
        return x
