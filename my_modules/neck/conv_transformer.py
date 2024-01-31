# Adapted from ActionFormer https://github.com/happyharrycn/actionformer_release/tree/main

import math

import torch
from mmdet.registry import MODELS
from torch import nn
from torch.nn import functional as F

from my_modules.layers.actionformer_layers import MaskedConv1d, LayerNorm, TransformerBlock


def get_sinusoid_encoding_table(d_model: int, max_len: int = 5000):
    """"Adapted from https://pytorch.org/tutorials/beginner/transformer_tutorial.html."""
    position = torch.arange(max_len).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
    pe = torch.zeros(1, max_len, d_model)
    pe[0, :, 0::2] = torch.sin(position * div_term)
    pe[0, :, 1::2] = torch.cos(position * div_term)
    return pe


@MODELS.register_module("convTransformer")
class ConvTransformer(nn.Module):
    """
    A neck that combines convolutions with transformer.
    Adapt from ActionFormer(https://github.com/happyharrycn/actionformer_release/blob/main/libs/modeling/backbones.py).
    The author of ActionFormer call this module a backbone, but we call it a neck. Because the ActionFormer works
    on off-the-shelf extracted features, while we perform end-to-end training and the feature extraction model is
    instead called a backbone in our codebase.
    Several modifications:
        1. We replace the original get_sinusoid_encoding function with get_sinusoid_encoding_table, which is much
        more computational friendly.
        3. Set default values (for THUMOS14) to some arguments and add Arguments description,
        making it more friendly to use.
    Args:
        in_channels (int): The input feature dimension.
        embed_dims (int): The embedding dimension (after convolution).
        num_heads (int): The number of head for self-attention in transformers.
        embed_kernel_sizez (int): The conv kernel size of the embedding network.
        max_seq_len (int): The max sequence length.
        arch (tuple): The architecture of the transformer. (#convs, #stem transformers, #branch transformers).
        attn_window_size (list): The size of local window for mha.
        scale_factor (int): The dowsampling rate for the branch.
        with_ln (bool): If to attach layernorm after conv.
        attn_pdrop (float): The dropout rate for the attention map.
        proj_pdrop (float): The dropout rate for the projection / MLP.
        path_pdrop (float): The droput rate for drop path.
        use_abs_pe (bool): If to use absolute position embedding.
        use_rel_pe (bool): If to use relative position embedding.
    """

    def __init__(
            self,
            in_channels=2048,
            # (256, 512, 1536, 1792, 2048, 2304, 2560, [2304, 1536, 256]) are used for different configs.
            embed_dims=512,  # (256, 384, 512, 768, [256, 384, 384]) are used for different configs.
            num_heads=4,  # (4, 12, 16) are used for different configs.
            embed_kernel_sizez=3,
            max_seq_len=2304,  # (192, 1024, 2304) are used for different configs.
            arch=(2, 2, 5),  # ((2, 2, 5), (2, 2, 6), (2, 2, 7)) are used for different configs.
            attn_window_size=[19, 19, 19, 19, 19, 19],
            # ([9, 9, 9, 9, 9, 9], [19, 19, 19, 19, 19, 19], [7, 7, 7, 7, 7, -1])
            scale_factor=2,
            with_ln=True,
            attn_pdrop=0.0,
            proj_pdrop=0.0,  # (0.0, 0.1) are used for different configs.
            path_pdrop=0.1,
            use_abs_pe=False,  # (True, False) are used for different configs.
            use_rel_pe=False,
            post_ln_norm=True,
    ):
        super().__init__()
        assert len(arch) == 3
        assert len(attn_window_size) == (1 + arch[2])
        self.n_in = in_channels
        self.arch = arch
        self.mha_win_size = attn_window_size
        self.max_seq_len = max_seq_len
        self.relu = nn.ReLU(inplace=True)
        self.scale_factor = scale_factor
        self.use_abs_pe = use_abs_pe
        self.use_rel_pe = use_rel_pe
        self.post_ln_norm = post_ln_norm

        self.fpn_strides = [scale_factor ** i for i in range(arch[-1] + 1)]

        # feature projection
        self.n_in = in_channels
        if isinstance(in_channels, (list, tuple)):
            assert isinstance(embed_dims, (list, tuple)) and len(in_channels) == len(embed_dims)
            self.proj = nn.ModuleList([
                MaskedConv1d(c0, c1, 1) for c0, c1 in zip(in_channels, embed_dims)
            ])
            in_channels = embed_dims = sum(embed_dims)
        else:
            self.proj = None

        # embedding network using convs
        self.embd = nn.ModuleList()
        self.embd_norm = nn.ModuleList()
        for idx in range(arch[0]):
            in_channels = embed_dims if idx > 0 else in_channels
            self.embd.append(
                MaskedConv1d(
                    in_channels, embed_dims, embed_kernel_sizez,
                    stride=1, padding=embed_kernel_sizez // 2, bias=(not with_ln)
                )
            )
            if with_ln:
                self.embd_norm.append(LayerNorm(embed_dims))
            else:
                self.embd_norm.append(nn.Identity())

        # position embedding (1, C, T), rescaled by 1/sqrt(n_embd)
        if self.use_abs_pe:
            pos_embd = get_sinusoid_encoding_table(embed_dims, self.max_seq_len, ) / (embed_dims ** 0.5)
            self.register_buffer("pos_embd", pos_embd, persistent=False)

        # stem network using (vanilla) transformer
        self.stem = nn.ModuleList()
        for idx in range(arch[1]):
            self.stem.append(
                TransformerBlock(
                    embed_dims, num_heads,
                    n_ds_strides=(1, 1),
                    attn_pdrop=attn_pdrop,
                    proj_pdrop=proj_pdrop,
                    path_pdrop=path_pdrop,
                    mha_win_size=self.mha_win_size[0],
                    use_rel_pe=self.use_rel_pe
                )
            )

        # main branch using transformer with pooling
        self.branch = nn.ModuleList()
        for idx in range(arch[2]):
            self.branch.append(
                TransformerBlock(
                    embed_dims, num_heads,
                    n_ds_strides=(self.scale_factor, self.scale_factor),
                    attn_pdrop=attn_pdrop,
                    proj_pdrop=proj_pdrop,
                    path_pdrop=path_pdrop,
                    mha_win_size=self.mha_win_size[1 + idx],
                    use_rel_pe=self.use_rel_pe
                )
            )

        max_div_factor = 1
        for l, (s, w) in enumerate(zip(self.fpn_strides, self.mha_win_size)):
            stride = s * (w // 2) * 2 if w > 1 else s
            assert max_seq_len % stride == 0, "max_seq_len must be divisible by fpn stride and window size"
            if max_div_factor < stride:
                max_div_factor = stride
        self.max_div_factor = max_div_factor

        self.fpn_norm = nn.ModuleList(
            [LayerNorm(embed_dims) if self.post_ln_norm else nn.Identity()
             for i in range(len(self.branch) + 1)])

        # init weights
        self.apply(self.__init_weights__)

    def __init_weights__(self, module):
        # set nn.Linear/nn.Conv1d bias term to 0
        if isinstance(module, (nn.Linear, nn.Conv1d)):
            if module.bias is not None:
                torch.nn.init.constant_(module.bias, 0.)

    @property
    def device(self):
        # a hacky way to get the device type
        # will throw an error if parameters are on different devices
        return list(set(p.device for p in self.parameters()))[0]

    @torch.no_grad()
    def preprocessing(self, feats, padding_val=0.0):
        # feats[N, C, 1, T]
        if feats.ndim == 4:
            feats = feats.squeeze(2)

        feat_seq_len = feats.shape[-1]
        assert feat_seq_len % self.max_div_factor == 0, "Seq_len must be divisible by fpn stride and window size"

        if self.training:
            assert feat_seq_len == self.max_seq_len, \
                "During training, the feature sequence length should equal self.max_seq_len"
        else:
            assert len(feats) == 1, "Only support batch_size = 1 during inference"

        # we use a hacky way to determine the padding position, the non zeros channels.
        # we assume that a valid feature position is unlikely to contain all zeros across channels
        masks = (feats != 0).any(dim=1, keepdim=True).detach()

        return feats, masks

    def forward(self, feats):
        # batch the video list into feats (B, C, T) and masks (B, 1, T)
        x, mask = self.preprocessing(feats)
        # x: batch size, feature channel, sequence length,
        # mask: batch size, 1, sequence length (bool)
        B, C, T = x.size()

        # feature projection
        if isinstance(self.n_in, (list, tuple)):
            x = torch.cat(
                [proj(s, mask)[0] \
                 for proj, s in zip(self.proj, x.split(self.n_in, dim=1))
                 ], dim=1
            )

        # embedding network
        for idx in range(len(self.embd)):
            x, mask = self.embd[idx](x, mask)
            x = self.relu(self.embd_norm[idx](x))

        # training: using fixed length position embeddings
        if self.use_abs_pe and self.training:
            assert T <= self.max_seq_len, "Reached max length."
            pe = self.pos_embd
            # add pe to x
            x = x + pe[:, :, :T] * mask.to(x.dtype)

        # inference: re-interpolate position embeddings for over-length sequences
        if self.use_abs_pe and (not self.training):
            if T >= self.max_seq_len:
                pe = F.interpolate(
                    self.pos_embd, T, mode='linear', align_corners=False)
            else:
                pe = self.pos_embd
            # add pe to x
            x = x + pe[:, :, :T] * mask.to(x.dtype)

        # stem transformer
        for idx in range(len(self.stem)):
            x, mask = self.stem[idx](x, mask)

        # prep for outputs
        out_feats = (self.fpn_norm[0](x),)
        out_masks = (mask,)

        # main branch with downsampling
        for branch, norm in zip(self.branch, self.fpn_norm[1:]):
            x, mask = branch(x, mask)
            out_feats += (norm(x),)
            out_masks += (mask,)

        return out_feats, out_masks
