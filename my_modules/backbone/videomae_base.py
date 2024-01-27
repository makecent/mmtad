import math

import torch
import torch.nn as nn
from einops import rearrange
from mmdet.registry import MODELS


@MODELS.register_module()
class VideoMAE_Base(nn.Module):
    """
    The VideoMAE model from https://arxiv.org/abs/2203.12602.
    The pre-training image size is (16x224x224), the frame interval is 2 and 4 for SSV2 and Kinetics, respectively.
    The patch size is 2x16x16.
    """

    def __init__(self):
        super(VideoMAE_Base, self).__init__()
        from transformers import VideoMAEModel
        model = VideoMAEModel.from_pretrained("MCG-NJU/videomae-base-finetuned-kinetics")
        pe = get_sinusoid_encoding_table(model.embeddings.config.hidden_size, max_len=5000)
        delattr(model.embeddings, "position_embeddings")
        model.embeddings.register_buffer("position_embeddings", pe, persistent=True)
        model.embeddings.forward = custom_forward.__get__(model.embeddings, model.embeddings.__class__)
        model.embeddings.patch_embeddings.forward = custom_forward2.__get__(model.embeddings.patch_embeddings,
                                                                            model.embeddings.patch_embeddings.__class__)

        self.vision_model = model

    def forward(self, x):
        # The inputs should be (N, M, C, T, H, W), N is the batch size and M = num_crops x num_clips.
        n, m, c, t, h, w = x.shape
        num_crops = 1  # TODO: compatible with dynamic num_crops, e.g. num_crops=3 when ThreeCrop as test augmentation
        num_clips = m // num_crops
        # x: (N, M, C, T, H, W) -> (NxM, T, C, H, W)
        x = rearrange(x, 'n m c t h w -> (n m) t c h w')
        x = self.vision_model(x).last_hidden_state
        # x: (NxM, T'xH'xW', C') -> (NxM, C', T', H', W')  T'=T/2, H'=H/16, W'=W/16
        x = rearrange(x, '(n m) (t h w) c -> (n m) c t h w', n=n, m=m, t=t // 2, h=(h // 16), w=(w // 16))
        if num_clips > 1:
            # x: (NxM, C', T', H', W') -> (N x num_crops, C', num_clips x T', H', W')
            x = rearrange(x, '(n crops clips) c t h w -> (n crops) c (clips t) h w',
                          n=n, crops=num_crops, clips=num_clips)
        return x


def custom_forward(self, pixel_values, bool_masked_pos):
    # create patch embeddings
    embeddings = self.patch_embeddings(pixel_values)

    # add position embeddings
    seq_len = embeddings.shape[1]
    embeddings = embeddings + self.position_embeddings[:, :seq_len, :].type_as(embeddings)

    # only keep visible patches
    # ~bool_masked_pos means visible
    if bool_masked_pos is not None:
        batch_size, _, num_channels = embeddings.shape
        embeddings = embeddings[~bool_masked_pos]
        embeddings = embeddings.reshape(batch_size, -1, num_channels)

    return embeddings


def custom_forward2(self, pixel_values):
    # permute to (batch_size, num_channels, num_frames, height, width)
    pixel_values = pixel_values.permute(0, 2, 1, 3, 4)
    embeddings = self.projection(pixel_values).flatten(2).transpose(1, 2)
    return embeddings


def get_sinusoid_encoding_table(d_model: int, max_len: int = 5000):
    """"We don't use the same name function used in VideoMAE, which adapts from the original Transformer, because
    it's too slow. Instead, we use the one used in torch tutorial:
    https://pytorch.org/tutorials/beginner/transformer_tutorial.html.
    Note that it's just the used one is much more computational friendly while the positional encoding values of the
    two version are exactly the same. Besides, we compute a max_len of encodings to support variable sequence length"""
    position = torch.arange(max_len).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
    pe = torch.zeros(1, max_len, d_model)
    pe[0, :, 0::2] = torch.sin(position * div_term)
    pe[0, :, 1::2] = torch.cos(position * div_term)
    return pe
