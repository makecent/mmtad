import math

import torch
import torch.nn as nn
from einops import rearrange
from mmdet.registry import MODELS


def crops_to_batch(forward_methods):
    def wrapper(self, inputs, *args, **kwargs):
        # inputs: (N, M, C, T, H, W) or (N, C, T, H, W)
        if inputs.ndim == 6:
            num_crops = inputs.shape[1]
            inputs = inputs.view(-1, *inputs.shape[2:])
        return forward_methods(self, inputs, *args, **kwargs)

    return wrapper


@MODELS.register_module()
class VideoMAE_Base(nn.Module):

    def __init__(self, input_shape=(224, 224, 16)):
        super(VideoMAE_Base, self).__init__()
        from transformers import VideoMAEModel
        model = VideoMAEModel.from_pretrained("MCG-NJU/videomae-base-finetuned-kinetics")
        pe = get_sinusoid_encoding_table(model.embeddings.config.hidden_size, max_len=5000)
        delattr(model.embeddings, "position_embeddings")
        model.embeddings.register_buffer("position_embeddings", pe, persistent=True)
        model.embeddings.forward = custom_forward.__get__(model.embeddings, model.embeddings.__class__)
        model.embeddings.patch_embeddings.forward = custom_forward2.__get__(model.embeddings.patch_embeddings,
                                                                            model.embeddings.patch_embeddings.__class__)
        # Interpolate the positional embeddings in the embedding layer to the target shape
        # self.resize_pos_embedding(model.embeddings, input_shape)
        # Change the forward function of embedding layer in X-CLIP to customized forward function
        # to accept dynamic input shape that coule be different to the training (but need to be square).

        self.vision_model = model

    @crops_to_batch
    def forward(self, x):
        n, _, t, _, _ = x.shape
        # x: (N, C, T, H, W) -> (NxT, C, H, W)
        x = rearrange(x, 'n c t h w -> n t c h w')
        outs = self.vision_model(x).last_hidden_state
        # outs: (N, T'xL, C') -> (N, C', T', L, 1) mimic the NCTHW T'=T/2, L=H'*W'
        outs = rearrange(outs, 'n (t l) c -> n c t l 1', n=n, t=t // 2)
        return outs

    # @staticmethod
    # def resize_pos_embedding(embed_layer, input_shape):
    #     """
    #     Resize the position embeddings of a Vision Transformer.
    #
    #     Args:
    #     - embedding_layer: The original embedding layer (torch.nn.Embedding).
    #     - new_size: The size of the new feature map (N, where the feature map is N x N).
    #
    #     Returns:
    #     - A new embedding layer with resized positional embeddings.
    #     """
    #     patch_embed = embed_layer.patch_embeddings
    #     if input_shape != patch_embed.image_size:
    #         patch_size = patch_embed.patch_size
    #         tubelet_size = patch_embed.tubelet_size
    #
    #         num_patches = (
    #                 (input_shape[0] // patch_size[0]) * (input_shape[1] // patch_size[1]) *
    #                 (input_shape[2] // tubelet_size)
    #         )
    #         embed_layer.num_patches = num_patches
    #         patch_embed.num_patches = num_patches
    #         patch_embed.image_size = input_shape
    #
    #         embed_layer.position_embeddings = get_sinusoid_encoding_table(num_patches, embed_layer.config.hidden_size)


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
