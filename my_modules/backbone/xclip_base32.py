import torch
import torch.nn as nn
from einops import rearrange
from mmdet.registry import MODELS
from torch.nn import functional as F


def crops_to_batch(forward_methods):
    def wrapper(self, inputs, *args, **kwargs):
        # inputs: (N, M, C, T, H, W) or (N, C, T, H, W)
        if inputs.ndim == 6:
            num_crops = inputs.shape[1]
            inputs = inputs.view(-1, *inputs.shape[2:])
        return forward_methods(self, inputs, *args, **kwargs)

    return wrapper


@MODELS.register_module()
class XCLIP_Base32(nn.Module):

    def __init__(self, input_shape=(224, 224)):
        super(XCLIP_Base32, self).__init__()
        from transformers import XCLIPVisionModel
        model = XCLIPVisionModel.from_pretrained("microsoft/xclip-base-patch32").vision_model
        # Interpolate the positional embeddings in the embedding layer to the target shape
        self.resize_pos_embedding(model.embeddings, input_shape)
        # Change the forward function of embedding layer in X-CLIP to customized forward function
        # which now accept input shape different to the training (but need to be square).
        model.embeddings.forward = custom_forward.__get__(model.embeddings, model.embeddings.__class__)
        # Remove the post_layernorm as it is not used.
        model.post_layernorm = nn.Identity()
        self.vision_model = model

    @crops_to_batch
    def forward(self, x):
        n = x.shape[0]
        # x: (N, C, T, H, W) -> (NxT, C, H, W)
        x = rearrange(x, 'n c t h w -> (n t) c h w')
        outs = self.vision_model(x).last_hidden_state
        # outs: (NxT, L , C') -> (N, C', T, L, 1) mimic the (N, C, T, H, W)
        outs = rearrange(outs, '(n t) l c -> n c t l 1', n=n)
        return outs

    @staticmethod
    def resize_pos_embedding(embed_layer, input_shape):
        """
        Resize the position embeddings of a Vision Transformer.

        Args:
        - embedding_layer: The original embedding layer (torch.nn.Embedding).
        - new_size: The size of the new feature map (N, where the feature map is N x N).

        Returns:
        - A new embedding layer with resized positional embeddings.
        """
        if input_shape != embed_layer.image_size:
            patch_size = embed_layer.patch_size
            embed_dim = embed_layer.embed_dim
            old_num_patches = embed_layer.num_patches
            old_pos_embed = embed_layer.position_embedding

            new_h, new_w = input_shape[0] // patch_size, input_shape[1] // patch_size
            num_patches = new_h * new_w
            num_positions = num_patches + 1  # +1 for the class token

            # Reshape the original embeddings to a 2D grid
            old_size = int((old_num_patches) ** 0.5)  # 7
            old_embeddings = old_pos_embed.weight[1:].view(1, old_size, old_size, embed_dim).permute(0, 3, 1, 2)

            # Interpolate embeddings
            new_embeddings = F.interpolate(old_embeddings, size=(new_h, new_w), mode='bilinear', align_corners=False)

            # Flatten the embeddings back to the original format and add the class token
            new_embeddings = new_embeddings.permute(0, 2, 3, 1).view(num_patches, embed_dim)
            new_embeddings = torch.cat([old_pos_embed.weight[0:1], new_embeddings], dim=0)

            # Create a new embedding layer
            new_embedding_layer = torch.nn.Embedding(num_positions, embed_dim)
            new_embedding_layer.weight = torch.nn.Parameter(new_embeddings)

            embed_layer.position_embedding = new_embedding_layer
            embed_layer.image_size = input_shape
            embed_layer.num_patches = num_patches
            embed_layer.num_positions = num_positions
            embed_layer.position_ids = torch.arange(num_positions).expand((1, -1))


def custom_forward(self, pixel_values: torch.FloatTensor) -> torch.Tensor:
    batch_size = pixel_values.shape[0]
    target_dtype = self.patch_embedding.weight.dtype
    patch_embeds = self.patch_embedding(pixel_values.to(dtype=target_dtype))  # shape = [*, width, grid, grid]
    patch_embeds = patch_embeds.flatten(2).transpose(1, 2)

    class_embeds = self.class_embedding.expand(batch_size, 1, -1)
    embeddings = torch.cat([class_embeds, patch_embeds], dim=1)
    if embeddings.shape[1] == self.position_embedding.num_embeddings:
        embeddings = embeddings + self.position_embedding(self.position_ids)
    else:
        # Interpolate the positional embeddings to support different sizes
        num_positions = embeddings.shape[1]
        num_patches = num_positions - 1  # -1 for the class token
        embed_dim = self.embed_dim

        # Reshape the original embeddings to a 2D grid
        old_size = int(self.num_patches ** 0.5)  # 7
        old_embeddings = self.position_embedding.weight[1:].view(1, old_size, old_size, embed_dim).permute(0, 3, 1, 2)

        # Interpolate embeddings
        new_size = int(num_patches ** 0.5)  # 3
        new_embeddings = F.interpolate(old_embeddings, size=(new_size, new_size), mode='bilinear', align_corners=False)

        # Flatten the embeddings back to the original format and add the class token
        new_embeddings = new_embeddings.permute(0, 2, 3, 1).view(num_patches, embed_dim)
        new_embeddings = torch.cat([self.position_embedding.weight[0:1], new_embeddings], dim=0)

        embeddings = embeddings + new_embeddings
    return embeddings
