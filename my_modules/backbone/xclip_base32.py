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

    def __init__(self, training_input_shape=(112, 112), test_input_shape=(128, 128)):
        super(XCLIP_Base32, self).__init__()
        from transformers import XCLIPVisionModel
        self.model = XCLIPVisionModel.from_pretrained("microsoft/xclip-base-patch32")
        if self.training:
            self.resize_pos_embedding(training_input_shape)
        else:
            self.resize_pos_embedding(test_input_shape)

    @crops_to_batch
    def forward(self, x):
        n = x.shape[0]
        # x: (N, C, T, H, W) -> (NxT, C, H, W)
        x = rearrange(x, 'n c t h w -> (n t) c h w')
        outs = self.model(x).last_hidden_state
        # outs: (NxT, L , C') -> (N, C', T, L, 1) mimic the (N, C, T, H, W)
        outs = rearrange(outs, '(n t) l c -> n c t l 1', n=n)
        return outs

    def resize_pos_embedding(self, input_shape):
        """
        Resize the position embeddings of a Vision Transformer.

        Args:
        - embedding_layer: The original embedding layer (torch.nn.Embedding).
        - new_size: The size of the new feature map (N, where the feature map is N x N).

        Returns:
        - A new embedding layer with resized positional embeddings.
        """
        embed_layer = self.model.vision_model.embeddings
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
