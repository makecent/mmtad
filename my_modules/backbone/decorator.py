from einops import rearrange
import torch

torch.fx.wrap('rearrange')


def merge_crops_and_clips(cls):
    # Save the original forward method
    original_forward = cls.forward

    # Define a new forward method incorporating the pre-action
    def new_forward(self, x, *args, **kwargs):
        if x.dim() == 6:
            n, m, c, t, h, w = x.shape
            num_crops = 1  # TODO: compatible with dynamic num_crops, e.g. num_crops=3 when ThreeCrop as test augmentation
            num_clips = m // num_crops
            # x: (N, M, C, T, H, W) -> (NxM, C, T, H, W)
            x = rearrange(x, 'n m c t h w -> (n m) c t h w')
        return original_forward(self, x, *args, **kwargs)

    # Replace the original forward method with the new one
    cls.forward = new_forward

    # Return the modified class
    return cls
