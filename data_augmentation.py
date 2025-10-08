import os
import torch

from config_sd import BUFFER_SIZE
from diffusers.models.autoencoders.vae import DiagonalGaussianDistribution


def no_img_conditioning_augmentation(
    images: torch.Tensor,
    prob: float = 0.0
) -> torch.Tensor:
    """
    Zeroes out the conditioning frames with probability `prob`.
    This is necessary to train the model on no frame conditioning,
    allowing for unconditional generation with CFG.
    """
    turn_off_conditioning_prob = torch.rand(images.shape[0],
                                            1,
                                            1,
                                            1,
                                            device=images.device)
    no_img_conditioning_prob = turn_off_conditioning_prob < prob
    no_img_conditioning_prob = no_img_conditioning_prob.unsqueeze(1).expand(
        -1, images.shape[1] - 1, -1, -1, -1)
    images[:, :-1] = torch.where(no_img_conditioning_prob,
                                 torch.zeros_like(images[:, :-1]),
                                 images[:, :-1])
    return images


def no_latent_img_conditioning_augmentation(
    latents: torch.Tensor,
    prob: float = 0.0
) -> torch.Tensor:
    """
    Zeroes out the conditioning latent frames with probability `prob`.
    This is necessary to train the model on no frame conditioning,
    allowing for unconditional generation with CFG.
    """
    turn_off_conditioning_prob = torch.rand(latents.shape[0],
                                            1,
                                            1,
                                            1,
                                            device=latents.device)
    no_latent_img_conditioning_prob = turn_off_conditioning_prob < prob
    no_latent_img_conditioning_prob = no_latent_img_conditioning_prob.unsqueeze(1).expand(
        -1, latents.shape[1] - 1, -1, -1, -1)
    latents[:, :-1] = torch.where(no_latent_img_conditioning_prob,
                                  torch.zeros_like(latents[:, :-1]),
                                  latents[:, :-1])
    return latents


def no_latent_img_conditioning_augmentation_with_black_latents(
    latents: torch.Tensor,
    dirpath: str,
    prob: float = 0.0,
) -> torch.Tensor:
    """
    Zeroes out the conditioning latent frames with probability `prob`.
    This is necessary to train the model on no frame conditioning,
    allowing for unconditional generation with CFG.
    """
    bs = latents.shape[0]
    turn_off_conditioning_prob = torch.rand(bs,
                                            1,
                                            1,
                                            1,
                                            device=latents.device)
    no_latent_img_conditioning_prob = turn_off_conditioning_prob < prob
    no_latent_img_conditioning_prob = no_latent_img_conditioning_prob.unsqueeze(1).expand(
        -1, latents.shape[1] - 1, -1, -1, -1)
    latent_black = torch.load(os.path.join(dirpath, "latent_black.pt"))
    latent_black_sample = DiagonalGaussianDistribution(latent_black.repeat(bs * BUFFER_SIZE, 1, 1, 1)).sample()
    latents[:, :-1] = torch.where(no_latent_img_conditioning_prob,
                                  latent_black_sample.reshape(bs, BUFFER_SIZE, *latents.shape[2:]).to(latents.device),
                                  latents[:, :-1])
    return latents