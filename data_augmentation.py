import torch


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
    print(latents.shape)
    print("no_latent_img_conditioning_prob", no_latent_img_conditioning_prob.shape, latents.shape)
    latents[:, :-1] = torch.where(no_latent_img_conditioning_prob,
                                  torch.zeros_like(latents[:, :-1]),
                                  latents[:, :-1])
    return latents