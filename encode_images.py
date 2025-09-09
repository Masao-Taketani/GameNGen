import argparse

from model import get_ft_vae_decoder


def parse_args():
    parser = argparse.ArgumentParser(description="Script to encode images into latents for efficient training.")
    parser.add_argument(
        "--data_dir_path",
        type=str,
        help="Path to a directory containing parquet files which are collected while training the PPO policy.",
    )

    args = parser.parse_args()
    return args

def main():
    args = parse_args()

    for imgs in dataloader:
        vae = get_ft_vae_decoder()
        vae.requires_grad_(False)
        parameters = vae.encode(
            imgs.to(dtype=weight_dtype)
        ).latent_dist.parameters