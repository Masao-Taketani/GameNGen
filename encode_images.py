import argparse
import numpy as np
import os

from model import get_ft_vae_decoder


def parse_args():
    parser = argparse.ArgumentParser(description="Script to encode images into latents for efficient training.")
    parser.add_argument(
        "--data_dir_path",
        type=str,
        help="Path to a directory containing parquet files which are collected while training the PPO policy.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=744,
        help="Batch size used for each image encoding.",
    )
    parser.add_argument(
        "--chunk_size",
        type=int,
        default=None,
        help="Chunk size used for parallel GPU processing by deviding the whole episode dataset into chunks.",
    )
    parser.add_argument(
        "--gpu_id",
        type=str,
        default=None,
        help="If multiple GPUs are available and you need to specify GPU id for parallel chunk processes, ",
             "use this argumment. If None and GPU(s) is/are avaiable, The first index is used."
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="bf16",
        help="Dtype used for VAE and batch data.",
    )

    args = parser.parse_args()
    return args


def save_data_as_npz(dpath, epi_id, kv):
    path = os.path.join(dpath, f"episode_{epi_id}.npz")
    np.savez(path, kv)

def main():
    args = parse_args()
    if torch.cuda.is_available():
        device_name = f'cuda:{args.gpu_id}' if args.gpu_id else 'cuda'
    else:
        device_name = 'cpu'
    device = torch.device(device_name)

    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    vae = get_ft_vae_decoder()
    vae.to(device, dtype=weight_dtype)

    for epi_id, (imgs, acts) in tqdm(enumerate(loader)):
        imgs = imgs.squeeze(dim=0)
        kv = {}
        params = []
        for i in range(0, len(imgs), args.batch_size):
            bs_imgs = imgs[i:i+args.batch_size].to(device, dtype=weight_dtype)
            with torch.inference_mode():
                parameters = vae.encode(bs_imgs).latent_dist.parameters.cpu().float().data.numpy()
                params.append(params)
        kv["parameters"] = np.concatena