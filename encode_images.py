import argparse
import numpy as np
import os

from model import get_ft_vae_decoder
from dataset import get_dataloader_prep


def parse_args():
    parser = argparse.ArgumentParser(description="Script to encode images into latents for efficient training.")
    parser.add_argument(
        "--dataset_basepath",
        type=str,
        help="Path to a directory containing parquet files which are collected while training the PPO policy.",
    )
    parser.add_argument(
        "--save_dir_path",
        type=str,
        help="Path to a directory to save latent images.",
    )
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=16,
        help=(
            "Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process."
        ),
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=744,
        help="Batch size used for image encoding within each episode data.",
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


def save_data_as_npz(dpath, epi_id, parameters, actions):
    path = os.path.join(dpath, f"latent_episode_{epi_id}.npz")
    np.savez(path, parameters=parameters, actions=actions)

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

    dataloader = get_dataloader_prep(
        basepath=args.dataset_basepath,
        num_workers=args.dataloader_num_workers,
    )

    vae = get_ft_vae_decoder()
    vae.to(device, dtype=weight_dtype)

    for epi_id, (imgs, acts) in tqdm(enumerate(dataloader)):
        imgs = imgs.squeeze(dim=0)
        params = []
        for i in range(0, len(imgs), args.batch_size):
            bs_imgs = imgs[i:i+args.batch_size].to(device, dtype=weight_dtype)
            with torch.inference_mode():
                parameters = vae.encode(bs_imgs).latent_dist.parameters.cpu().float().data.numpy()
                params.append(params)
        save_data_as_npz(args.save_dir_path, epi_id, np.concatenate(params, axis=0), acts.squeeze(dim=0).numpy())