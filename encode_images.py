import argparse
import numpy as np
import os
from tqdm import tqdm
import torch

from model import get_vae
from dataset import get_dataloader_prep
from config_sd import HEIGHT, WIDTH, H_PAD, W_PAD


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
        "--dtype",
        type=str,
        default="bf16",
        help="Dtype used for VAE and batch data.",
    )
    # Ignore the following arguments if you are not using multiple GPUs to process.
    parser.add_argument(
        "--num_chunks",
        type=int,
        default=None,
        help="The number of data for each chunk used for parallel GPU processing by deviding the whole episode dataset into chunks.",
    )
    parser.add_argument(
        "--chunk_id",
        type=int,
        default=None,
        help=(
            "Chunk ID used for parallel GPU processing by deviding the whole episode dataset into chunks, "
            "and process each chunk independently specifying the ID.",
        ),
    )
    parser.add_argument(
        "--gpu_id",
        type=str,
        default=None,
        help=(
            "If multiple GPUs are available and you need to specify GPU id for parallel chunk processes, "
            "use this argumment. If None and GPU(s) is/are avaiable, The first index is used."
        ),
    )
    
    parser.add_argument(
        "--model_folder_or_id",
        type=str,
        default=None,
        help=(
            "Specify VAE model folder or HF id. If None, 'PRETRAINED_MODEL_NAME_OR_PATH' is used."
        ),
    )

    args = parser.parse_args()
    return args


def save_data_as_npz(dpath, epi_id, parameters, actions):
    path = os.path.join(dpath, f"latent_episode_{epi_id}.npz")
    np.savez(path, parameters=parameters, actions=actions)

def save_data_as_pt(dpath, epi_id, parameters, actions):
    path = os.path.join(dpath, f"latent_episode_{epi_id}.pt")
    data = {
        "parameters": parameters,
        "actions": actions
    }
    torch.save(data, path)

# This is used when prediction index is less than buffer size. For more details, please check the following part of code.
# https://github.com/Masao-Taketani/GameNGen/blob/834d3d081ef782a92bb6928fe66cb88b8aa9c7d7/dataset.py#L127
def save_latent_black(dpath, vae, device, weight_dtype):
    black_img = torch.zeros(1, 3, HEIGHT + H_PAD, WIDTH + W_PAD).to(device, dtype=weight_dtype)
    # Leave the batch dim of parameters as 1, so that it is used as time dimension during training data preparation.
    parameters = vae.encode(black_img).latent_dist.parameters.cpu().float().data
    path = os.path.join(dpath, f"latent_black.pt")
    torch.save(parameters, path)

def main():
    args = parse_args()
    if torch.cuda.is_available():
        device_name = f'cuda:{args.gpu_id}' if args.gpu_id else 'cuda'
    else:
        device_name = 'cpu'
    device = torch.device(device_name)

    weight_dtype = torch.float32
    if args.dtype == "fp16":
        weight_dtype = torch.float16
    elif args.dtype == "bf16":
        weight_dtype = torch.bfloat16

    dataloader = get_dataloader_prep(
        basepath=args.dataset_basepath,
        num_chunk=args.num_chunks,
        chunk_id=args.chunk_id,
        num_workers=args.dataloader_num_workers,
    )
    start_epi_id = dataloader.dataset.epi_id
    vae = get_vae(args.model_folder_or_id)
    vae.to(device, dtype=weight_dtype)

    for epi_id, (imgs, acts) in tqdm(enumerate(dataloader), total=len(dataloader)):
        imgs = imgs.squeeze(dim=0)
        params = []
        for i in range(0, len(imgs), args.batch_size):
            bs_imgs = imgs[i:i+args.batch_size].to(device, dtype=weight_dtype)
            with torch.inference_mode():
                #parameters = vae.encode(bs_imgs).latent_dist.parameters.cpu().float().data.numpy()
                parameters = vae.encode(bs_imgs).latent_dist.parameters.cpu().float().data
                params.append(parameters)
        #save_data_as_npz(args.save_dir_path, start_epi_id+epi_id, np.concatenate(params, axis=0), acts.squeeze(dim=0).numpy())
        save_data_as_pt(args.save_dir_path, start_epi_id+epi_id, torch.cat(params, dim=0), acts.squeeze(dim=0))
    
    save_latent_black(args.save_dir_path, vae, device, weight_dtype)


if __name__ == "__main__":
    main()