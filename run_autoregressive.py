import argparse
import random
import numpy as np
import torch
from diffusers.image_processor import VaeImageProcessor
from diffusers.models.autoencoders.vae import DiagonalGaussianDistribution
from PIL import Image
from tqdm import tqdm
import os
import random

from config_sd import BUFFER_SIZE, CFG_GUIDANCE_SCALE, DEFAULT_NUM_INFERENCE_STEPS
from dataset import EpisodeDatasetLatent, EpisodeDatasetMod
from run_inference import (
    decode_and_postprocess,
    encode_conditioning_frames_wo_batch_dim,
    next_latent,
    prepare_conditioning_frames,
)
from model import load_model

"""Action mapping for the Doom environment:
Built action space of size 18 from buttons [
    <Button.ATTACK: 0> 
    <Button.MOVE_FORWARD: 13> 
    <Button.MOVE_LEFT: 11>
    <Button.MOVE_RIGHT: 10>
    <Button.TURN_RIGHT: 14>
    <Button.TURN_LEFT: 15>
    ]
"""


def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def generate_rollout(
    unet,
    vae,
    action_embedding,
    noise_scheduler,
    image_processor,
    actions: list[int],
    initial_frame_context: torch.Tensor,
    initial_action_context: torch.Tensor,
    num_inference_steps: int
) -> list[Image]:
    device = unet.device
    all_latents = []
    current_actions = initial_action_context
    context_latents = initial_frame_context

    for i in tqdm(range(len(actions))):
        # Generate next frame latents
        target_latents = next_latent(
            unet=unet,
            noise_scheduler=noise_scheduler,
            action_embedding=action_embedding,
            context_latents=context_latents.unsqueeze(0),
            device=device,
            actions=current_actions.unsqueeze(0),
            skip_action_conditioning=False,
            num_inference_steps=num_inference_steps,
            do_classifier_free_guidance=True,
            guidance_scale=CFG_GUIDANCE_SCALE,
        )
        all_latents.append(target_latents)
        current_actions = torch.cat(
            [
                current_actions[(-BUFFER_SIZE + 1) :],
                torch.tensor([actions[i]]).to(device),
            ]
        )

        # Update context latents using sliding window
        # Always take exactly BUFFER_SIZE most recent frames
        context_latents = torch.cat(
            [context_latents[(-BUFFER_SIZE + 1) :], target_latents], dim=0
        )

    # Decode all latents to images
    all_images = []
    for latent in all_latents:  # Skip the initial context frames
        with torch.inference_mode():
            all_images.append(
                decode_and_postprocess(
                    vae=vae, image_processor=image_processor, latents=latent
                )
            )
    return all_images


def get_epi_files(basepath, file_format="pt"):
    samples = []
    for dirpath, dirnames, filenames in os.walk(basepath):
        for filename in filenames:
            if filename.split(".")[-1] == file_format:
                if file_format == "pt" and filename != "latent_black.pt": 
                    samples.append(os.path.join(dirpath, filename))
                else:
                    samples.append(os.path.join(dirpath, filename))
    return samples


def load_pt(fpath):
    return torch.load(fpath)


def collate_pixels_and_actions(epi_data):
    return {'pixel_values': torch.stack(epi_data['pixel_values']),
            'input_ids': epi_data['input_ids']}


def main(basepath: str, num_episodes: int, episode_length: int, unet_model_folder: str, 
         vae_model_folder: str, start_from_pixels: bool, num_inference_steps: int, outdir: str) -> None:
    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )

    dataset = get_epi_files(basepath, file_format="pt") if start_from_latents else get_epi_files(basepath, file_format="parquet")
    ds_length = len(dataset)
    epi_indices = random.sample(range(ds_length), num_episodes)

    unet, vae, action_embedding, noise_scheduler = load_model(
        unet_model_folder, vae_model_folder, device=device
    )

    vae_scale_factor = 2 ** (len(vae.config.block_out_channels) - 1)
    image_processor = VaeImageProcessor(vae_scale_factor=vae_scale_factor)

    for epi_idx in epi_indices:
        episode = dataset[epi_idx]
        epi_data = Dataset.from_parquet(episode) if start_from_pixels else load_pt(episode)
        start_idx = random.randint(0, len(epi_data["input_ids"]) - episode_length - BUFFER_SIZE)

        if start_from_pixels:
            # Haven't tested this part yet
            collate_epi_data = collate_pixels_and_actions(epi_data[start_idx:start_idx+BUFFER_SIZE])

            # Encode initial context frames
            with torch.inference_mode():
                context_latents = encode_conditioning_frames_wo_batch_dim(
                    vae,
                    images=collate_epi_data["pixel_values"],
                    dtype=torch.float32,
                )

            # Store all generated latents - split context frames into individual tensors
            initial_frame_context = context_latents  # [BUFFER_SIZE, 4, 32, 40]
            initial_action_context = collate_epi_data["input_ids"][:BUFFER_SIZE].to(device)
        else:
            parameters = epi_data["parameters"][start_idx:start_idx+BUFFER_SIZE]
            initial_frame_context = DiagonalGaussianDistribution(parameters).sample().to(device)
            initial_frame_context = prepare_conditioning_frames(
                vae,
                latents=initial_frame_context,
                device=unet.device,
                dtype=context_latents.dtype,
            )
            initial_action_context = epi_data["actions"][start_idx:start_idx+BUFFER_SIZE].to(device)

        future_actions = epi_data["actions"][start_idx+BUFFER_SIZE:start_idx+BUFFER_SIZE+episode_length]

        all_images = generate_rollout(
            unet=unet,
            vae=vae,
            action_embedding=action_embedding,
            noise_scheduler=noise_scheduler,
            image_processor=image_processor,
            actions=future_actions,
            initial_frame_context=initial_frame_context,
            initial_action_context=initial_action_context,
            num_inference_steps=num_inference_steps,
        )

        os.makedirs(outdir, exist_ok=True)
        all_images[0].save(
            os.path.join(outdir, f"rollout_{epi_idx}.gif"),
            save_all=True,
            append_images=all_images[1:],
            duration=100,  # 100ms per frame
            loop=1,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run inference with customizable parameters"
    )
    parser.add_argument(
        "--start_from_pixels",
        action="store_true",
        help=(
            "The inference starts with pixels if True. Otherwise, it starts with "
            "latents to save calculation resources.",
        ),
    )
    parser.add_argument(
        "--dataset_basepath",
        type=str,
        help=(
            "The latent dataset base path pointing a folder containing pt files."
        ),
    )
    parser.add_argument(
        "--gif_outdir",
        type=str,
        default="rollouts",
        help=(
            "The GIF output dir."
        ),
    )
    
    parser.add_argument(
        "--num_inference_steps",
        type=int,
        default=4,
        help=(
            "The number of inference steps to generate for each frame."
        ),
    )
    parser.add_argument(
        "--num_episodes",
        type=int,
        default=20,
        help=(
            "The number of episodes to generate."
        ),
    )
    parser.add_argument(
        "--episode_length",
        type=int,
        default=300,
        help=(
            "The number of frames to generate per episode."
        ),
    )
    parser.add_argument(
        "--unet_model_folder",
        type=str,
        help="Path to the folder containing the trained Unet model weights",
    )
    parser.add_argument(
        "--vae_ft_model_folder",
        type=str,
        default=None,
        help="Path to the folder containing the finetuned VAE model weights. "
             "If None, use the 'arnaudstiegler/game-n-gen-vae-finetuned' is used.",
    )
    parser.add_argument(
        "--action_dim",
        type=int,
        default=18,
        help=(
            "The number of action dim the RL agent has trained with."
        ),
    )
    parser.add_argument(
        "--seed", 
        type=int, 
        default=9052924, 
        help="A seed for reproducible inference."
    )
    args = parser.parse_args()
    set_seed(args.seed)
    main(args.dataset_basepath, args.num_episodes, args.episode_length, 
         args.unet_model_folder, args.vae_ft_model_folder, args.start_from_pixels,
         args.num_inference_steps, args.gif_outdir)