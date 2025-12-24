import argparse
import random
import numpy as np
import torch
from diffusers.image_processor import VaeImageProcessor
from diffusers.models.autoencoders.vae import DiagonalGaussianDistribution
from datasets import Dataset
from PIL import Image
from tqdm import tqdm
import os
import random
import cv2
import keyboard
import time

from config_sd import BUFFER_SIZE, CFG_GUIDANCE_SCALE, DEFAULT_NUM_INFERENCE_STEPS
from dataset import preprocess_train
from run_inference import (
    decode_and_postprocess,
    encode_conditioning_frames_wo_batch_dim,
    next_latent,
    prepare_conditioning_frames,
)
from model import load_model

"""Action mapping for the Doom environment:
Built action space of size 20 from buttons 
[
 <Button.ATTACK: 0> 
 <Button.MOVE_FORWARD: 13> 
 <Button.MOVE_LEFT: 11>
 <Button.MOVE_RIGHT: 10> 
 <Button.MOVE_BACKWARD: 12>
 <Button.TURN_RIGHT: 14> 
 <Button.TURN_LEFT: 15>
]

0:  [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
1:  [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]
2:  [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0]
3:  [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0]
4:  [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0]
5:  [0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0]
6:  [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]
7:  [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0]
8:  [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0]
9:  [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0]
10: [0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0]
11: [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

0:  no-op
1:  turn left
2:  turn right
3:  move backward
4:  turn left + move backward
5:  turn right + move backward
6:  move right
7:  move left
8:  move forward
9:  turn left + move forward
10: turn right + move forward
11: attack
"""


def set_seed(seed: int):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

def decode_latents(vae, image_processor, latents):
    with torch.inference_mode():
        image = decode_and_postprocess(
                    vae=vae, 
                    image_processor=image_processor, 
                    latents=latents, 
                    output_type="pt"
                )
    return image

def generate_single_future_frame(
    unet,
    vae,
    action_embedding,
    noise_scheduler,
    image_processor,
    context_latents: torch.Tensor,
    current_actions: torch.Tensor,
    num_inference_steps: int,
    discretized_noise_level: int,
) -> list[Image]:
    device = unet.device

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
                        discretized_noise_level=discretized_noise_level,
    )

    # Update context latents using sliding window
    # Always take exactly BUFFER_SIZE most recent frames
    context_latents = torch.cat(
        [context_latents[(-BUFFER_SIZE + 1) :], target_latents], dim=0
    )

    future_image = decode_latents(vae, image_processor, target_latents)
    return future_image, context_latents

def get_epi_files(basepath: str, 
                  file_format: str = "pt"):
    samples = []
    for dirpath, dirnames, filenames in os.walk(basepath):
        for filename in filenames:
            if filename.split(".")[-1] == file_format:
                if file_format == "pt" and filename != "latent_black.pt": 
                    samples.append(os.path.join(dirpath, filename))
                else:
                    samples.append(os.path.join(dirpath, filename))
    return samples

def load_pt(fpath: str):
    return torch.load(fpath)

def collate_pixels_and_actions(epi_data):
    return {'pixel_values': torch.stack(epi_data['pixel_values']),
            'input_ids': epi_data['input_ids']}

def select_action(turn_left: str, 
                  turn_right: str, 
                  move_back: str, 
                  turn_left_move_back: str, 
                  turn_right_move_back: str,
                  move_right: str, 
                  move_left: str, 
                  move_forward: str, 
                  turn_left_move_forward: str, 
                  turn_right_move_forward: str,
                  attack: str, 
                  device: torch.device):
    if keyboard.is_pressed(turn_left):
        action = torch.tensor([1], dtype=torch.int64).to(device)
    elif keyboard.is_pressed(turn_right):
        action = torch.tensor([2], dtype=torch.int64).to(device)
    elif keyboard.is_pressed(move_back):
        action = torch.tensor([3], dtype=torch.int64).to(device)
    elif keyboard.is_pressed(turn_left_move_back):
        action = torch.tensor([4], dtype=torch.int64).to(device)
    elif keyboard.is_pressed(turn_right_move_back):
        action = torch.tensor([5], dtype=torch.int64).to(device)
    elif keyboard.is_pressed(move_right):
        action = torch.tensor([6], dtype=torch.int64).to(device)
    elif keyboard.is_pressed(move_left):
        action = torch.tensor([7], dtype=torch.int64).to(device)
    elif keyboard.is_pressed(move_forward):
        action = torch.tensor([8], dtype=torch.int64).to(device)
    elif keyboard.is_pressed(turn_left_move_forward):
        action = torch.tensor([9], dtype=torch.int64).to(device)
    elif keyboard.is_pressed(turn_right_move_forward):
        action = torch.tensor([10], dtype=torch.int64).to(device)
    elif keyboard.is_pressed(attack):
        action = torch.tensor([11], dtype=torch.int64).to(device)
    else:
        # No operation
        action = torch.tensor([0], dtype=torch.int64).to(device)
    return action

def convert_from_torch_to_numpy(img):
    img = torch.squeeze(img, 0)
    img = img.cpu().numpy()
    img = np.transpose(img, axes=(1, 2, 0))
    return (np.clip((img)*255.0, 0.0, 255.0)).astype(np.uint8)

def display_init_img(vae, image_processor, img_latent):
    img = decode_latents(vae, image_processor, img_latent)
    cv2.imshow(f'inference', img)
    cv2.waitKey(1000)

def render(img):
    img = convert_from_torch_to_numpy(img)[...,::-1]
    cv2.imshow(f'inference', img)

def create_action_log(action_log_dir, action_log):
    os.makedirs(action_log_dir, exist_ok=True)
    np.savez_compressed(os.path.join(action_log_dir, 'action_log'), actions=np.array(action_log))
    print(f"Action log is created at '{action_log_dir}'.")

def main(basepath: str, unet_model_folder: str, vae_model_folder: str, start_from_pixels: bool, 
         num_inference_steps: int, num_episode_steps: int | None, gif_rec: bool, rec_path_wo_ext: str,
         discretized_noise_level: int, action_log_dir: str, conduct_headless_test: bool,
         action_key_for_headless: int) -> None:
    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )

    # Specify actions
    turn_left = "a"
    turn_right = "g"
    move_back = "x"
    turn_left_move_back = "z"
    turn_right_move_back = "c"
    move_right = "f"
    move_left = "s"
    move_forward = "e" 
    turn_left_move_forward = "w" 
    turn_right_move_forward = "r"
    attack = "d"

    unet, vae, action_embedding, noise_scheduler = load_model(
        unet_model_folder, vae_model_folder, device=device
    )
    vae_scale_factor = 2 ** (len(vae.config.block_out_channels) - 1)
    image_processor = VaeImageProcessor(vae_scale_factor=vae_scale_factor)

    dataset = get_epi_files(basepath, file_format="parquet") if start_from_pixels \
                            else get_epi_files(basepath, file_format="pt")
    ds_length = len(dataset)
    epi_idx = random.sample(range(ds_length), 1)[0]
    episode = dataset[epi_idx]
    epi_data = Dataset.from_parquet(episode) if start_from_pixels else load_pt(episode)
    start_idx = random.randint(0, len(epi_data["actions"]) - BUFFER_SIZE)

    if start_from_pixels:
        print("Intializing context pixels and actions")
        epi_data = epi_data.with_transform(preprocess_train)
        collate_epi_data = collate_pixels_and_actions(epi_data[start_idx:start_idx+BUFFER_SIZE])
        init_img = collate_epi_data["pixel_values"][-1]
        
        # Encode initial context frames
        with torch.inference_mode():
            context_latents = encode_conditioning_frames_wo_batch_dim(
                vae,
                images=collate_epi_data["pixel_values"],
                dtype=torch.float32,
            ) # [BUFFER_SIZE, 4, 32, 40]

        current_actions = collate_epi_data["input_ids"][start_idx:start_idx+BUFFER_SIZE].to(device)
    else:
        print("Intializing context latents and actions")
        parameters = epi_data["parameters"][start_idx:start_idx+BUFFER_SIZE]
        context_latents = DiagonalGaussianDistribution(parameters).sample().to(device)
        init_img = decode_latents(vae, image_processor, context_latents)
        context_latents = prepare_conditioning_frames(
            vae,
            latents=context_latents,
            device=unet.device,
            dtype=context_latents.dtype,
        )
        current_actions = epi_data["actions"][start_idx:start_idx+BUFFER_SIZE].to(device)

    if not conduct_headless_test: display_init_img(vae, image_processor, init_img)
    
    if args.cv2_rec:
        fourcc = cv2.VideoWriter_fourcc(*'MP4V')
        rec_path = rec_path_wo_ext + '.mp4'
        video_writer = cv2.VideoWriter(rec_path, fourcc, 10.0, (320, 256))
        np_img = convert_from_torch_to_numpy(init_img)[...,::-1]
        video_writer.write(np_img)
    elif args.gif_rec:
        np_imgs = []
        np_imgs.append(convert_from_torch_to_numpy(init_img))

    if action_log_dir:
        os.makedirs(action_log_dir, exist_ok=True)
        action_log = []

    i = 0
    while True:
        print(f"step: {i+1}")
        frame_start_time = time.time()

        if not conduct_headless_test and keyboard.is_pressed('q'): break

        if conduct_headless_test:
            action = torch.tensor([action_key_for_headless], dtype=torch.int64).to(device)
        else:
            action = select_action(turn_left, turn_right, move_back, turn_left_move_back, 
                                    turn_right_move_back, move_right, move_left, move_forward, 
                                    turn_left_move_forward, turn_right_move_forward, attack, device)

        current_actions = torch.cat(
            [
                current_actions[(-BUFFER_SIZE + 1) :],
                torch.tensor([action]).to(device),
            ]
        )

        if action_log_dir: action_log.append(action.to("cpu").item())

        future_image, context_latents = generate_single_future_frame(
                                            unet=unet,
                                            vae=vae,
                                            action_embedding=action_embedding,
                                            noise_scheduler=noise_scheduler,
                                            image_processor=image_processor,
                                            context_latents=context_latents,
                                            current_actions=current_actions,
                                            num_inference_steps=num_inference_steps,
                                            discretized_noise_level=discretized_noise_level,
                                        )

        cur_img = future_image
        if not conduct_headless_test: render(cur_img)

        if args.cv2_rec:
            np_img = convert_from_torch_to_numpy(cur_img)[...,::-1]
            video_writer.write(np_img)
        elif args.gif_rec:
            np_imgs.append(convert_from_torch_to_numpy(cur_img))

        wait = 1 / args.max_fps - (time.time() - frame_start_time)
        i += 1
        if num_episode_steps is not None and i == num_episode_steps: break
        if wait > 0: time.sleep(wait)

    if args.gif_rec:
        pil_imgs = []
        for img in np_imgs:
            pil_imgs.append(Image.fromarray(img))
        rec_path = rec_path_wo_ext + '.gif'
        pil_imgs[0].save(rec_path, save_all=True, append_images=pil_imgs[1:], 
                optimize=False, duration=40, loop=0)

    if action_log_dir: create_action_log(action_log_dir, action_log)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run inference with customizable parameters"
    )
    parser.add_argument(
        "--conduct_headless_test",
        action="store_true",
        help=(
            "If you don't have a GPU locally and have it remotely, use this flag to test the script ",
            "remotely."
        ),
    )
    parser.add_argument(
        '--action_key_for_headless', 
        type=int, 
        default=11,
        help=(
            "If `conduct_headless_test` is True, specify an action key to use it repeatly."
        ),
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
        '--max_fps', 
        type=int, 
        default=30,
        help=(
            "Maximum number of FPS. The FPS the generation model produce can not exceed the "
            "one you specify here."
        ),
    )
    parser.add_argument(
        "--discretized_noise_level",
        type=int,
        default=9,
        help=(
            "Discretized noise level used for context images for autoregressive predition."
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
        "--num_episode_steps",
        type=int,
        default=None,
        help=(
            "The number of episode steps to generate."
        ),
    )
    parser.add_argument(
        "--unet_model_folder",
        type=str,
        default="Masao-Taketani/vizdoom-diffusion-dynamic-model",
        help="Path to the folder containing the trained Unet model weights",
    )
    parser.add_argument(
        "--vae_ft_model_folder",
        type=str,
        default="Masao-Taketani/vizdoom-finetuned-decoder",
        help="Path to the folder containing the finetuned VAE model weights. "
             "If None, use the 'arnaudstiegler/game-n-gen-vae-finetuned' is used.",
    )
    parser.add_argument(
        "--action_dim",
        type=int,
        default=12,
        help=(
            "The number of action dim the RL agent has trained with."
        ),
    )
    parser.add_argument('--action_log_dir', type=str, default=None, 
                        help='If you would like to save the action log, you need to specify a directory path \
                              to save the log. Otherwise, it is not recorded.')
    parser.add_argument(
        "--seed", 
        type=int, 
        default=9052924, 
        help="A seed for reproducible inference."
    )
    parser.add_argument('--cv2_rec', action='store_true')
    parser.add_argument('--gif_rec', action='store_true')
    parser.add_argument('--rec_path_wo_ext', type=str, default='recorded_play')

    args = parser.parse_args()
    set_seed(args.seed)
    main(args.dataset_basepath, args.unet_model_folder, args.vae_ft_model_folder, args.start_from_pixels,
         args.num_inference_steps, args.num_episode_steps, args.gif_rec, args.rec_path_wo_ext,
         args.discretized_noise_level, args.action_log_dir, args.conduct_headless_test, 
         args.action_key_for_headless)