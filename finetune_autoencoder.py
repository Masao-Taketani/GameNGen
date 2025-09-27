import os
from datetime import datetime
import argparse
import math
import time

import torch
import torch.nn.functional as F
from datasets import load_dataset, preprocess_train
from diffusers import AutoencoderKL
from torch import optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import get_cosine_schedule_with_warmup
from config_sd import PRETRAINED_MODEL_NAME_OR_PATH

import wandb


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Fine-tune VAE model")
    parser.add_argument(
        "--report_to",
        type=str,
        default="tensorboard",
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`'
            ' (default), `"wandb"` and `"comet_ml"`. Use `"all"` to report to all integrations.'
        ),
    )
    parser.add_argument(
        "--dataset_basepath",
        type=str,
        help=(
            "The parquet dataset base path pointing a folder containing files that 🤗 Datasets can understand to train on."
        ),
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="sd-model-finetuned",
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=2,
        help="Batch size (per device) for dataloader.",
    )
    parser.add_argument(
        "--num_chunks",
        type=int,
        default=70,
        help="Specify the number of chunks which are rotated for dataloader to train.",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=96,
        help="Number of workers used for dataloader.",
    )
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=700000,
        help="Total number of training steps to perform.",
        required=True
    )
    parser.add_argument(
        "--validation_steps",
        type=int,
        default=50,
        help=(
            "Run fine-tuning validation every 50 steps. The validation process consists of running the model on a batch of images"
        ),
    )
    parser.add_argument(
        "--chunk_id_interval",
        type=int,
        default=10000,
        help=(
            "Change chunk ID for the specified interval."
        ),
    )
    parser.add_argument(
        "--use_adamw",
        action="store_true",
        help=(
            "Whether or not to use AdamW optimizer. If 'use_adamw' is not True, "
            "the default optimizer Adafactor, which is used in the paper, is used."
        ),
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=2e-5,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="constant",
        help=(
            'The scheduler type to use. Choose between ["cosine", "constant", "constant_with_warmup"]'
        ),
    )
    parser.add_argument(
        "--weight_decay", type=float, default=0.0, help="Weight decay to use."
    )
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=4,
        help=(
            "Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process."
        ),
    )
    parser.add_argument(
        "--gradient_clipping", default=1.0, type=float, help="Gradient clipping."
    )
    parser.add_argument(
        "--push_to_hub",
        action="store_true",
        help="Whether or not to push your model to the Hugging Face model hub after saving it.",
    )
    parser.add_argument(
        "--repo_id",
        type=str,
        default=None,
        help="Hugging Face repo ID used when push_to_hub is True.",
    )
    return parser.parse_args()


def make_decoder_trainable(model: AutoencoderKL):
    for param in model.encoder.parameters():
        param.requires_grad_(False)
    for param in model.decoder.parameters():
        param.requires_grad_(True)


def eval_model(model: AutoencoderKL, test_loader: DataLoader) -> float:
    model.eval()
    with torch.no_grad():
        test_loss = 0
        progress_bar = tqdm(test_loader, desc=f"Evaluating")

        for batch in progress_bar:
            data = batch["pixel_values"].to(model.device)
            """
            The following part is not sampled as for the encoder output according to the link below.
            https://github.com/huggingface/diffusers/blob/c586aadef6bb66d355fa40a2b95a0bea8a6fe79c/src/diffusers/models/autoencoders/autoencoder_kl.py#L438
            Don't know why. May change it later.
            """
            reconstruction = model(data).sample
            loss = F.mse_loss(reconstruction, data, reduction="mean")
            test_loss += loss.item()

            recon = model.decode(model.encode(data).latent_dist.sample()).sample
            wandb.log(
                {
                    "original": [wandb.Image(img) for img in data],
                    "reconstructed": [wandb.Image(img) for img in recon],
                }
            )
        return test_loss / len(test_loader)


def eval_model_mod(model: AutoencoderKL, data: torch.tensor) -> float:
    model.eval()
    with torch.no_grad():
        """
        The following part is not sampled as for the encoder output according to the link below.
        https://github.com/huggingface/diffusers/blob/c586aadef6bb66d355fa40a2b95a0bea8a6fe79c/src/diffusers/models/autoencoders/autoencoder_kl.py#L438
        Don't know why. May change it later.
        """
        reconstruction = model(data).sample
        loss = F.mse_loss(reconstruction, data, reduction="mean")

        recon = model.decode(model.encode(data).latent_dist.sample()).sample
        wandb.log(
            {
                "original": [wandb.Image(img) for img in data],
                "reconstructed": [wandb.Image(img) for img in recon],
            }
        )
        return loss.item()

def collect_all_parquet_files(basepath):
    samples = []
    for dirpath, dirnames, filenames in os.walk(basepath):
        for filename in filenames:
            if filename.split(".")[-1] == "parquet": samples.append(os.path.join(dirpath, filename))
    return samples


def change_chunk_id_for_dataloader(plus_one_idx, parquet_paths, samples_per_chunk, chunk_id, batch_size, num_workers):
    print("About to change/prepare dataset for dataloader")
    start = time.perf_counter()
    if chunk_id < plus_one_idx: samples_per_chunk += 1
    paths = samples[chunk_id*samples_per_chunk:(chunk_id+1)*samples_per_chunk]
    dataset = Dataset.from_parquet(paths)
    end = time.perf_counter()
    time_diff = end - start
    print(f"Done changing/preparing dataset for dataloader. It took {time_diff:.0f} sec for chunk ID {chunk_id}.")
    dataset = dataset.with_transform(preprocess_train)
    dataloader = DataLoader(
            dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers
        )
    return dataloader


def update_chunk_id(chunk_id, num_chunks):
    chunk_id += 1
    chunk_id %= num_chunk
    return chunk_id

def main():
    args = parse_args()
    if args.report_to == "wandb":
        import wandb
        wandb.init(
            project="gamengen-vae-training",
            config={
                # Model parameters
                "model": PRETRAINED_MODEL_NAME_OR_PATH,
                # Training parameters
                "num_steps": args.max_train_steps,
                "eval_step": args.validation_steps,
                "batch_size": args.batch_size,
                "learning_rate": args.learning_rate,
                "gradient_clip_norm": args.gradient_clipping,
                "hf_model_folder": args.hf_model_folder,
            },
            name=f"vae-finetuning-{datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}",
        )

    parquet_paths = collect_all_parquet_files(args.dataset_basepath)
    random.shuffle(parquet_paths)
    samples_per_chunk = len(parquet_paths) // args.num_chunks
    plus_one_idx = len(parquet_paths) - samples_per_chunk * args.num_chunks
    
    # Dataset Setup
    chunk_id = 0
    dataloader = change_chunk_id_for_dataloader(plus_one_idx, parquet_paths, samples_per_chunk, chunk_id, batch_size, num_workers)

    # Model Setup
    vae = AutoencoderKL.from_pretrained(PRETRAINED_MODEL_NAME_OR_PATH, subfolder="vae")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = vae.to(device)
    make_decoder_trainable(model)
    # Optimizer Setup
    optimizer_cls = optim.AdamW if args.use_adamw else optim.Adafactor
    optimizer = optimizer_cls(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )

    if args.lr_scheduler == "constant_with_warmup":
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=NUM_WARMUP_STEPS,
            num_training_steps=NUM_EPOCHS * len(train_loader),
        )

    step = 0
    progress_bar = tqdm(
                        range(step, args.max_train_steps),
                        initial=step,
                        total=args.max_train_steps,
                        desc="Steps",
                        # Only show the progress bar once on each machine.
                        #disable=not accelerator.is_local_main_process,
                   )
    
    break_while = False
    while True:
        for batch in train_dataloader:
            model.train()
            data = batch["pixel_values"].to(device)
            optimizer.zero_grad()

            """
            The following part is not sampled as for the encoder output according to the link below.
            https://github.com/huggingface/diffusers/blob/c586aadef6bb66d355fa40a2b95a0bea8a6fe79c/src/diffusers/models/autoencoders/autoencoder_kl.py#L438
            Don't know why. May change it later.
            """
            reconstruction = model(data).sample
            loss = F.mse_loss(reconstruction, data, reduction="mean")

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.gradient_clipping)
            optimizer.step()
            if args.lr_scheduler == "constant_with_warmup": scheduler.step()

            if args.lr_scheduler == "constant_with_warmup": current_lr = scheduler.get_last_lr()[0]

            log_dic = {"loss": loss.item()}
            if args.lr_scheduler == "constant_with_warmup": log_dic["lr"] = current_lr
            progress_bar.set_postfix(log_dic)
            if args.report_to == "wandb": wandb.log(log_dic)

            progress_bar.update(1)
            step += 1

            if step % args.validation_steps == 0:
                # Use only two samples for validation
                val_loss = eval_model_mod(model, data[:2])
                # save model to hub
                model.save_pretrained(
                    os.path.join(args.output_dir, "vae"),
                    repo_id=args.repo_id if args.push_to_hub else None,
                    push_to_hub=args.push_to_hub,
                )
                #if args.report_to == "wandb": wandb.log({"val_loss": val_loss})

            if step >= args.max_train_steps:
                break_while = True
                break
            if step % args.chunk_id_interval == 0: chunk_id = update_chunk_id(chunk_id, num_chunks)
        
        if break_while: break


if __name__ == "__main__":
    main()
