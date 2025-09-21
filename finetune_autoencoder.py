from datetime import datetime
import argparse

import torch
import torch.nn.functional as F
from datasets import load_dataset
from diffusers import AutoencoderKL
from torch import optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import get_cosine_schedule_with_warmup
from config_sd import PRETRAINED_MODEL_NAME_OR_PATH

import wandb
from dataset import preprocess_train

# Fine-tuning parameters
NUM_WARMUP_STEPS = 500
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-5
GRADIENT_CLIP_NORM = 1.0


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
        "--learning_rate",
        type=float,
        default=2e-5,
        help="Initial learning rate (after the potential warmup period) to use.",
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

    # Dataset Setup
    dataset = load_dataset(args.dataset_basepath)
    split_dataset = dataset["train"].train_test_split(test_size=500, seed=42)
    train_dataset = split_dataset["train"].with_transform(preprocess_train)
    test_dataset = split_dataset["test"].with_transform(preprocess_train)
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=8
    )
    test_loader = DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=8
    )
    # Model Setup
    vae = AutoencoderKL.from_pretrained(PRETRAINED_MODEL_NAME_OR_PATH, subfolder="vae")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = vae.to(device)
    make_decoder_trainable(model)
    # Optimizer Setup
    optimizer = optim.AdamW(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=NUM_WARMUP_STEPS,
        num_training_steps=NUM_EPOCHS * len(train_loader),
    )

    step = 0
    for epoch in range(NUM_EPOCHS):
        train_loss = 0
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{NUM_EPOCHS}")

        for batch in progress_bar:
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
            scheduler.step()

            train_loss += loss.item()
            current_lr = scheduler.get_last_lr()[0]

            progress_bar.set_postfix({"loss": loss.item(), "lr": current_lr})

            if args.report_to == "wandb":
                wandb.log(
                    {
                        "train_loss": loss.item(),
                        "learning_rate": current_lr,
                    }
                )

            step += 1
            if step % args.validation_steps == 0:
                test_loss = eval_model(model, test_loader)
                # save model to hub
                model.save_pretrained(
                    os.path.join(args.output_dir, "vae"),
                    repo_id=args.repo_id if args.push_to_hub else None,
                    push_to_hub=args.push_to_hub,
                )
                if args.report_to == "wandb": wandb.log({"test_loss": test_loss})


if __name__ == "__main__":
    main()
