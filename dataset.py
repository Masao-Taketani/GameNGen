import torch
from PIL import Image
import io
from torchvision import transforms
from datasets import load_dataset, Dataset
from datasets.arrow_dataset import Column
import random
import os
import logging
import math
from diffusers.models.autoencoders.vae import DiagonalGaussianDistribution
import numpy as np
import time

from config_sd import HEIGHT, WIDTH, H_PAD, W_PAD, BUFFER_SIZE, ZERO_OUT_ACTION_CONDITIONING_PROB
from data_augmentation import no_img_conditioning_augmentation, no_latent_img_conditioning_augmentation


IMG_TRANSFORMS = transforms.Compose(
        [
            transforms.Resize((HEIGHT, WIDTH), interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.Pad([W_PAD, H_PAD // 2, W_PAD, H_PAD // 2]),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ]
    )

def collate_fn(examples):
    processed_images = []
    for example in examples:
        #print("example:", example)
        # BUFFER_SIZE conditioning frames + 1 target frame
        processed_images.append(
            torch.stack(example["pixel_values"]))

    # Stack all examples
    # images has shape: (batch_size, frame_buffer, 3, height, width)
    images = torch.stack(processed_images)
    images = images.to(memory_format=torch.contiguous_format).float()

    # TODO: UGLY HACK
    images = no_img_conditioning_augmentation(images, prob=ZERO_OUT_ACTION_CONDITIONING_PROB)
    return {
        "pixel_values": images,
        #"input_ids": torch.stack([torch.tensor(example["input_ids"][:BUFFER_SIZE+1]).clone().detach() for example in examples]),
        "input_ids": torch.stack([example["input_ids"][:BUFFER_SIZE+1] for example in examples]),
    }


def preprocess_train(examples):
    images = [
            IMG_TRANSFORMS(Image.open(io.BytesIO(img)).convert("RGB"))
            for img in examples["frames"]
        ]

    actions = torch.tensor(examples["actions"]) if isinstance(examples["actions"], list) else examples["actions"]
    return {"pixel_values": images, "input_ids": actions}


class EpisodeDataset:
    def __init__(self, dataset_name: str):
        self.dataset = load_dataset(dataset_name)['train']
        self.action_dim = max(action for action in self.dataset['actions'])
        self.dataset = self.dataset.with_transform(preprocess_train)

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        if idx < BUFFER_SIZE:
            padding = [IMG_TRANSFORMS(Image.new('RGB', (WIDTH, HEIGHT), color='black')) for _ in range(BUFFER_SIZE - idx)]
            return {'pixel_values': padding + self.dataset[:idx+1]['pixel_values'], 'input_ids': torch.concat([torch.zeros(len(padding), dtype=torch.long), self.dataset[:idx+1]['input_ids']])}
        return self.dataset[idx-BUFFER_SIZE:idx+1]

    def get_action_dim(self) -> int:
        return self.action_dim


class EpisodeDatasetPrep:
    def __init__(self, basepath: str, num_chunk: int = None, chunk_id: int = None):
        self.samples = []
        self.epi_id = 0
        for dirpath, dirnames, filenames in os.walk(basepath):
            for filename in filenames:
                if filename.split(".")[-1] == "parquet": self.samples.append(os.path.join(dirpath, filename))
        if num_chunk is not None and chunk_id is not None:
            chunk_size = math.ceil(len(self.samples) / num_chunk)
            self.samples = self.samples[chunk_id*chunk_size:(chunk_id+1)*chunk_size]
            self.epi_id = chunk_id * chunk_size

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        path = self.samples[idx]
        dataset = Dataset.from_parquet(path)
        length = len(dataset)
        images = [
            IMG_TRANSFORMS(Image.open(io.BytesIO(img)).convert("RGB"))
            for img in dataset["frames"]
        ]
        actions = torch.tensor(dataset["actions"]) if isinstance(dataset["actions"], list) or isinstance(dataset["actions"], Column) else dataset["actions"]
        return torch.stack(images), actions


class EpisodeDatasetLatent:
    def __init__(self, basepath: str, action_dim: int):
        self.action_dim = action_dim
        self.samples = []
        for dirpath, dirnames, filenames in os.walk(basepath):
            for filename in filenames:
                if filename.split(".")[-1] == "pt": self.samples.append(os.path.join(dirpath, filename))

    def __len__(self) -> int:
        return len(self.samples)

    def load_latent_black(self, data_path):
        dpath = os.path.dirname(data_path)
        return torch.load(os.path.join(dpath, "latent_black.pt"))

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        path = self.samples[idx]
        epi_data = torch.load(path)
        length = epi_data["parameters"].shape[0]
        parameters = epi_data["parameters"]
        actions = epi_data["actions"]
        
        # Since each data has to include the buffer (0<=buffer<=BUFFER_SIZE) and label (prediction idx), the prediction idx has to be at least 0
        # and up to length - 1
        pred_idx = random.randint(0, length-1)
        if pred_idx < BUFFER_SIZE:
            latent_black = self.load_latent_black(path)
            padding = DiagonalGaussianDistribution(latent_black.repeat(BUFFER_SIZE - pred_idx, 1, 1, 1)).sample()
            latents = DiagonalGaussianDistribution(parameters[:pred_idx+1]).sample()
            latents = torch.concat([padding, latents])
            actions = torch.concat([torch.zeros(len(padding), dtype=torch.long), actions[:pred_idx+1]])
        else:
            parameters, actions = parameters[pred_idx-BUFFER_SIZE:pred_idx+1], actions[pred_idx-BUFFER_SIZE:pred_idx+1]
            latents = DiagonalGaussianDistribution(parameters).sample()
        
        return {'latent_values': latents, 'input_ids': actions}


class EpisodeDatasetMod:
    def __init__(self, basepath: str, action_dim: int):
        self.action_dim = action_dim
        self.samples = []
        for dirpath, dirnames, filenames in os.walk(basepath):
            for filename in filenames:
                if filename.split(".")[-1] == "parquet": self.samples.append(os.path.join(dirpath, filename))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        path = self.samples[idx]
        dataset = Dataset.from_parquet(path)
        length = len(dataset)
        images = [
            IMG_TRANSFORMS(Image.open(io.BytesIO(img)).convert("RGB"))
            for img in dataset["frames"]
        ]
        actions = torch.tensor(dataset["actions"]) if isinstance(dataset["actions"], list) or isinstance(dataset["actions"], Column) else dataset["actions"]

        # Since each data includes the buffer and label, the start has to be at least 1
        # and the last index has to be length - 1
        start_ep_idx = random.randint(1, length-1)
        if start_ep_idx < BUFFER_SIZE:
            padding = [IMG_TRANSFORMS(Image.new('RGB', (WIDTH, HEIGHT), color='black')) for _ in range(BUFFER_SIZE - start_ep_idx)]
            return {'pixel_values': padding + images[:start_ep_idx+1], 'input_ids': torch.concat([torch.zeros(len(padding), dtype=torch.long), actions[:start_ep_idx+1]])}
        return {'pixel_values': images[start_ep_idx-BUFFER_SIZE:start_ep_idx+1], 'input_ids': actions[start_ep_idx-BUFFER_SIZE:start_ep_idx+1]}


def get_dataloader(dataset_name: str, batch_size: int = 1, num_workers: int = 1, shuffle: bool = False) -> torch.utils.data.DataLoader:
    dataset = EpisodeDataset(dataset_name)
    return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_fn, num_workers=num_workers)

def get_dataloader_prep(basepath: str, num_chunk: int = None, chunk_id: int = None, num_workers: int = 1) -> torch.utils.data.DataLoader:
    dataset = EpisodeDatasetPrep(basepath, num_chunk, chunk_id)
    # batch_size 1 indicates the loader returns one episode data each time
    return torch.utils.data.DataLoader(dataset, batch_size=1, sampler=torch.utils.data.SequentialSampler(dataset), num_workers=num_workers, drop_last=False)

def get_dataloader_mod(basepath: str, action_dim: int, batch_size: int = 1, num_workers: int = 1, shuffle: bool = False) -> torch.utils.data.DataLoader:
    dataset = EpisodeDatasetMod(basepath, action_dim)
    return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_fn, num_workers=num_workers, persistent_workers=True, drop_last=True)

def get_latent_dataloader(basepath: str, action_dim: int, batch_size: int = 1, num_workers: int = 1, shuffle: bool = False) -> torch.utils.data.DataLoader:
    dataset = EpisodeDatasetLatent(basepath, action_dim)
    return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, persistent_workers=True, drop_last=True)

def get_single_batch(dataset_name: str) -> dict[str, torch.Tensor]:
    dataloader = get_dataloader(dataset_name, batch_size=1, num_workers=1, shuffle=False)
    return next(iter(dataloader))

def get_single_batch_mod(basepath: str, action_dim: int) -> dict[str, torch.Tensor]:
    dataloader = get_dataloader_mod(basepath, action_dim, batch_size=1, num_workers=1, shuffle=False)
    return next(iter(dataloader))