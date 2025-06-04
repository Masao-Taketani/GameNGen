import torch
from PIL import Image
import io
from torchvision import transforms
from datasets import load_dataset
import random

from config_sd import HEIGHT, WIDTH, BUFFER_SIZE, ZERO_OUT_ACTION_CONDITIONING_PROB
from data_augmentation import no_img_conditioning_augmentation


IMG_TRANSFORMS = transforms.Compose(
        [
            transforms.Resize((HEIGHT, WIDTH), interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.Pad([0, 8, 0, 8]),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ]
    )

def collate_fn(examples):
    processed_images = []
    for example in examples:
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
        "input_ids": torch.stack([example["input_ids"][:BUFFER_SIZE+1].clone().detach() for example in examples]),
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
        actions = torch.tensor(dataset["actions"]) if isinstance(dataset["actions"], list) else dataset["actions"]
        # Since each data includes the buffer and label, the start has to be at least 1
        # and the last index has to be length - 1
        start_ep_idx = random.randint(1, length-1)
        if start_ep_idx < BUFFER_SIZE:
            padding = [IMG_TRANSFORMS(Image.new('RGB', (WIDTH, HEIGHT), color='black')) for _ in range(BUFFER_SIZE - start_ep_idx)]
            return {'pixel_values': padding + images[:idx+1], 'input_ids': torch.concat([torch.zeros(len(padding), dtype=torch.long), actions[:idx+1])}
        return {'pixel_values': images[idx-BUFFER_SIZE:idx+1], 'input_ids': actions[idx-BUFFER_SIZE:idx+1]}


def get_dataloader(dataset_name: str, batch_size: int = 1, num_workers: int = 1, shuffle: bool = False) -> torch.utils.data.DataLoader:
    dataset = EpisodeDataset(dataset_name)
    return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_fn, num_workers=num_workers)

def get_dataloader_mod(basepath: str, batch_size: int = 1, num_workers: int = 1, shuffle: bool = False) -> torch.utils.data.DataLoader:
    dataset = EpisodeDatasetMod(basepath)
    return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_fn, num_workers=num_workers)

def get_single_batch(dataset_name: str) -> dict[str, torch.Tensor]:
    dataloader = get_dataloader(dataset_name, batch_size=1, num_workers=1, shuffle=False)
    return next(iter(dataloader))

def get_single_batch_mod(basepath: str) -> dict[str, torch.Tensor]:
    dataloader = get_dataloader_mod(basepath, batch_size=1, num_workers=1, shuffle=False)
    return next(iter(dataloader))