# Diffusion Models Are Real-Time Game Engines

This is an unofficial repo of [GameNGen](https://arxiv.org/abs/2408.14837). I have referred to [arnaudstiegler/gameNgen-repro](https://github.com/arnaudstiegler/gameNgen-repro/tree/main) in order to create this repo.

## Modifications
Here are the list of modifications from [arnaudstiegler/gameNgen-repro](https://github.com/arnaudstiegler/gameNgen-repro/tree/main).
- Fixed the environment setup with Anaconda and write down the precise commands in order to properly reproduce the code (For this development, I start from `continuumio/anaconda3:latest` Docker image).
- Use the original paper's training setup as much as I can
- Incorporate much more efficient diffusion training method and data files

## To Do
- [x] Set `MOVE_LEFT` and `MOVE_RIGHT` as independent actions
- [x] Enable distributed training
- [ ] Implement playable generative environments
- [x] Try using the original settings
  - **VizDoom**
  - [x] Add `MOVE_BACKWARD` action
  - [x] Enable to collect VizDoom dataset with multiprocessing during RL policy training
  - **Diffusion Models**
  - [x] 2e-5 for the learning rate
  - [x] Adafactor optimizer
  - [x] A context length of 64
  - [x] Train iteratively with respect to steps, not epochs<br>
  ~~- [ ] Set action embedding as `trainable_params`~~
  - [x] Wrap `unet` and `action_embedding` as one model in order to synchronize those two models' parameters with multi-gpu training. For more details, refer to the [issue](https://github.com/huggingface/accelerate/issues/668)
  - [x] Pad every image frame to get 320x256 from 320x240
  - [x] Create dataloader that makes sure it won't pick frames from two distinct episodes for each batch
- [x] Efficient Diffusion Training
  - [x] Convert image dataset into latent embedding one and train only using the embeddings from the beginning, not handling image dataset at all during the training
  - [x] Convert dataset file format from `.parquet` (for images & actions) to `.pt` (for latent embeddings & actions) to directly handle Torch tensors to train

## Artifacts

All artifacts are available on Hugging Face Hub:

Checkpoints:
- [gameNgen-baseline-20ksteps](https://huggingface.co/arnaudstiegler/gameNgen-baseline-20ksteps)
- [sd-model-gameNgen-60ksteps](https://huggingface.co/arnaudstiegler/sd-model-gameNgen-60ksteps)

Datasets:
- [vizdoom-5-episodes-skipframe-4-lvl5](https://huggingface.co/arnaudstiegler/vizdoom-5-episodes-skipframe-4-lvl5)
- [vizdoom-500-episodes-skipframe-4-lvl5](https://huggingface.co/arnaudstiegler/vizdoom-500-episodes-skipframe-4-lvl5)

Vizdoom Agent:
- `ViZDoomPPO/logs/models/deathmatch_simple/best_model.zip` (local)

## Scripts

### Generate the training data

First, follow the commands below in order to create an environment to train Vizdoom agent.
```
conda create -n vizdoom python=3.11 -y
conda activate vizdoom
apt update && apt install libgl1 swig g++ -y
pip install setuptools==65.5.0 pip==21 wheel==0.38.0
cd ViZDoomPPO/
pip install -r vizdoom_requirements.txt
```

Then, run the following command to train an agent and build the dataset on vizdoom at the same time.
```
python train_ppo_and_collect_data_parallel.py --out_base_dir [dataset directory path]
```

Once the agent is trained, generate episodes and upload them as a HF dataset using:
```
python load_model_generate_dataset.py --episodes {number of episodes} --output parquet --upload --hf_repo {name of the repo}
```

Note: you can also generate a gif file to QA the behavior of the agent by running:
```
python load_model_generate_dataset.py --episodes 1 --output gif
```

### Convert image dataset to latent embedding dataset
You are going to convert the collected image dataset (`.parquet` files) into latent embedding dataset (`.pt` files), so that you can save lots of memory during diffusion training. For that, you are recommended to use at least one GPU. Use the following command to execute with single GPU.
```
python encode_images.py --dataset_basepath [directory path under which parquet files are placed] --save_dir_path [pt file save directory path] --dataloader_num_workers [number of workers for dataloader] --batch_size [batch size to process for one step] --dtype [data type used for inference]
```

You can also use multiple GPUs to process the conversion in parallel (each GPU processes handles a different episode). In order to do that, specify number of chunks for parallel processing. Also, specifry which chunk ID each GPU process needs to handle. Lastly, assign GPU ID to run the command. For example, if you device to use 3 GPUs, the number of chunks should be 3, chunk ID and GPU ID should be a number between 0 and 2. Use the following command to execute with multiple GPUs.
```
python encode_images.py --dataset_basepath [directory path under which parquet files are placed] --save_dir_path [pt file save directory path] --dataloader_num_workers [number of workers for dataloader] --batch_size [batch size to process for one step] --dtype [data type used for inference] --num_chunks [number of chunks to split your dataset] --chunk_id [chunk ID] --gpu_id [GPU ID]
```


### Train the diffusion model

Second step is that follow the commands below in order to create an environment to train diffusion model.
```
conda deactivate
conda create -n diffusion python=3.11 -y
conda activate diffusion
pip install -r diffusion_requirements.txt
```

#### Single GPU
If you only have single gpu, follow the instruction below to train the diffusion model.
Debug training with single GPU.
```
sh train_diffusion_scripts/debug_single_gpu.sh
```

Start full training with single GPU.
```
sh train_diffusion_scripts/single_gpu.sh
```

#### Multiple GPUs
If you have more than single GPU, follow the instruction below to train the diffusion model.
Debug training with multiple GPUs.
```
sh train_diffusion_scripts/debug_multi_gpus.sh
```

Start full training with multiple GPUs.
```
sh train_diffusion_scripts/multi_gpus.sh
```

### Train the auto-encoder

```
python finetune_autoencoder.py --hf_model_folder {path to the model folder}
```

### Run inference (generating a single image)

```
python run_inference.py --model_folder arnaudstiegler/sd-model-gameNgen-60ksteps
```

### Run autoregressive inference

This will generate rollouts, where each new frame is generated by the model conditioned on the previous frames and actions.
We initially fill the buffer using the small dataset, and sample actions from the dataset (i.e it matches what the agent did in the episode)

```
python run_autoregressive.py --model_folder arnaudstiegler/sd-model-gameNgen-60ksteps
```

## References

### Paper
- [Diffusion Models Are Real-Time Game Engines](https://arxiv.org/abs/2408.14837)

### GitHub Repos
- [arnaudstiegler/gameNgen-repro](https://github.com/arnaudstiegler/gameNgen-repro/tree/main)
- [lkiel/rl-doom](https://github.com/lkiel/rl-doom)
