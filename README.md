# Diffusion Models Are Real-Time Game Engines

This is an unofficial repo of [GameNGen](https://arxiv.org/abs/2408.14837). I have referred to [arnaudstiegler/gameNgen-repro](https://github.com/arnaudstiegler/gameNgen-repro/tree/main) in order to create this repo.

## Modifications
Here are the list of modifications from [arnaudstiegler/gameNgen-repro](https://github.com/arnaudstiegler/gameNgen-repro/tree/main).
- Fixed the environment setup with Anaconda and write down the precise commands in order to properly reproduce the code (For this development, I start from `continuumio/anaconda3:latest` Docker image). Thus, to precisely replicate the dev environment, set up the Docker environment with the following command.
```
cd GameNGen
docker run --log-opt max-size=10m --log-opt max-file=2 -it --rm --gpus '"device={input your GPU device ID(s) here}"' -v .:/work continuumio/anaconda3:latest
```
Although `--log-opt max-file` is optional, I strongly recommend using `--log-opt max-size` option since the RL training script for ViZDoom outputs lots of logs to stdout, which conssumes excessive disk space.
- Use the original paper's training setup as much as I can
- Incorporate much more efficient diffusion training method and data files

## To Do
- [x] Set `MOVE_LEFT` and `MOVE_RIGHT` as independent actions
- [x] Add no-op action to the action space
- [x] Enable distributed training
- [ ] Implement playable generative environments
- [x] Try using the original settings
  - **VizDoom**
  - [x] Add `MOVE_BACKWARD` action
  - [x] Enable to collect VizDoom dataset with multiprocessing during RL policy training
  - **Diffusion Models**
  - [] Use 4 for the number of inference steps for training evaluation and autoregressive inference<br>
  ~~- [x] 2e-5 for the learning rate~~<br>
  ~~- [x] Adafactor optimizer~~<br>
  (I've tried the above two, but got worse results compared to the settings used for [arnaudstiegler/gameNgen-repro](https://github.com/arnaudstiegler/gameNgen-repro/tree/main). So, I didn't use those at the end)
  - [x] A context length of 64
  - [x] Train iteratively with respect to steps, not epochs<br>
  ~~- [ ] Set action embedding as `trainable_params`~~
  - [x] Wrap `unet` and `action_embedding` as one model in order to synchronize those two models' parameters with multi-gpu training. For more details, refer to the [issue](https://github.com/huggingface/accelerate/issues/668)
  - [x] Pad every image frame to get 320x256 from 320x240
  - [x] Create dataloader that makes sure it won't pick frames from two distinct episodes when creating each training sample
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

### Generate training data for diffusion models while training an RL agent 

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

### Train the diffusion model

Second step is that follow the commands below in order to create an environment to train diffusion model.
```
conda deactivate
conda create -n diffusion python=3.11 -y
conda activate diffusion
pip install -r diffusion_requirements.txt
```

### Convert image dataset to latent embedding dataset
You are going to convert the collected image dataset (`.parquet` files) into latent embedding dataset (`.pt` files), so that you can save lots of memory during diffusion training. For that, you are recommended to use at least one GPU. Use the following command to execute with single GPU.
```
python encode_images.py --dataset_basepath [directory path under which parquet files are placed] --save_dir_path [pt file save directory path] --dataloader_num_workers [number of workers for dataloader] --batch_size [batch size to process for one step] --dtype [data type used for inference]
```

You can also use multiple GPUs to process the conversion in parallel (each GPU processes handles a different episode). In order to do that, specify number of chunks for parallel processing. Also, specifry which chunk ID each GPU process needs to handle. Lastly, assign GPU ID to run the command. For example, if you decide to use 3 GPUs, the number of chunks should be 3, chunk ID and GPU ID should be a number between 0 and 2. Use the following command to execute with multiple GPUs.
```
python encode_images.py --dataset_basepath [directory path under which parquet files are placed] --save_dir_path [pt file save directory path] --dataloader_num_workers [number of workers for dataloader] --batch_size [batch size to process for one step] --dtype [data type used for inference] --num_chunks [number of chunks to split your dataset] --chunk_id [chunk ID] --gpu_id [GPU ID]
```

#### Single GPU
If you only have single gpu, after modifying some arguments(such as dataset path) of `train_diffusion_scripts/single_gpu_latents.sh`, follow the instruction below to train the diffusion model.
```
sh train_diffusion_scripts/single_gpu_latents.sh
```

#### Multiple GPUs
If you have more than single GPU, after modifying some arguments(such as dataset path) of `train_diffusion_scripts/multi_gpus_latents.sh`, follow the instruction below to train the diffusion model.
```
sh train_diffusion_scripts/multi_gpus_latents.sh
```

### Train the auto-encoder

#### Single GPU

#### Multiple GPUs
If you have more than single GPU, after modifying some arguments(such as dataset path) of `finetune_vae_scripts/multi_gpus.sh`, follow the instruction below to finetune the decoder of VAE
```
sh finetune_vae_scripts/multi_gpus.sh
```

### Run inference (generating a single image)

```
python run_inference.py --model_folder arnaudstiegler/sd-model-gameNgen-60ksteps
```

### Run autoregressive inference

The following command will generate rollouts, where each new frame is generated by the model conditioned on the previous frames and actions.
We initially fill the buffer using 64 frames of training data and their corresponding actions, and only use actions from the dataset after that (i.e it matches what the agent did in the episode).
Here are the explanations of arguments that are used when executing.
- start_from_pixels (store_true): Start autoregressive inference using original images. If not specified, the program starts autoregressive inference using latent images
- dataset_basepath (str): Specify your parquet directory path if `start_from_pixels` flag is raised. Otherwise, specify your pt directory path
- num_episodes (int): Total number of episodes to generate
- episode_length (int): Total steps for each episode to generate
- unet_model_folder (str): Specify your trained U-net folder
- vae_ft_model_folder (str): Specify your finetuned VAE folder
- num_inference_steps (int): Number of inference steps to generate each frame


```
python run_autoregressive.py --dataset_basepath [your parquet or pt path] --num_episodes [mumber of episodes] --episode_length [number of steps] --unet_model_folder [your unet folder] --vae_ft_model_folder [your vae folder] --num_inference_steps [number of inference steps] ([optional boolean flag]--start_from_pixels)
```

## References

### Paper
- [Diffusion Models Are Real-Time Game Engines](https://arxiv.org/abs/2408.14837)

### GitHub Repos
- [arnaudstiegler/gameNgen-repro](https://github.com/arnaudstiegler/gameNgen-repro/tree/main)
- [lkiel/rl-doom](https://github.com/lkiel/rl-doom)
