accelerate launch --multi_gpu --mixed_precision=bf16 train_text_to_image.py \
	--dataset_name arnaudstiegler/vizdoom-episode \
	--train_batch_size 60 \
   	--learning_rate 2e-5 \
   	--max_train_steps 20  \
    --validation_steps 5  \
   	--dataloader_num_workers 18 \
	--max_train_samples 2 \
	--use_cfg \
	--output_dir diffusion_model