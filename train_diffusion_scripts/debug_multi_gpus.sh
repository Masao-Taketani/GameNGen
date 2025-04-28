accelerate launch --multi_gpu --mixed_precision=bf16 train_text_to_image.py \
	--dataset_name arnaudstiegler/vizdoom-episode \
	--train_batch_size 60 \
   	--learning_rate 5e-5 \
   	--num_train_epochs 3 \
   	--validation_steps 10 \
   	--dataloader_num_workers 18 \
	--max_train_samples 2 \
	--use_cfg \
	--output_dir sd-model-finetuned