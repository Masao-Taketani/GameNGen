accelerate launch --multi_gpu --mixed_precision=bf16 train_text_to_image.py \
	--dataset_name arnaudstiegler/vizdoom-500-episodes-skipframe-4-lvl5 \
	--train_batch_size 60 \
   	--learning_rate 5e-5 \
   	--num_train_epochs 3 \
   	--validation_steps 1000 \
   	--dataloader_num_workers 18 \
	--use_cfg \
	--output_dir sd-model-finetuned \
    --lr_scheduler cosine