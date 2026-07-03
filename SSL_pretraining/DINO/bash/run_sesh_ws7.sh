#!/usr/bin/env bash


cd ..

python main_dino.py \
--exp_name dinov1_bs64_ep460_super_fmap_v3_ksize9 \
--output_dir ~/data/dinov1_models \
--data_path ~/AIML_rot_corrected \
--arch resnet50 \
--optimizer sgd \
--lr 0.03 \
--weight_decay 1e-4 \
--weight_decay_end 1e-4 \
--global_crops_scale 0.14 1 \
--local_crops_scale 0.05 0.14 \
--batch_size_per_gpu 64 \
--num_workers 10 \
--wandb_log_freq 100 \
--saveckp_freq 20 \
--epochs 61 \
--use_conv_head True \
--out_dim 50176 \
--kernel_size 9

