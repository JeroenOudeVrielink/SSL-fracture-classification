#!/usr/bin/env bash

# Num workers was 14 (optimal) but caused OOM errors for RAM on ws7

cd ../..

python main_dino.py \
--exp_name dinov1_bs128_ep100_no_hflip \
--output_dir /dinov1_models/dinov1_bs128_ep100_no_hflip_2024-05-09_01-37-57 \
--data_path /AIML_rot_corrected \
--arch resnet50 \
--optimizer sgd \
--lr 0.03 \
--weight_decay 1e-4 \
--weight_decay_end 1e-4 \
--global_crops_scale 0.14 1 \
--local_crops_scale 0.05 0.14 \
--batch_size_per_gpu 128 \
--num_workers 14 \
--wandb_log_freq 100 \
--saveckp_freq 20 \
--epochs 101 \
--custom_augmentation True \
--p_horizontal_flip 0 \
--from_ckpt True


sleep 30s


python main_dino.py \
--exp_name dinov1_bs128_ep100_random_rotation \
--output_dir /dinov1_models \
--data_path /AIML_rot_corrected \
--arch resnet50 \
--optimizer sgd \
--lr 0.03 \
--weight_decay 1e-4 \
--weight_decay_end 1e-4 \
--global_crops_scale 0.14 1 \
--local_crops_scale 0.05 0.14 \
--batch_size_per_gpu 128 \
--num_workers 14 \
--wandb_log_freq 100 \
--saveckp_freq 20 \
--epochs 101 \
--custom_augmentation True \
--p_horizontal_flip 0 \
--p_random_rotation 0.5
