#!/usr/bin/env bash

# Num workers was 14 (optimal) but caused OOM errors for RAM on ws7

python main_dino.py \
--exp_name dinov1_bs128_ep300_no_local_to_global \
--output_dir /home/jvrielink/data/dinov1_models/dinov1_bs128_ep300_no_local_to_global_2024-02-01_10-16-32 \
--data_path /home/jvrielink/AIML_rot_corrected \
--arch resnet50 \
--optimizer sgd \
--lr 0.03 \
--weight_decay 1e-4 \
--weight_decay_end 1e-4 \
--global_crops_scale 0.9 1 \
--local_crops_scale 0.9 1 \
--batch_size_per_gpu 128 \
--num_workers 10 \
--wandb_log_freq 100 \
--saveckp_freq 10 \
--epochs 301 \
--from_ckpt True


