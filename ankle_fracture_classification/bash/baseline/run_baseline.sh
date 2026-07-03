#!/usr/bin/env bash

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH=${PYTHONPATH}:$(pwd)/..


cd ../..

run_name=baseline_bs32_save_model
date_time=`date +"%m-%d_%T"`
save_dir=data/${run_name}_${date_time}

seeds=(1 2 3)

# Data params
num_workers=6
save_interval=1

# Training params
num_epochs=100
batch_size=16

# Network params
arch=resnet50
enable_attn_loss=False
pretrained_method="imagenet_pretrained"

for seed in ${seeds[@]}; do
  identifier=${run_name}_${arch}_e${num_epochs}_b${batch_size}_s${seed}
  python3 -u modelling/train.py \
                      --save_path ${save_dir}/${identifier} \
                      --seed ${seed} \
                      --num_workers ${num_workers} \
                      --save_interval ${save_interval} \
                      --num_epochs ${num_epochs} \
                      --batch_size ${batch_size} \
                      --arch ${arch} \
                      --enable_attn_loss ${enable_attn_loss} \
                      --pretrained_method ${pretrained_method}
done
