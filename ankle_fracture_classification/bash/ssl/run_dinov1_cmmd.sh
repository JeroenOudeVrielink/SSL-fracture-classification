#!/usr/bin/env bash

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH=${PYTHONPATH}:$(pwd)/..

cd ../.. 
run_name=$1
ckpt_dir_path=$2
root_dir=$3

# Pretraining Params
pretrained_method="dinov1"

# Data params
num_workers=6
save_interval=1

# Training params
num_epochs=100
batch_size=16

# Network params
arch=resnet50
enable_attn_loss=False

seeds=($4)
checkpoints=(
    # checkpoint0020
    # checkpoint0040
    # checkpoint0060
    # checkpoint0080
    # checkpoint0100
    # checkpoint0120
    # checkpoint0140
    # checkpoint0160
    checkpoint0180
    checkpoint0200
    checkpoint0220
    checkpoint0240
    checkpoint0260
    checkpoint0280
    checkpoint0300
    # checkpoint0320
    # checkpoint0340
    # checkpoint0360
    # checkpoint0380
    # checkpoint0400
)


for seed in ${seeds[@]}; do
  save_dir=${root_dir}/seed_${seed}

  for checkpoint in "${checkpoints[@]}"; do
    identifier=${run_name}_${arch}_e${num_epochs}_b${batch_size}_s${seed}_${checkpoint}
    ckpt_path=${ckpt_dir_path}/${checkpoint}.pth
    python3 -u modelling/train.py \
                        --save_path ${save_dir}/${identifier} \
                        --seed ${seed} \
                        --num_workers ${num_workers} \
                        --save_interval ${save_interval} \
                        --num_epochs ${num_epochs} \
                        --batch_size ${batch_size} \
                        --arch ${arch} \
                        --enable_attn_loss ${enable_attn_loss} \
                        --ckpt_path ${ckpt_path} \
                        --pretrained_method ${pretrained_method}
  done
done
