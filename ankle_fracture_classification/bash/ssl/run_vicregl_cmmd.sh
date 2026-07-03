#!/usr/bin/env bash

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH=${PYTHONPATH}:$(pwd)/..

cd ../.. 
run_name=$1
ckpt_dir_path=$2
root_dir=$3

# Pretraining Params
pretrained_method="vicregl"

# Data params
num_workers=6
save_interval=1

# Training params
num_epochs=100
batch_size=32

# Network params
arch=resnet50
enable_attn_loss=False

seeds=($4)
checkpoints=(
  # model_resnet50_ep20
  # model_resnet50_ep40
  # model_resnet50_ep60
  model_resnet50_ep80
  # model_resnet50_ep100
  # model_resnet50_ep120
  # model_resnet50_ep140
  # model_resnet50_ep160
  # model_resnet50_ep180
  # model_resnet50_ep200
  # model_resnet50_ep220
  # model_resnet50_ep240
  # model_resnet50_ep260
  # model_resnet50_ep280
  # model_resnet50_ep300
)

date_time=`date +"%m-%d_%T"`
root_dir=data/${run_name}_${date_time}

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
                        --pretrained_method ${pretrained_method} \
                        --save_model_checkpoint False
  done
done
