#!/usr/bin/env bash

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH=${PYTHONPATH}:$(pwd)/..

cd ../.. 

run_name=spark_in448_bs32_ep400_laplace_recon
ckpt_dir_path="/data/spark_models/spark_in448_bs32_ep400_laplace_recon_01-25_06:46:24"

# Pretraining Params
pretrained_method="spark"

# Data params
num_workers=4
save_interval=100

# Training params
num_epochs=100
batch_size=16

# Network params
arch=resnet50
enable_attn_loss=False

seeds=(1 2 3)
checkpoints=(
  resnet50_1kpretrained_timm_style_ep20
  resnet50_1kpretrained_timm_style_ep40
  resnet50_1kpretrained_timm_style_ep60
  resnet50_1kpretrained_timm_style_ep80
  resnet50_1kpretrained_timm_style_ep100
  resnet50_1kpretrained_timm_style_ep120
  resnet50_1kpretrained_timm_style_ep140
  resnet50_1kpretrained_timm_style_ep160
  resnet50_1kpretrained_timm_style_ep180
  resnet50_1kpretrained_timm_style_ep200
  resnet50_1kpretrained_timm_style_ep220
  resnet50_1kpretrained_timm_style_ep240
  resnet50_1kpretrained_timm_style_ep260
  resnet50_1kpretrained_timm_style_ep280
  resnet50_1kpretrained_timm_style_ep300
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