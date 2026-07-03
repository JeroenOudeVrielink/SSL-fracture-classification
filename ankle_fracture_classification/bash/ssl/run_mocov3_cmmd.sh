#!/usr/bin/env bash

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH=${PYTHONPATH}:$(pwd)/..

cd ../.. 
run_name=$1
ckpt_dir_path=$2
root_dir=$3

# Pretraining Params
pretrained_method="mocov3"

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
  checkpoint_0019
  checkpoint_0039
  checkpoint_0059
  checkpoint_0079
  checkpoint_0099
  checkpoint_0119
  checkpoint_0139
  checkpoint_0159
  checkpoint_0179
  checkpoint_0199
  checkpoint_0219
  checkpoint_0239
  checkpoint_0259
  checkpoint_0279
  checkpoint_0299
)


date_time=`date +"%m-%d_%T"`
root_dir=data/${run_name}_${date_time}

for seed in ${seeds[@]}; do
  save_dir=${root_dir}/seed_${seed}

  for checkpoint in "${checkpoints[@]}"; do
    identifier=${run_name}_${arch}_e${num_epochs}_b${batch_size}_s${seed}_${checkpoint}
    ckpt_path=${ckpt_dir_path}/${checkpoint}.pth.tar
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
