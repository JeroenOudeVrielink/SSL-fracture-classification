#!/usr/bin/env bash



seeds=(1 2 3)
date_time=`date +"%m-%d_%T"`

for seed in ${seeds[@]}; do
    run_name="dinov1_bs128_img_subset_ckpt180_300"
    ckpt_dir_path="/data/dinov1_models/dinov1_bs128_img_subset_2024-06-05_06-38-55"
    root_dir=data/${run_name}_${date_time}
    ./run_dinov1_cmmd.sh $run_name $ckpt_dir_path $root_dir $seed
done