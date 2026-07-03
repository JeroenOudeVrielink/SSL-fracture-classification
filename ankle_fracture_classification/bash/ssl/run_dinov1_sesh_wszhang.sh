#!/usr/bin/env bash

sleep 18h

seeds=(1 2 3)


date_time=`date +"%m-%d_%T"`
# date_time="02-19_05:44:49"

for seed in ${seeds[@]}; do
    run_name="dinov1_bs128_0.9995momentum_teacher"
    ckpt_dir_path="/data/dinov1_models/dinov1_bs128_0.9995momentum_teacher_2024-03-05_05-06-59"
    root_dir=data/${run_name}_${date_time}
    ./run_dinov1_cmmd.sh $run_name $ckpt_dir_path $root_dir $seed

    run_name="dinov1_bs128_increased_crop_scale"
    ckpt_dir_path="/data/dinov1_models/dinov1_bs128_increased_crop_scale_2024-03-05_22-02-55"
    root_dir=data/${run_name}_${date_time}
    ./run_dinov1_cmmd.sh $run_name $ckpt_dir_path $root_dir $seed
        
    run_name="dinov1_bs128_no_norm_last_layer"
    ckpt_dir_path="/data/dinov1_models/dinov1_bs128_no_norm_last_layer_2024-03-04_07-15-14"
    root_dir=data/${run_name}_${date_time}
    ./run_dinov1_cmmd.sh $run_name $ckpt_dir_path $root_dir $seed

    run_name="dinov1_bs128_base_param_ep61"
    ckpt_dir_path="/data/dinov1_models/dinov1_bs128_base_param_ep61_2024-03-06_22-49-12"
    root_dir=data/${run_name}_${date_time}
    ./run_dinov1_cmmd.sh $run_name $ckpt_dir_path $root_dir $seed
done
