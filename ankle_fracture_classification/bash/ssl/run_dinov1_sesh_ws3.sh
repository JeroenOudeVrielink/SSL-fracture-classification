#!/usr/bin/env bash

sleep 8h

seeds=(1 2 3)


date_time=`date +"%m-%d_%T"`

for seed in ${seeds[@]}; do
    run_name="dinov1_bs128_increased_solarization"
    ckpt_dir_path="/data/dinov1_models/dinov1_bs128_increased_solarization_2024-03-13_04-44-51"
    root_dir=data/${run_name}_${date_time}
    ./run_dinov1_cmmd.sh $run_name $ckpt_dir_path $root_dir $seed

    run_name="dinov1_bs128_smoothing_insteadof_blurr"
    ckpt_dir_path="/data/dinov1_models/dinov1_bs128_smoothing_insteadof_blurr_2024-03-13_19-49-21"
    root_dir=data/${run_name}_${date_time}
    ./run_dinov1_cmmd.sh $run_name $ckpt_dir_path $root_dir $seed

    run_name="dinov1_bs128_relu_activation"
    ckpt_dir_path="/data/dinov1_models/dinov1_bs128_relu_activation_2024-03-14_06-41-38"
    root_dir=data/${run_name}_${date_time}
    ./run_dinov1_cmmd.sh $run_name $ckpt_dir_path $root_dir $seed
done