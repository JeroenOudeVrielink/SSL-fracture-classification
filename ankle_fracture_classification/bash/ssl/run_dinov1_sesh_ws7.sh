#!/usr/bin/env bash


seeds=(1 2 3)


date_time=`date +"%m-%d_%T"`
# date_time="02-19_05:44:49"

for seed in ${seeds[@]}; do
    run_name="dinov1_bs64_ep60_super_fmap_v3_ksize5"
    ckpt_dir_path="/data/dinov1_models/dinov1_bs64_ep460_super_fmap_v3_ksize5_2024-03-16_19-10-05"
    root_dir=data/${run_name}_${date_time}
    ./run_dinov1_cmmd.sh $run_name $ckpt_dir_path $root_dir $seed

    run_name="dinov1_bs128_increased_crop_scale_v2"
    ckpt_dir_path="/data/dinov1_models/dinov1_bs128_increased_crop_scale_v2_2024-03-18_01-49-05"
    root_dir=data/${run_name}_${date_time}
    ./run_dinov1_cmmd.sh $run_name $ckpt_dir_path $root_dir $seed

    run_name="dinov1_bs64_ep60_super_fmap_v3_ksize9"
    ckpt_dir_path="/data/dinov1_models/dinov1_bs64_ep460_super_fmap_v3_ksize9_2024-03-19_10-18-33"
    root_dir=data/${run_name}_${date_time}
    ./run_dinov1_cmmd.sh $run_name $ckpt_dir_path $root_dir $seed
done
