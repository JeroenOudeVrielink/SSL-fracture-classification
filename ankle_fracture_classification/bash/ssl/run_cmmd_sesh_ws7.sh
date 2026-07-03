#!/usr/bin/env bash


seeds=(3)


date_time=`date +"%m-%d_%T"`
# date_time="02-19_05:44:49"

for seed in ${seeds[@]}; do
    run_name="moco_v2_bs512_base_params"
    ckpt_dir_path="/data/moco_models/moco_v2_bs512_base_params_2024-03-20_02-36-24"
    root_dir=data/${run_name}_${date_time}
    ./run_mocov2_cmmd.sh $run_name $ckpt_dir_path $root_dir $seed

    run_name="vicregl_bs128_alpha1"
    ckpt_dir_path="/data/vicregl_models/vicregl_bs128_alpha1_2024-03-23_07-29-10"
    root_dir=data/${run_name}_${date_time}
    ./run_vicregl_cmmd.sh $run_name $ckpt_dir_path $root_dir $seed
done
