#!/usr/bin/env bash

cd ssl

seeds=(1 2 3)
date_time=`date +"%m-%d_%T"`

for seed in ${seeds[@]}; do
    run_name="vicregl_bs128_cov_coeff_50"
    ckpt_dir_path="/models_local/vicregl_bs128_cov_coeff_50_2024-06-03_05-11-02"
    root_dir=data/${run_name}_${date_time}
    ./run_vicregl_cmmd.sh $run_name $ckpt_dir_path $root_dir $seed
done
