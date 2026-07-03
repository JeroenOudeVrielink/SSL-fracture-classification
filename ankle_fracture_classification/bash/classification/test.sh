#!/usr/bin/env bash


seeds=(1 2 3)


date_time=`date +"%m-%d_%T"`

for seed in ${seeds[@]}; do
    run_name="test"
    ckpt_dir_path="/data/classification_models/Gen3_v2_imgnet_pretrained_color_jitter_rotation_lr_schedule_04-14_05:42:37"
    root_dir=data/${run_name}_${date_time}
    pretrained_method="classification_adapt_layer"
    ./run_classification_cmmd.sh $run_name $ckpt_dir_path $root_dir $seed $pretrained_method
done
