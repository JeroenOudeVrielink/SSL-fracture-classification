#!/usr/bin/env bash




date_time=`date +"%m-%d_%T"`


seeds=(2 3)
for seed in ${seeds[@]}; do
    run_name="Gen3_v1_imgnet_pretrained_color_jitter_rotation_lr_schedule_clip09"
    ckpt_dir_path="/data/classification_models/Gen3_v1_imgnet_pretrained_color_jitter_rotation_lr_schedule_clip09_04-13_17:12:13"
    root_dir=data/${run_name}_${date_time}
    pretrained_method="classification"
    ./run_classification_cmmd.sh $run_name $ckpt_dir_path $root_dir $seed $pretrained_method
done 

