#!/usr/bin/env bash

dirs_zhang=(
dinov1_bs128_ep1000_ckpt780_880_06-22_07:47:44
dinov1_bs128_ep1000_ckpt900_1000_06-22_07:49:45
)

data_dir=/home/zhibin/fracture-attention-guidance/data
for dir in "${dirs_zhang[@]}"; do
    echo copying dir $dir ....
    rsync -r --exclude *.jpg --exclude *.ckpt --rsh=ssh zhibin@10.12.98.141:$data_dir/$dir /home/jvrielink/fracture-attention-guidance/data/
    # rsync -r --exclude *.jpg --rsh=ssh zhibin@10.12.98.141:$data_dir/$dir /home/jeroen-ov/University/Master_thesis/fracture-attention-guidance/data/
done

dirs_workstation3=(
)

data_dir=~/fracture-attention-guidance/data
for dir in "${dirs_workstation3[@]}"; do
    echo copying dir $dir ....
    rsync -r --exclude *.jpg jvrielink@129.127.9.231:$data_dir/$dir /home/jvrielink/fracture-attention-guidance/data/
done

dirs_workstation7=(
    # dinov1_bs128_ep100_05-13_07:20:14
    # dinov1_bs128_ep100_no_hflip_05-13_02:09:20
    # dinov1_bs128_ep100_random_rotation_05-12_02:18:31
)

data_dir=~/fracture-attention-guidance/data
for dir in "${dirs_workstation7[@]}"; do
    echo copying dir $dir ....
    rsync -r --exclude *.jpg jvrielink@10.12.65.30:$data_dir/$dir ~/University/Master_thesis/fracture-attention-guidance/data
done
