#!/bin/bash

target_dir=/home/jvrielink/fracture-attention-guidance/data/dinov1_bs128_ep1000_06-22_07:49:45  # Replace with the actual path
seed_dirs=("seed_1" "seed_2" "seed_3")
source_dirs=(/home/jvrielink/fracture-attention-guidance/data/dinov1_bs128_ep1000_ckpt20_400_06-15_01:20:58
/home/jvrielink/fracture-attention-guidance/data/dinov1_bs128_ep1000_ckpt420_760_06-20_07:07:11
/home/jvrielink/fracture-attention-guidance/data/dinov1_bs128_ep1000_ckpt780_880_06-22_07:47:44)  # Add your source directories

for source_dir in "${source_dirs[@]}"; do
    for seed_dir in "${seed_dirs[@]}"; do
        source_path="$source_dir/$seed_dir/"  # Use wildcard to get all content
        target_path="$target_dir"

        # echo "Copying from $source_path to $target_path"
        # cp -r "$source_path" "$target_path"

        # Check if the source directory exists
        if [[ -d "$source_path" ]]; then
            echo "Copying from $source_path to $target_path"
            cp -r "$source_path" "$target_path"
        else
            echo "Warning: $source_path does not exist."
        fi
    done
done

echo "Copy operation complete!"

