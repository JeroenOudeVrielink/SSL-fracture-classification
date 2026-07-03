#!/usr/bin/env bash

# Num workers was 14 (optimal) but caused OOM errors for RAM on ws7

./run_pretrain_random_rotation.sh

./run_pretrain_smoothing_instead_blurring.sh




