#!/usr/bin/env bash


./run_pretrain_out_dim4096.sh

sleep 30s

./run_pretrain_disable_centering.sh

sleep 30s

./run_pretrain_out_dim8192.sh

