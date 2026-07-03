#!/usr/bin/env bash

sleep 6h

rsync -r -P --rsh=ssh zhibin@10.90.184.232:~/data/classification_models/Gen3_v1_imgnet_pretrained_color_jitter_rotation_lr_schedule_04-11_02:00:18 /home/jvrielink/data_hdd/classification_models

