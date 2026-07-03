cd ..

python train.py \
--run_name Gen3_v1_imgnet_pretrained_color_jitter_rotation_lr_schedule_clip06 \
--model_version "v1" \
--max_epochs 100 \
--train_batch_size 128 \
--eval_batch_size 128 \
--num_workers 8 \
--learning_rate 1e-4 \
--imgnet_pretrained True \
--aug_color_jitter True \
--aug_rotation True \
--lr_end_factor 0.01 \
--save_top_k -1 \
--clip_p 0.6

sleep 30s

python train.py \
--run_name Gen3_v1_imgnet_pretrained_color_jitter_rotation_lr_schedule_clip07 \
--model_version "v1" \
--max_epochs 100 \
--train_batch_size 128 \
--eval_batch_size 128 \
--num_workers 8 \
--learning_rate 1e-4 \
--imgnet_pretrained True \
--aug_color_jitter True \
--aug_rotation True \
--lr_end_factor 0.01 \
--save_top_k -1 \
--clip_p 0.7

sleep 30s

python train.py \
--run_name Gen3_v1_imgnet_pretrained_color_jitter_rotation_lr_schedule_clip08 \
--model_version "v1" \
--max_epochs 100 \
--train_batch_size 128 \
--eval_batch_size 128 \
--num_workers 8 \
--learning_rate 1e-4 \
--imgnet_pretrained True \
--aug_color_jitter True \
--aug_rotation True \
--lr_end_factor 0.01 \
--save_top_k -1 \
--clip_p 0.8

sleep 30s

python train.py \
--run_name Gen3_v1_imgnet_pretrained_color_jitter_rotation_lr_schedule_clip09 \
--model_version "v1" \
--max_epochs 100 \
--train_batch_size 128 \
--eval_batch_size 128 \
--num_workers 8 \
--learning_rate 1e-4 \
--imgnet_pretrained True \
--aug_color_jitter True \
--aug_rotation True \
--lr_end_factor 0.01 \
--save_top_k -1 \
--clip_p 0.9


sleep 30s

python train.py \
--run_name Gen3_v2_imgnet_pretrained_color_jitter_rotation_lr_schedule \
--model_version "v2" \
--max_epochs 100 \
--train_batch_size 128 \
--eval_batch_size 128 \
--num_workers 8 \
--learning_rate 1e-4 \
--imgnet_pretrained True \
--aug_color_jitter True \
--aug_rotation True \
--lr_end_factor 0.01 \
--save_top_k -1

sleep 30s

python train.py \
--run_name Gen3_v3_imgnet_pretrained_color_jitter_rotation_lr_schedule \
--model_version "v3" \
--max_epochs 100 \
--train_batch_size 128 \
--eval_batch_size 128 \
--num_workers 8 \
--learning_rate 1e-4 \
--imgnet_pretrained True \
--aug_color_jitter True \
--aug_rotation True \
--lr_end_factor 0.01 \
--save_top_k -1