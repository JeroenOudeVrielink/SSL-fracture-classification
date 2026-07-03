cd ..

python train.py \
--run_name v1_imgnet_pretrained_color_jitter_rotation_lr_schedule \
--model_version "v1" \
--max_epochs 3 \
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