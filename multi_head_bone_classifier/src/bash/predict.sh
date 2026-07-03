cd ..

python predict.py \
--run_name Gen3_v1_imgnet_pretrained_color_jitter_rotation_lr_schedule_prediction_square_resize \
--checkpoint_load_path /classification_models/Gen3_v1_imgnet_pretrained_color_jitter_rotation_lr_schedule_04-05_22:42:44/epoch99_summedf1_1.64.ckpt \
--model_version "v1" \
--eval_batch_size 512 \
--num_workers 16