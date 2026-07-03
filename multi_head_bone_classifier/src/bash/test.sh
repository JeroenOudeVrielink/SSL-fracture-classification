cd ..

python train.py \
--run_name test \
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
--sample_weights_path /code/src/utils/weights/body_sample_weights.pt \
--body_loss_weights_path /code/src/utils/weights/body_loss_weights2.pt \
--view_loss_weights_path /code/src/utils/weights/view_loss_weights2.pt