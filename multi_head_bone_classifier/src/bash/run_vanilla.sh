cd ..

python train.py \
--run_name Gen3_v1_vanilla \
--model_version "v1" \
--max_epochs 100 \
--train_batch_size 128 \
--eval_batch_size 128 \
--num_workers 8 \
--learning_rate 1e-4