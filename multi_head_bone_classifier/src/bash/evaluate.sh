cd ..

python evaluate.py \
--run_name Gen3_v1_sgd_epoch_97 \
--checkpoint_load_path /classification_models/Gen3_v1_sgd_04-10_02:31:54/epoch97_summedf1_1.52.ckpt \
--model_version "v1" \
--eval_batch_size 128 \
--num_workers 8