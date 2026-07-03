
python main_vicregl.py \
--fp16 \
--exp-dir /data/vicregl_models \
--arch resnet50 \
--epochs 400 \
--batch-size 128 \
--optimizer lars \
--base-lr 0.3 \
--weight-decay 1e-06 \
--size-crops 224 \
--num-crops 2 \
--min_scale_crops 0.08 \
--max_scale_crops 1.0 \
--alpha 0.75 \
--num-workers 10 \
--exp_name "vicregl_base_params_bs128" \
--data_path /data/AIML_rot_corrected \
--log-tensors-interval 100 \
--checkpoint-freq 10