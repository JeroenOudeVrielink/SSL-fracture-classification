
python main_vicregl.py \
--fp16 \
--exp-dir /data/vicregl_models \
--arch resnet50 \
--epochs 301 \
--batch-size 128 \
--optimizer lars \
--base-lr 0.3 \
--weight-decay 1e-06 \
--size-crops 224 \
--num-crops 2 \
--min_scale_crops 0.08 \
--max_scale_crops 1.0 \
--alpha 1.0 \
--num-workers 14 \
--exp_name "vicregl_bs128_alpha1" \
--data_path /data/AIML_rot_corrected \
--log-tensors-interval 100 \
--checkpoint-freq 20