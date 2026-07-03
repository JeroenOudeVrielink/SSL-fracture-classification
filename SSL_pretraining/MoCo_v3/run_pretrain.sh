python main_moco.py \
--moco-m-cos \
--crop-min=.2 \
--gpu=0 \
--data="/jvrielink/AIML_rot_corrected" \
--annotations_file="annotations/img_paths.csv" \
-b=256 \
--epochs=400 \
--workers=14 \
--exp_name="moco_base_params_bs256_ep400" \
--output_dir="/data/moco_models"



