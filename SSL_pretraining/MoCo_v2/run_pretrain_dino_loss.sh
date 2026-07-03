python main_moco.py \
--mlp \
--moco-t 0.2 \
--aug-plus \
--cos \
-a resnet50 \
--lr 0.03 \
--batch-size 512 \
--dist-url 'tcp://localhost:10001' \
--multiprocessing-distributed \
--world-size 1 \
--rank 0 \
--data="/jvrielink/AIML_rot_corrected" \
--epochs=301 \
--workers=12 \
--exp_name="mocov2_with_dino_loss_bs512_ep300" \
--output_dir="/data/moco_models" \
--moco-m 0.996



