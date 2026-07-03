# docker run --gpus all -v $pwd -w /code -it pytorch-test
docker run \
-it \
--rm \
-v $(pwd):/code \
--gpus '"device=0"' \
--mount type=bind,src=/home/zhibin,target=/jvrielink \
--mount type=bind,src=/home/zhibin/data/AIML_rot_corrected,target=/AIML_rot_corrected \
--mount type=bind,src=/home/zhibin/data/dinov1_models,target=/dinov1_models \
--shm-size 64G \
jvrielink/pytorch_dinov1