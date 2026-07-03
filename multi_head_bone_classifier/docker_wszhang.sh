# docker run --gpus all -v $pwd -w /code -it pytorch-test
docker run \
-it \
--rm \
-v $(pwd):/code \
--gpus '"device=0"' \
--mount type=bind,src=/home/zhibin/data/AIML_rot_corrected,target=/AIML_rot_corrected \
--mount type=bind,src=/home/zhibin/data/classification_models,target=/classification_models \
--shm-size 64G \
jvrielink/pytorch