# docker run --gpus all -v $pwd -w /code -it pytorch-test
docker run \
-it \
--rm \
-v $(pwd):/code \
--gpus '"device=1"' \
--mount type=bind,src=/home/zhibin/AIML_half_size,target=/AIML_half_size \
--mount type=bind,src=/home/zhibin/data/models,target=/models \
--shm-size 64G \
jvrielink/pytorch