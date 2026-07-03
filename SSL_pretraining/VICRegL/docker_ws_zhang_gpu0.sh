# docker run --gpus all -v $pwd -w /code -it pytorch-test
docker run \
-it \
--rm \
-v $(pwd):/code \
--gpus '"device=0"' \
--mount type=bind,src=/home/zhibin,target=/zhibin \
--mount type=bind,src=/home/zhibin/data,target=/data \
--shm-size 64G \
jvrielink/pytorch_vicregl