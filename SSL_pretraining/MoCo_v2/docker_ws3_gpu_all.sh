# docker run --gpus all -v $pwd -w /code -it pytorch-test
docker run \
-it \
--rm \
-v $(pwd):/code \
--gpus all \
--mount type=bind,src=/home/jvrielink,target=/jvrielink \
--mount type=bind,src=/home/jvrielink/data,target=/data \
--shm-size 64G \
jvrielink/pytorch_moco