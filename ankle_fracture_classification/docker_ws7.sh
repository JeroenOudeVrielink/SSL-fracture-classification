# docker run --gpus all -v $pwd -w /code -it pytorch-test
docker run \
-it \
--rm \
-v $(pwd):/code \
--gpus '"device=0"' \
--mount type=bind,src=/home/jvrielink/fracture,target=/mnt/sdb1/Data_remote/fracture/ \
--mount type=bind,src=/home/jvrielink/data_hdd/models,target=/models_hdd \
--mount type=bind,src=/home/jvrielink/data,target=/data \
--mount type=bind,src=/home/jvrielink/models,target=/models_local \
--shm-size 64G \
jvrielink/pytorch_ankle


# --mount type=bind,src=/home/jvrielink/data,target=/data \
