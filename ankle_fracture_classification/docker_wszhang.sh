# docker run --gpus all -v $pwd -w /code -it pytorch-test
docker run \
-it \
--rm \
-v $(pwd):/code \
--gpus '"device=0"' \
--mount type=bind,src=/home/zhibin/fracture,target=/mnt/sdb1/Data_remote/fracture/ \
--mount type=bind,src=/home/zhibin/data,target=/data \
--mount type=bind,src=/home/zhibin,target=/zhibin_home \
--shm-size 64G \
jvrielink/pytorch_ankle