# docker run --gpus all -v $pwd -w /code -it pytorch-test
docker run \
-it \
--rm \
-v $(pwd):/code \
--mount type=bind,src=/home/jvrielink/AIML_half_size,target=/AIML_half_size \
--mount type=bind,src=/home/jvrielink/data/models,target=/models \
--mount type=bind,src=/home/jvrielink/data/results,target=/results \
--gpus '"device=1"' \
--shm-size 64G \
jvrielink/pytorch