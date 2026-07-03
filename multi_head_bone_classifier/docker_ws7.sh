# docker run --gpus all -v $pwd -w /code -it pytorch-test
docker run \
-it \
--rm \
-v $(pwd):/code \
--mount type=bind,src=/home/jvrielink/data_hdd/AIML_rot_corrected,target=/AIML_rot_corrected \
--mount type=bind,src=/home/jvrielink/data_hdd/classification_models,target=/classification_models \
--gpus all \
--shm-size 64G \
jvrielink/pytorch