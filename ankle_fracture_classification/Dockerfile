FROM pytorch/pytorch:2.0.0-cuda11.7-cudnn8-runtime

WORKDIR /code

RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y

COPY ./requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# ENV WANDB_API_KEY="e2acc047b3e4f05edb42a802cb361ef41eacb607"

COPY . ./code

# RUN git config user.email "jeroenov98@gmail.com"
# RUN git config user.name "Jeroen Oude Vrielink"

ENV PYTHONPATH "${PYTHONPATH}:/code"
ENV PYTHONPATH "${PYTHONPATH}:/code/modelling"
ENV PYTHONPATH "${PYTHONPATH}:/code/pretrained"

# CMD ["sh", "src/utils/export_wandb_api_key.sh"]

