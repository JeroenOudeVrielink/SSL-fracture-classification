FROM pytorch/pytorch:latest

WORKDIR /code

COPY ./requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

ENV WANDB_API_KEY="e2acc047b3e4f05edb42a802cb361ef41eacb607"

COPY . ./code

# CMD ["sh", "src/utils/export_wandb_api_key.sh"]

