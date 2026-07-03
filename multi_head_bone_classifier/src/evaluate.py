import os
from datetime import datetime

import torch

from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger
from pathlib import Path

from data_modules.data_module import AIMLDataModule
from model.get_model import get_model

from parse_args import get_args
from preprocessing.transform import get_val_transform, get_test_transform
from utils.save_model_output import save_model_output
from utils.compute_evaluation import compute_and_save_metrics
import json
import shutil

# Make sure to login to wandb before running this script
# Run: wandb login

# Added datetime to name to avoid conflicts
time = datetime.now().strftime("%m-%d_%H:%M:%S")

# Set torch float precision for proper use GPU
torch.set_float32_matmul_precision("high")


def train(hparams):
    run_name = f"{hparams.run_name}_{time}"

    # Create save directory
    save_dir = Path(hparams.results_save_path) / run_name
    save_dir.mkdir(parents=True, exist_ok=True)

    config_file = os.path.dirname(hparams.checkpoint_load_path) + "/config.json"
    shutil.copy(config_file, save_dir)

    # Expands paths
    hparams.data_path = os.path.expanduser(hparams.data_path)
    if hparams.checkpoint_load_path:
        hparams.checkpoint_load_path = os.path.expanduser(hparams.checkpoint_load_path)
    if hparams.checkpoint_save_path:
        hparams.checkpoint_save_path = os.path.expanduser(hparams.checkpoint_save_path)

    # Create data module
    data_module = AIMLDataModule(
        data_path=hparams.data_path,
        annotation_path=hparams.annotations_path,
        eval_batch_size=hparams.eval_batch_size,
        val_transform=get_val_transform(hparams),
        test_transform=get_test_transform(hparams),
        num_workers=hparams.num_workers,
    )

    # Create model
    model = get_model(hparams)

    config = model.hparams
    config = vars(config)

    # Create trainer
    trainer = Trainer(
        accelerator="auto",
        devices="auto",
        limit_predict_batches=hparams.limit_predict_batches,
    )

    if hparams.predict:
        # Get output
        out = trainer.predict(model, data_module)
        # Save and transform the output
        body_preds, view_preds = save_model_output(out, save_dir)
        true_labels_path = (
            Path(hparams.data_path) / hparams.annotations_path / "test.pkl"
        )

        with open(hparams.body_part_names_path, "r") as json_file:
            body_part_names = json.load(json_file)
        with open(hparams.view_names_path, "r") as json_file:
            view_names = json.load(json_file)

        # Compute metrics and save them
        compute_and_save_metrics(
            true_labels_path,
            body_preds,
            view_preds,
            hparams.checkpoint_load_path,
            save_dir,
            body_part_names,
            view_names,
        )

    else:
        trainer.test(model, data_module)


if __name__ == "__main__":
    hparams = get_args()
    train(hparams)
