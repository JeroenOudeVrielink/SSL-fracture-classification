import os
from datetime import datetime

import torch

from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import WandbLogger

from data_modules.data_module import AIMLDataModule
from model.get_model import get_model

from parse_args import get_args
from preprocessing.transform import get_train_transform, get_val_transform
import json
from pathlib import Path

# Make sure to login to wandb before running this script
# Run: wandb login

# Added datetime to name to avoid conflicts
time = datetime.now().strftime("%m-%d_%H:%M:%S")

# Set torch float precision for proper use GPU
torch.set_float32_matmul_precision("high")


def train(hparams):
    run_name = f"{hparams.run_name}_{time}"
    hparams.checkpoint_save_path = os.path.join(hparams.checkpoint_save_path, run_name)

    Path(hparams.checkpoint_save_path).mkdir(parents=True, exist_ok=True)
    with open(os.path.join(hparams.checkpoint_save_path, "config.json"), "w") as f:
        json.dump(vars(hparams), f, indent=4)

    # Create wandb logger
    wandb_logger = WandbLogger(
        name=run_name,
        project="multitask-bone-classification",
        entity="jeroenov98",
        save_dir="logs/",
        log_model=True,
        config=hparams,
    )

    # Expands paths
    hparams.data_path = os.path.expanduser(hparams.data_path)
    if hparams.checkpoint_load_path:
        hparams.checkpoint_load_path = os.path.expanduser(hparams.checkpoint_load_path)
    if hparams.checkpoint_save_path:
        hparams.checkpoint_save_path = os.path.expanduser(hparams.checkpoint_save_path)

    age_prediction = True if hparams.model_version == "v3" else False
    # Create data module
    data_module = AIMLDataModule(
        data_path=hparams.data_path,
        annotation_path=hparams.annotations_path,
        train_batch_size=hparams.train_batch_size,
        eval_batch_size=hparams.eval_batch_size,
        train_transform=get_train_transform(hparams),
        val_transform=get_val_transform(hparams),
        test_transform=get_val_transform(hparams),
        num_workers=hparams.num_workers,
        sample_weights_path=hparams.sample_weights_path,
        age_prediction=age_prediction,
    )

    # Create model
    model = get_model(hparams)

    if hparams.metric_in_filename:
        filename = "epoch{epoch:02d}_summedf1_{val/summed_f1:.2f}"
    else:
        filename = "epoch{epoch:02d}"

    # Create checkpoint callback
    checkpoint_callback = ModelCheckpoint(
        monitor=hparams.monitor,
        mode="max",
        dirpath=hparams.checkpoint_save_path,
        filename=filename,
        save_last=True,
        save_top_k=hparams.save_top_k,
        auto_insert_metric_name=False,
        save_on_train_epoch_end=True,
    )
    lr_monitor = LearningRateMonitor(logging_interval="epoch")

    callbacks = [checkpoint_callback, lr_monitor]

    # Create trainer
    trainer = Trainer(
        accelerator="auto",
        devices="auto",
        max_epochs=hparams.max_epochs,
        logger=wandb_logger,
        log_every_n_steps=hparams.log_every_n_steps,
        val_check_interval=1.0,
        limit_val_batches=hparams.limit_val_batches,
        limit_train_batches=hparams.limit_train_batches,
        callbacks=callbacks,
    )

    # Train Model
    trainer.fit(model, data_module, ckpt_path=hparams.checkpoint_load_path)


if __name__ == "__main__":
    hparams = get_args()
    train(hparams)
