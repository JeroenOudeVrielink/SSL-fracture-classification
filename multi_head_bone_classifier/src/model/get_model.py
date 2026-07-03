import torch

from model.v1 import ModelV1
from model.v2 import ModelV2
from model.v3 import ModelV3


def get_model(hparams):
    body_loss_weights = None
    view_loss_weights = None
    if hparams.body_loss_weights_path is not None:
        body_loss_weights = torch.load(hparams.body_loss_weights_path)
    if hparams.view_loss_weights_path is not None:
        view_loss_weights = torch.load(hparams.view_loss_weights_path)

    if hparams.model_version == "v1":
        if hparams.checkpoint_load_path is not None:
            model = ModelV1.load_from_checkpoint(hparams.checkpoint_load_path)
            print("Loading model from checkpoint: ", hparams.checkpoint_load_path)
            return model
        return ModelV1(
            num_classes1=hparams.num_classes1,
            num_classes2=hparams.num_classes2,
            input_size=hparams.input_size,
            learning_rate=hparams.learning_rate,
            lr_start_factor=hparams.lr_start_factor,
            lr_end_factor=hparams.lr_end_factor,
            lr_total_iters=hparams.max_epochs,
            body_loss_weights=body_loss_weights,
            view_loss_weights=view_loss_weights,
            imgnet_pretrained=hparams.imgnet_pretrained,
            clip_p=hparams.clip_p,
            body_part_names_path=hparams.body_part_names_path,
            view_names_path=hparams.view_names_path,
            optimizer_type=hparams.optimizer_type,
        )
    elif hparams.model_version == "v2":
        if hparams.checkpoint_load_path is not None:
            model = ModelV2.load_from_checkpoint(hparams.checkpoint_load_path)
            return model
        return ModelV2(
            num_classes1=hparams.num_classes1,
            num_classes2=hparams.num_classes2,
            input_size=hparams.input_size,
            learning_rate=hparams.learning_rate,
            lr_start_factor=hparams.lr_start_factor,
            lr_end_factor=hparams.lr_end_factor,
            lr_total_iters=hparams.max_epochs,
            body_loss_weights=body_loss_weights,
            view_loss_weights=view_loss_weights,
            clip_p=hparams.clip_p,
            body_part_names_path=hparams.body_part_names_path,
            view_names_path=hparams.view_names_path,
        )
    elif hparams.model_version == "v3":
        if hparams.checkpoint_load_path is not None:
            model = ModelV3.load_from_checkpoint(hparams.checkpoint_load_path)
            return model
        return ModelV3(
            num_classes1=hparams.num_classes1,
            num_classes2=hparams.num_classes2,
            input_size=hparams.input_size,
            learning_rate=hparams.learning_rate,
            lr_start_factor=hparams.lr_start_factor,
            lr_end_factor=hparams.lr_end_factor,
            lr_total_iters=hparams.max_epochs,
            body_loss_weights=body_loss_weights,
            view_loss_weights=view_loss_weights,
            imgnet_pretrained=hparams.imgnet_pretrained,
            clip_p=hparams.clip_p,
            body_part_names_path=hparams.body_part_names_path,
            view_names_path=hparams.view_names_path,
        )

    else:
        raise ValueError(f"Model version {hparams.model_version} not supported")
