import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import LinearLR
import pytorch_lightning as pl
from utils.metrics_calculator import MetricsCalculator
from model.clipped_cce import ClippedCrossEntropyLoss


class ModelV3(pl.LightningModule):
    def __init__(
        self,
        num_classes1: int = 24,
        num_classes2: int = 4,
        input_size: tuple = (3, 244, 244),
        learning_rate: float = 1e-3,
        lr_start_factor: float = 1,
        lr_end_factor: float = 0.1,
        lr_total_iters: int = 10,
        body_loss_weights=None,
        view_loss_weights=None,
        imgnet_pretrained=False,
        clip_p=1.0,
        body_part_names_path: str = None,
        view_names_path: str = None,
    ):
        super(ModelV3, self).__init__()

        self.num_classes1 = num_classes1
        self.num_classes2 = num_classes2
        self.input_size = input_size
        self.learning_rate = learning_rate
        self.lr_start_factor = lr_start_factor
        self.lr_end_factor = lr_end_factor
        self.lr_total_iters = lr_total_iters
        self.body_loss_weights = body_loss_weights
        self.view_loss_weights = view_loss_weights
        self.imgnet_pretrained = imgnet_pretrained
        self.clip_p = clip_p
        self.body_part_names_path = body_part_names_path
        self.view_names_path = view_names_path

        # Load a pre-trained ResNet-50 model as the encoder
        weights = None
        if imgnet_pretrained:
            print("Using imagenet pretrained weights")
            weights = ResNet50_Weights.IMAGENET1K_V1
        else:
            print("Using random weights; NOT pretrained!")
        resnet = resnet50(weights=weights)

        # Remove the final classification layer (head) + pooling layer
        self.encoder = nn.Sequential(*list(resnet.children())[:-2])

        self.pool_and_flatten = nn.Sequential(
            nn.AdaptiveAvgPool2d(output_size=(1, 1)), nn.Flatten()
        )

        # Linear layers for deconders 1 and 2
        self.decoder1 = nn.Sequential(
            nn.Linear(2048, num_classes1),
        )
        self.decoder2 = nn.Sequential(
            nn.Linear(2048, num_classes2),
        )
        self.decoder3 = nn.Sequential(
            nn.Linear(2048, 1),
        )

        # If using clipped cross entropy loss
        if clip_p < 1.0:
            print("Using clipped cross entropy loss")
            self.body_criterion = ClippedCrossEntropyLoss(
                clip_p=clip_p, weight=body_loss_weights
            )
            self.view_criterion = ClippedCrossEntropyLoss(
                clip_p=clip_p, weight=view_loss_weights
            )
        else:
            print("Using normal cross entropy loss")
            self.body_criterion = nn.CrossEntropyLoss(weight=body_loss_weights)
            self.view_criterion = nn.CrossEntropyLoss(weight=view_loss_weights)

        self.age_criterion = nn.L1Loss()

        self.metrics = MetricsCalculator(
            num_classes1, num_classes2, body_part_names_path, view_names_path
        )
        self.save_hyperparameters()

    def forward(self, x):
        # Encode with resnet
        x = self.encoder(x)
        # Pool and flatten
        x = self.pool_and_flatten(x)
        # decoder 1
        out1 = self.decoder1(x)
        # decoder 2
        out2 = self.decoder2(x)
        # decoder 3
        out3 = self.decoder3(x)
        return out1, out2, out3

    def step(self, batch):
        x, y_body_part, y_view, y_age = batch
        logits1, logits2, logits3 = self(x)
        # convert one-hot-encoding to integer labels
        y1 = torch.argmax(y_body_part, dim=-1)
        y2 = torch.argmax(y_view, dim=-1)
        # Compute loss for encoder 1
        loss1 = self.body_criterion(logits1, y1)
        # Compute loss for encoder 2
        loss2 = self.view_criterion(logits2, y2)
        # Comptue loss for decoder 3
        loss3 = self.age_criterion(logits3, y_age)
        # Total loss
        loss = loss1 + loss2 + loss3
        # get prediction
        y_hat1 = torch.argmax(logits1, dim=-1)
        y_hat2 = torch.argmax(logits2, dim=-1)

        y_hat3, y3 = logits3, y_age
        return y_hat1, y1, y_hat2, y2, y_hat3, y3, loss1, loss2, loss3, loss

    def training_step(self, batch, batch_idx):
        y_hat1, y1, y_hat2, y2, y_hat3, y3, loss1, loss2, loss3, loss = self.step(batch)
        losses = {
            "loss": loss,
            "loss_body": loss1,
            "loss_view": loss2,
            "loss_age": loss3,
        }
        log = self.metrics.get_dict(y_hat1, y1, y_hat2, y2, losses, prefix="train/")
        self.log_dict(log)
        return loss

    def validation_step(self, batch, batch_idx):
        y_hat1, y1, y_hat2, y2, y_hat3, y3, loss1, loss2, loss3, loss = self.step(batch)
        losses = {
            "loss": loss,
            "loss_body": loss1,
            "loss_view": loss2,
            "loss_age": loss3,
        }
        log = self.metrics.get_dict(y_hat1, y1, y_hat2, y2, losses, prefix="val/")
        class_wise_log = self.metrics.get_class_wise_dict(y_hat1, y1, y_hat2, y2)
        log.update(class_wise_log)
        self.log_dict(log)
        return loss

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)
        if self.lr_end_factor is not None:
            scheduler = LinearLR(
                optimizer,
                start_factor=self.lr_start_factor,
                end_factor=self.lr_end_factor,
                total_iters=self.lr_total_iters,
            )
            return {"optimizer": optimizer, "lr_scheduler": scheduler}
        return optimizer
