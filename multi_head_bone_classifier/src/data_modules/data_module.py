from pytorch_lightning import LightningDataModule
from pytorch_lightning.utilities.types import EVAL_DATALOADERS
from torch.utils.data import DataLoader, WeightedRandomSampler
from data_modules.dataset import AIMLDataset
from data_modules.predict_dataset import PredictAIMLDataset
import torch


class AIMLDataModule(LightningDataModule):
    def __init__(
        self,
        data_path: str = "path/to/dir",
        annotation_path: str = "path/to/annotation",
        train_batch_size: int = 32,
        eval_batch_size: int = 32,
        train_transform=None,
        val_transform=None,
        test_transform=None,
        num_workers: int = 0,
        sample_weights_path: str = None,
        age_prediction: bool = False,
    ):
        super().__init__()
        self.data_path = data_path
        self.annotation_path = annotation_path
        self.train_batch_size = train_batch_size
        self.eval_batch_size = eval_batch_size
        self.train_transform = train_transform
        self.val_transform = val_transform
        self.test_transform = test_transform
        self.num_workers = num_workers
        self.age_prediction = age_prediction
        self.sampler = None
        self.shullfe = True
        if sample_weights_path is not None:
            sample_weights = torch.load(sample_weights_path)
            self.sampler = WeightedRandomSampler(
                weights=sample_weights,
                num_samples=sample_weights.shape[0],
                replacement=True,
            )
            self.shullfe = False

    def setup(self, stage: str):
        self.train_data = AIMLDataset(
            self.annotation_path + "/train.pkl",
            self.data_path,
            transform=self.train_transform,
            age_prediction=self.age_prediction,
        )
        self.val_data = AIMLDataset(
            self.annotation_path + "/val.pkl",
            self.data_path,
            transform=self.val_transform,
            age_prediction=self.age_prediction,
        )
        self.test_data = AIMLDataset(
            self.annotation_path + "/test.pkl",
            self.data_path,
            self.test_transform,
            age_prediction=self.age_prediction,
        )

        self.predict_data = PredictAIMLDataset(
            self.annotation_path + "/unknown.pkl",
            self.data_path,
            self.test_transform,
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_data,
            batch_size=self.train_batch_size,
            shuffle=self.shullfe,
            num_workers=self.num_workers,
            sampler=self.sampler,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_data,
            batch_size=self.eval_batch_size,
            num_workers=self.num_workers,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_data,
            batch_size=self.eval_batch_size,
            num_workers=self.num_workers,
        )

    def predict_dataloader(self):
        return DataLoader(
            self.predict_data,
            batch_size=self.eval_batch_size,
            num_workers=self.num_workers,
        )
