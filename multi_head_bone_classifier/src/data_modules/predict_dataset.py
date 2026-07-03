import os
import pandas as pd
from torch.utils.data import Dataset
from PIL import Image
import torch


class PredictAIMLDataset(Dataset):
    def __init__(
        self,
        annotations_file: str,
        data_path: str,
        transform=None,
    ):
        self.img_paths_labels = pd.read_pickle(
            os.path.join(data_path, annotations_file)
        )
        self.data_path = data_path
        self.transform = transform

    def __len__(self):
        return len(self.img_paths_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.data_path, self.img_paths_labels.iloc[idx])
        image = Image.open(img_path)

        if self.transform:
            image = self.transform(image)

        return image
