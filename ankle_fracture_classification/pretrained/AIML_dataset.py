import os
import pandas as pd
from torch.utils.data import Dataset
from PIL import Image
import torch


class AIMLDataset(Dataset):
    def __init__(
        self,
        annotations_file: str,
        data_path: str,
        transform=None,
        age_prediction=False,
    ):
        self.img_paths_labels = pd.read_pickle(
            os.path.join(data_path, annotations_file)
        )
        self.data_path = data_path
        self.transform = transform
        self.age_prediction = age_prediction

    def __len__(self):
        return len(self.img_paths_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.data_path, self.img_paths_labels.iloc[idx, 0])
        image = Image.open(img_path)

        gt_body_part = torch.Tensor(self.img_paths_labels.iloc[idx, 1])
        gt_view = torch.Tensor(self.img_paths_labels.iloc[idx, 2])

        if self.transform:
            image = self.transform(image)

        if self.age_prediction:
            gt_age = torch.Tensor([self.img_paths_labels.iloc[idx, 3]])
            return image, gt_body_part, gt_view, gt_age

        return image, gt_body_part, gt_view
