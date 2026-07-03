import torch
import torch.utils.data as data
import pandas as pd
from torchvision import transforms as T
from torch.nn.functional import interpolate
import numpy as np
from PIL import Image
import os
from transform_utils import (
    random_crop,
    random_horizontal_flip,
    random_rotation,
    border_padding,
)
import cv2


class FractureDataset(data.Dataset):
    def __init__(
        self,
        root,
        csv_path,
        config,
        target_translator=None,
        set=None,
        use_random_crop=None,
        transform=None,
        no_padding=False,
    ):
        super(FractureDataset, self).__init__()

        self.root = root

        self.use_random_crop = use_random_crop
        self.transform = transform

        input_size = config["args"]["input_size"]
        self.resize_size = (input_size, input_size)

        self.df = pd.read_csv(csv_path, index_col=0)
        self.config = config

        for k in self.df:
            if k in self.config["binary_cls_cols"] or k in self.config["cls_cols_dict"]:
                variations = list(self.df[k].loc[self.df[k].notna()].unique())

                num_variations = len(variations)
                # if any(np.isnan([v for v in variations if type(v) is not str])):
                #     num_variations -= 1

                if num_variations < 100:
                    print(
                        "[{}] [{} variations]: {}".format(k, num_variations, variations)
                    )
                else:
                    print("[{}] [{} variations]".format(k, num_variations))

        self.ignore_value = -1e10
        # self.df.loc[self.df['smoking_status'] == -3, 'smoking_status'] = self.ignore_value
        self.df.fillna(self.ignore_value, inplace=True)
        self._class2index(target_translator)
        self.scribble_type = self.config["args"]["scribble_type"]

        selected = None
        for s in set:
            subset = s
            subset_selection = self.df["set"] == subset
            if selected is None:
                selected = subset_selection
            else:
                selected = selected | subset_selection

        self.df = self.df.loc[selected]

        print("Dataset size: {}".format(len(self.df)))

        self.df.reset_index(drop=True)
        print("scribble type: {}".format(self.scribble_type))
        self.no_padding = no_padding

    def _class2index(self, target_translator):
        compute_translator = False
        if target_translator is not None:
            self.target_translator = target_translator
        else:
            self.target_translator = {
                "binary_cls_cols": dict(),
                "cls_cols_dict": dict(),
            }
            compute_translator = True

        for t in self.config["binary_cls_cols"]:
            if compute_translator:
                choices = [c for c in self.df[t].unique() if c != self.ignore_value]
                choices = np.sort(choices)
                indices = np.arange(len(choices))
                translator = dict(zip(choices, indices))
                self.target_translator[t] = translator

            for c in self.target_translator[t]:
                self.df.loc[self.df[t] == c, t] = self.target_translator[t][c]

        for t in self.config["cls_cols_dict"]:
            if compute_translator:
                choices = [c for c in self.df[t].unique() if c != self.ignore_value]
                choices = np.sort(choices)
                indices = np.arange(len(choices))
                translator = dict(zip(choices, indices))
                self.target_translator[t] = translator

            for c in self.target_translator[t]:
                self.df.loc[self.df[t] == c, t] = self.target_translator[t][c]

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: Tuple (image, target). target is the object returned by ``coco.loadAnns``.
        """

        random_codes = np.random.rand(6)

        info = self.df.iloc[index]
        image_path = info["image_path"]
        subset = info["set"]
        try:
            path = os.path.join(self.root, image_path)
        except:
            raise ValueError("{} {}".format(self.root, image_path))

        pil_image = Image.open(path).convert("RGB")

        source_size = pil_image.size

        if self.use_random_crop:
            pil_image = random_crop(
                pil_image, random_codes=random_codes[:4], crop_ratio_min=0.8
            )
            pil_image = random_horizontal_flip(pil_image, random_code=random_codes[4])
            pil_image = random_rotation(
                pil_image, random_code=random_codes[5], max_angle=30
            )

        if self.no_padding:
            new_image = pil_image
        else:
            new_image = border_padding(pil_image)

        if self.transform is not None:
            source_image = self.transform(new_image)
            image = source_image

        targets = list()
        targets.extend(
            [
                torch.tensor([info[c]], dtype=torch.float)
                for c in self.config["regression_cols"]
            ]
        )
        targets.extend(
            [
                torch.tensor([info[c]], dtype=torch.float)
                for c in self.config["binary_cls_cols"]
            ]
        )
        targets.extend([self.onehot(c, info[c]) for c in self.config["cls_cols_dict"]])
        targets = torch.cat(targets, 0)

        seg_targets = list()
        seg_targets.extend(
            [
                self.load_seg_mask(info, c, list(source_size)[::-1], random_codes)
                for c in self.config["seg_cols_dict"]
            ]
        )
        if len(seg_targets) == 0:
            seg_targets = torch.zeros([0] + list(self.resize_size), dtype=torch.float)
        else:
            seg_targets = torch.cat(seg_targets, 0)
        return image, targets, seg_targets, source_image

    def load_seg_mask(self, info, c, source_size, random_codes):
        num_channels = self.config["seg_cols_dict"][c]
        mask_size = source_size + [num_channels]
        # num_elements = np.prod(mask_size)

        if info[c] == self.ignore_value:
            if info["fibula"] == self.ignore_value:
                # np_mask = np.ones(mask_size, dtype=np.float) * self.ignore_value
                mask = (
                    torch.ones(
                        [num_channels] + list(self.resize_size), dtype=torch.float
                    )
                    * self.ignore_value
                )
                _c = None
            else:
                _c = "fibula"
        else:
            _c = c

        if _c:
            if self.scribble_type == "scribble":
                _c += "_scribble"

            path = os.path.join(self.root, info[_c])
            np_mask = cv2.imread(path)
            np_mask = np_mask[:, :, 0:1].astype(float) / 255.0
            if self.scribble_type == "bbox":
                bbox_coords = cv2.boundingRect(np_mask.astype(np.uint8))
                start_x, start_y, len_x, len_y = bbox_coords
                np_mask[start_y : start_y + len_y, start_x : start_x + len_x] = 1
            elif self.scribble_type == "seg" and self.scribble_type == "scribble":
                pass
            np_mask[np.where(np_mask == 0)] = self.ignore_value

            if self.use_random_crop:
                np_mask = random_crop(
                    np_mask, random_codes=random_codes[:4], crop_ratio_min=0.8
                )
                np_mask = random_horizontal_flip(np_mask, random_code=random_codes[4])
                np_mask = random_rotation(
                    np_mask,
                    random_code=random_codes[5],
                    max_angle=30,
                    border_value=self.ignore_value,
                )
            np_padded_mask = border_padding(np_mask, padding_value=self.ignore_value)

            mask = torch.tensor(np_padded_mask, dtype=torch.float)
            mask = mask.permute(2, 0, 1)

            mask = interpolate(
                mask.view([1] + list(mask.shape)), size=self.resize_size, mode="nearest"
            )
            mask = mask.repeat(1, 4, 1, 1)
            mask = mask.view(list(mask.shape)[1:])
        return mask

    def onehot(self, c, v):
        size = self.config["cls_cols_dict"][c]
        output = torch.zeros(size, dtype=torch.float)
        v = int(v)
        if v == self.ignore_value:
            output += self.ignore_value
        else:
            output[v] = 1
        return output


def collate_fn(data):
    """Creates mini-batch tensors from the list of tuples (image, caption).

    We should build custom collate_fn rather than using default collate_fn,
    because merging caption (including padding) is not supported in default.

    Args:
        data: list of tuple (image, caption).
            - image: torch tensor of shape (3, 256, 256).
            - caption: torch tensor of shape (?); variable length.

    Returns:
        images: torch tensor of shape (batch_size, 3, 256, 256).
        targets: torch tensor of shape (batch_size, padded_length).
        lengths: list; valid length for each padded caption.
    """
    # Sort a data list by caption length (descending order).
    images, targets, seg_targets, source_images = zip(*data)

    # Merge images (from tuple of 3D tensor to 4D tensor).
    images = torch.stack(images, 0)
    source_images = torch.stack(source_images, 0)
    targets = torch.stack(targets, 0)
    seg_targets = torch.stack(seg_targets, 0)

    return images, targets, seg_targets, source_images


def get_loader_from_dataset(
    data_root,
    csv_path,
    config,
    target_translator,
    batch_size,
    transform,
    set,
    use_random_crop,
    shuffle,
    num_workers,
    drop_last,
    no_padding=False,
):
    dataset = FractureDataset(
        root=data_root,
        csv_path=csv_path,
        config=config,
        target_translator=target_translator,
        set=set,
        use_random_crop=use_random_crop,
        transform=transform,
        no_padding=no_padding,
    )

    # Data loader for wsi dataset
    # This will return (images, captions, lengths) for each iteration.
    # images: a tensor of shape (batch_size, 3, 224, 224).
    # captions: a tensor of shape (batch_size, padded_length)
    # lengths: a list indicating valid length for each caption. length is (batch_size).
    data_loader = torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        drop_last=drop_last,
        collate_fn=collate_fn,
    )

    return data_loader
