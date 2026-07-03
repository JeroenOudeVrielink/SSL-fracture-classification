from torchvision import transforms
from PIL import Image
import numpy as np
import cv2
import utils
from torch import nn


# class DataAugmentationDINO(object):
#     def __init__(self, global_crops_scale, local_crops_scale, local_crops_number):
#         flip_and_color_jitter = transforms.Compose(
#             [
#                 transforms.RandomHorizontalFlip(p=0.5),
#                 transforms.RandomApply(
#                     [
#                         transforms.ColorJitter(
#                             brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1
#                         )
#                     ],
#                     p=0.8,
#                 ),
#                 transforms.RandomGrayscale(p=0.2),
#             ]
#         )
#         normalize = transforms.Compose(
#             [
#                 transforms.ToTensor(),
#                 transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
#             ]
#         )

#         # first global crop
#         self.global_transfo1 = transforms.Compose(
#             [
#                 transforms.RandomResizedCrop(
#                     224, scale=global_crops_scale, interpolation=Image.BICUBIC
#                 ),
#                 flip_and_color_jitter,
#                 utils.GaussianBlur(1.0),
#                 normalize,
#             ]
#         )
#         # second global crop
#         self.global_transfo2 = transforms.Compose(
#             [
#                 transforms.RandomResizedCrop(
#                     224, scale=global_crops_scale, interpolation=Image.BICUBIC
#                 ),
#                 flip_and_color_jitter,
#                 utils.GaussianBlur(0.1),
#                 utils.Solarization(0.2),
#                 normalize,
#             ]
#         )
#         # transformation for the local small crops
#         self.local_crops_number = local_crops_number
#         self.local_transfo = transforms.Compose(
#             [
#                 transforms.RandomResizedCrop(
#                     96, scale=local_crops_scale, interpolation=Image.BICUBIC
#                 ),
#                 flip_and_color_jitter,
#                 utils.GaussianBlur(p=0.5),
#                 normalize,
#             ]
#         )


class EdgePreservingFilter(nn.Module):
    def __init__(
        self,
        sigma_s=20,
        sigma_r=0.075,
    ):
        super(EdgePreservingFilter, self).__init__()
        self.sigma_s = sigma_s
        self.sigma_r = sigma_r

    def forward(self, pil_img):
        x = np.array(pil_img)
        smooth = cv2.edgePreservingFilter(
            x, flags=1, sigma_s=self.sigma_s, sigma_r=self.sigma_r
        )
        smooth = Image.fromarray(smooth)
        return smooth


class CustomDataAugmentationDINO(object):
    def __init__(
        self,
        global_crops_scale,
        local_crops_scale,
        local_crops_number,
        global_crop_size=224,
        local_crop_size=96,
        use_dino_augmentation=True,
        use_edge_preserving_filter=False,
        p_horizontal_flip=0.5,
        p_color_jitter=0.8,
        p_solarization=0.2,
        disable_gaussian_blur=False,
        p_random_rotation=0,
    ):
        flip_and_color_jitter = transforms.Compose(
            [
                transforms.RandomHorizontalFlip(p=p_horizontal_flip),
                transforms.RandomApply(
                    [
                        transforms.ColorJitter(
                            brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1
                        )
                    ],
                    p=p_color_jitter,
                ),
                transforms.RandomApply(
                    [
                        transforms.RandomRotation(
                            degrees=20,
                            interpolation=transforms.InterpolationMode.BILINEAR,
                        )
                    ],
                    p=p_random_rotation,
                ),
            ]
        )
        normalize = transforms.Compose(
            [
                transforms.Grayscale(num_output_channels=3),
                transforms.ToTensor(),
                # transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            ]
        )

        global_crop_aug_1 = [
            transforms.RandomResizedCrop(
                global_crop_size,
                scale=global_crops_scale,
                interpolation=Image.BICUBIC,
            )
        ]
        if use_dino_augmentation:
            global_crop_aug_1 += [
                flip_and_color_jitter,
                utils.GaussianBlur(0 if disable_gaussian_blur else 1.0),
            ]
        if use_edge_preserving_filter:
            global_crop_aug_1 += [EdgePreservingFilter()]
        global_crop_aug_1 += [normalize]
        # first global crop
        self.global_transfo1 = transforms.Compose(global_crop_aug_1)

        global_crop_aug_2 = [
            transforms.RandomResizedCrop(
                global_crop_size,
                scale=global_crops_scale,
                interpolation=Image.BICUBIC,
            )
        ]
        if use_dino_augmentation:
            global_crop_aug_2 += [
                flip_and_color_jitter,
                utils.GaussianBlur(0 if disable_gaussian_blur else 0.1),
                utils.Solarization(p=p_solarization),
            ]
        if use_edge_preserving_filter:
            global_crop_aug_2 += [
                transforms.RandomApply([EdgePreservingFilter()], p=0.1)
            ]
        global_crop_aug_2 += [normalize]
        # second global crop
        self.global_transfo2 = transforms.Compose(global_crop_aug_2)

        local_crop_aug = [
            transforms.RandomResizedCrop(
                local_crop_size,
                scale=local_crops_scale,
                interpolation=Image.BICUBIC,
            )
        ]
        if use_dino_augmentation:
            local_crop_aug += [
                flip_and_color_jitter,
                utils.GaussianBlur(p=0 if disable_gaussian_blur else 0.5),
            ]
        if use_edge_preserving_filter:
            local_crop_aug += [transforms.RandomApply([EdgePreservingFilter()], p=0.5)]
        local_crop_aug += [normalize]
        # transformation for the local small crops
        self.local_crops_number = local_crops_number
        self.local_transfo = transforms.Compose(local_crop_aug)
        pass

    def __call__(self, image):
        crops = []
        crops.append(self.global_transfo1(image))
        crops.append(self.global_transfo2(image))
        for _ in range(self.local_crops_number):
            crops.append(self.local_transfo(image))
        return crops


class DataAugmentationDINO(object):
    def __init__(self, global_crops_scale, local_crops_scale, local_crops_number):
        flip_and_color_jitter = transforms.Compose(
            [
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomApply(
                    [
                        transforms.ColorJitter(
                            brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1
                        )
                    ],
                    p=0.8,
                ),
                # transforms.RandomGrayscale(p=0.2),
            ]
        )
        normalize = transforms.Compose(
            [
                transforms.Grayscale(num_output_channels=3),
                transforms.ToTensor(),
                # transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            ]
        )

        # first global crop
        self.global_transfo1 = transforms.Compose(
            [
                transforms.RandomResizedCrop(
                    224, scale=global_crops_scale, interpolation=Image.BICUBIC
                ),
                flip_and_color_jitter,
                utils.GaussianBlur(1.0),
                utils.Solarization(0.5),
                normalize,
            ]
        )
        # second global crop
        self.global_transfo2 = transforms.Compose(
            [
                transforms.RandomResizedCrop(
                    224, scale=global_crops_scale, interpolation=Image.BICUBIC
                ),
                flip_and_color_jitter,
                utils.GaussianBlur(0.1),
                utils.Solarization(0.2),
                normalize,
            ]
        )
        # transformation for the local small crops
        self.local_crops_number = local_crops_number
        self.local_transfo = transforms.Compose(
            [
                transforms.RandomResizedCrop(
                    96, scale=local_crops_scale, interpolation=Image.BICUBIC
                ),
                flip_and_color_jitter,
                utils.GaussianBlur(p=0.5),
                normalize,
            ]
        )

    def __call__(self, image):
        crops = []
        crops.append(self.global_transfo1(image))
        crops.append(self.global_transfo2(image))
        for _ in range(self.local_crops_number):
            crops.append(self.local_transfo(image))
        return crops
