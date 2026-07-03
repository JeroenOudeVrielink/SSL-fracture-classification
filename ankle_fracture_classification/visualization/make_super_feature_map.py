import torch
from torch import nn
from torchvision.utils import make_grid
import matplotlib.pyplot as plt
from torchvision import transforms
from PIL import Image
import numpy as np
import sys

sys.path.append("/home/jvrielink/Thesis/fracture-attention-guidance")
sys.path.append("/home/jvrielink/Thesis/fracture-attention-guidance/modelling")
sys.path.append("/home/jvrielink/Thesis/fracture-attention-guidance/pretrained")

from pretrained.pretrained_resnet import get_pretrained_resnet50

MODEL_PATH = "/mnt/sdb1/Data_remote/models/spark_base_params_bs256_ep400_01-10_04:56:48/resnet50_1kpretrained_timm_style_ep300.pth"
IMG_PATH = "visualization/test_imgs/3.png"


def patches_to_grid(images, nrow=7):
    """
    Arrange patches into a grid of images with gradients enabled.

    Args:
        images (Tensor): Input tensor of patches to be arranged into a grid.
        nrow (int, optional): Number of patches displayed in each row of the grid. Default is 7.

    Returns:
        Tensor: A tensor containing the grid of images.
    """
    # Get the shape of the input tensor
    batch_size, n_patches, height, width = images.shape

    # Check if the number of patches is divisible by nrow
    assert n_patches % nrow == 0, "Number of patches must be divisible by nrow"

    # Calculate number of columns in the grid
    ncol = n_patches // nrow

    # Unfold the patches along height and width dimensions
    # (batch_size, n_patches, nrow, height, width)
    unfolded_height = images.unfold(2, height, height)
    # (batch_size, n_patches, nrow, ncol, height, width)
    unfolded_patches = unfolded_height.unfold(3, width, width)

    # Reshape and permute to form the grid
    grid = unfolded_patches.contiguous().view(batch_size, nrow, ncol, height, width)
    grid = (
        grid.permute(0, 1, 3, 2, 4)
        .contiguous()
        .view(batch_size, nrow * height, ncol * width)
    )

    return grid


def inverse_super_fmap(x):
    patches = torch.zeros(49, 32, 32)
    patch_number = 0
    for i in range(32, 225, 32):
        for j in range(32, 225, 32):
            patch = x[i - 32 : i, j - 32 : j]
            patches[patch_number] = patch
            patch_number += 1

    new_patches = torch.zeros(1024, 7, 7)
    patch_number = 0
    for i in range(0, 32):
        for j in range(0, 32):
            new_patch = patches[:, i, j]
            new_patches[patch_number] = new_patch.reshape(7, 7)
            patch_number += 1
    return new_patches


def display_example_img():
    x = torch.randn(1, 2048, 7, 7)
    x = x.reshape(x.shape[0], x.shape[1], x.shape[2] * x.shape[3])
    x = x.permute(2, 1, 0)
    for i in range(x.shape[0]):
        x[i].fill_(float(i))
    x = x.permute(2, 1, 0)
    x = x.reshape(x.shape[0], 2048, 7, 7)
    images = make_grid(
        x[0][0:1024].unsqueeze(1),
        nrow=32,
        normalize=True,
        value_range=(0, 50),
        padding=0,
    )
    # Convert the tensor to a numpy array
    grid_image = images.permute(1, 2, 0).numpy()

    # Plot the grid image
    plt.imshow(grid_image)
    plt.axis("off")
    plt.show()

    # batch, 2, 1024, 7, 7
    x = x.reshape(x.shape[0], 2, 1024, x.shape[2], x.shape[3])
    # batch, 2, 32, 32, 49
    x = x.reshape(x.shape[0], x.shape[1], 32, 32, x.shape[3] * x.shape[4])
    # batch 2 49 32 32
    x = x.permute(0, 1, 4, 2, 3)

    # images = make_grid(
    #     x[0][0].unsqueeze(1), nrow=7, normalize=True, value_range=(0, 50), padding=0
    # )
    # # Convert the tensor to a numpy array
    x1 = x[:][0]
    image = patches_to_grid(x1)
    # image = x[0][0]
    grid_image = image[0]

    # Plot the grid image
    plt.imshow(grid_image)
    plt.axis("off")
    plt.show()

    origional_format = inverse_super_fmap(grid_image)
    images = make_grid(
        origional_format.unsqueeze(1),
        nrow=32,
        normalize=True,
        value_range=(0, 50),
        padding=0,
    )
    # Convert the tensor to a numpy array
    grid_image = images.permute(1, 2, 0).numpy()

    # Plot the grid image
    plt.imshow(grid_image)
    plt.axis("off")
    plt.show()


class Visualizer:
    def __init__(self, model, pretrained_method="Not given", img_size=224):
        self.model = model
        self.pretrained_method = pretrained_method
        self.img_size = img_size
        self.transform = transforms.Compose(
            [
                transforms.Resize(224),
                transforms.CenterCrop(224),
                transforms.Grayscale(num_output_channels=3),
                transforms.ToTensor(),
            ]
        )

    def get_feature_maps(self, image):
        image = self.transform(image).unsqueeze(0)
        feature_maps = self.model(image)
        return feature_maps

    def make_super_feature_map(self, x):
        # batch, 2, 1024, 7, 7
        x = x.reshape(x.shape[0], 2, 1024, x.shape[2], x.shape[3])
        # batch, 2, 32, 32, 49
        x = x.reshape(x.shape[0], x.shape[1], 32, 32, x.shape[3] * x.shape[4])
        # batch 2 49 32 32
        x = x.permute(0, 1, 4, 2, 3)
        image1 = make_grid(x[0][0].unsqueeze(1), nrow=7, padding=0)
        image2 = make_grid(x[0][1].unsqueeze(1), nrow=7, padding=0)
        return image1.permute(1, 2, 0).numpy(), image2.permute(1, 2, 0).numpy()

    def extract_and_plot_feature_maps(self, img_path, save_path="debug.png"):
        image = Image.open(img_path)
        feature_maps = self.get_feature_maps(image)

        image1, image2 = self.make_super_feature_map(feature_maps)

        plt.imshow(image1[:, :, 0], cmap="hot")
        plt.axis("off")
        plt.show()

        plt.imshow(image2[:, :, 0], cmap="hot")
        plt.axis("off")
        plt.show()


def str_to_pretrained_method(path):
    if "spark" in path:
        return "spark"
    elif "dinov1" in path:
        return "dinov1"
    elif "vicregl" in path:
        return "vicregl"
    elif "moco" in path:
        return "mocov3"


if __name__ == "__main__":
    display_example_img()
    # pretrained_method = str_to_pretrained_method(MODEL_PATH)
    # model, _, _ = get_pretrained_resnet50(pretrained_method, MODEL_PATH)
    # visualizer = Visualizer(model, pretrained_method)
    # visualizer.extract_and_plot_feature_maps(IMG_PATH)
