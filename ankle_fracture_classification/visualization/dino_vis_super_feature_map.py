import sys

sys.path.append("/home/jvrielink/Thesis/fracture-attention-guidance/pretrained")
from pretrained.pretrained_resnet import preprocess_state_dict
import torch
from dino_head import DINOHeadV2, DINOHeadV3
import torch.nn as nn
from torchvision.models import resnet50
import torchvision
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt
import torchvision.transforms.functional as F
import numpy as np
from make_super_feature_map import inverse_super_fmap
import seaborn as sns
from pathlib import Path

# CHECKPOINT_PATH = "~/data-hdd/models/dinov1_bs64_conv_head_v3_2024-03-12_17-00-48/checkpoint0060.pth"
CHECKPOINT_PATH = "/home/jvrielink/data_hdd/models/dinov1_bs128_ep400_super_fmap_v3_2024-03-15_02-13-47/checkpoint0300.pth"
IMG_PATH = "visualization/test_imgs/9.png"
PLOT_SAVE_DIR = "visualization/super_fmap_overlay"

transform = transforms.Compose(
    [
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
    ]
)


class DINOModule(torch.nn.Module):
    def __init__(self, dino_head_version):
        super(DINOModule, self).__init__()
        resnet = resnet50()
        self.backbone = nn.Sequential(*list(resnet.children())[:-2])
        if dino_head_version == "v2":
            self.head = DINOHeadV2()
        elif dino_head_version == "v3":
            self.head = DINOHeadV3(in_dim=2048, out_dim=224 * 224)

    def forward(self, x):
        x = self.backbone(x)
        x, super_fmap = self.head(x)
        return x, super_fmap


def split_image_into_pixel_groups(image):
    """
    Splits a 224x224 image into 1024 7x7 images, where each 7x7 image
    contains a specific pixel position from each 32x32 patch.

    Args:
        image (torch.Tensor): A PyTorch tensor representing the 224x224 image.
                              Assumed to have shape (C, 224, 224) where C is the number of channels.

    Returns:
        list: A list of 1024 PyTorch tensors, each of shape (C, 7, 7) representing
              the pixel groups.
    """

    if image.shape[-2:] != (224, 224):
        raise ValueError("Image must have dimensions of 224x224")

    # Split into 32x32 patches
    patches = image.unfold(1, 32, 32).unfold(2, 32, 32)
    patches = patches.contiguous().view(-1, 32, 32)  # (num_patches, 32, 32)

    # Reshape to isolate each pixel position across patches
    pixel_groups = patches.permute(0, 2, 1).reshape(-1, 7, 7)  # (1024, 7, 7)

    return pixel_groups


def show(imgs):
    if not isinstance(imgs, list):
        imgs = [imgs]
    fig, axs = plt.subplots(ncols=len(imgs), squeeze=False)
    for i, img in enumerate(imgs):
        img = img.detach()
        img = F.to_pil_image(img)
        axs[0, i].imshow(np.asarray(img))
        axs[0, i].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])


def get_pretrained_dino_head(ckpt_path):
    checkpoint = torch.load(ckpt_path, "cpu")
    state_dict = checkpoint["student"]
    state_dict = preprocess_state_dict(
        state_dict, remove=[], replace="module.", replace_with=""
    )
    try:
        model = DINOModule("v3")
        model.load_state_dict(state_dict)
    except:
        model = DINOModule("v2")
        model.load_state_dict(state_dict)
    return model


def show_super_feature_map(ckpt_path, img_path):
    model = get_pretrained_dino_head(ckpt_path)
    img = Image.open(img_path)
    x = transform(img).unsqueeze(0)
    _, super_fmap = model(x)
    super_fmap = super_fmap.squeeze(0).squeeze(0).detach().cpu()

    np_super_fmap = super_fmap.numpy()
    plt.imshow(np_super_fmap, cmap="hot")
    plt.axis("off")
    plt.show()

    inv_super_fmap = inverse_super_fmap(super_fmap)
    inv_super_fmap.unsqueeze_(1)
    grid = torchvision.utils.make_grid(
        inv_super_fmap[0:256],
        nrow=16,
        padding=0,
    )

    # Assuming single-channel images; adjust if you have multiple channels
    # show(grid)
    grid_image = grid.permute(1, 2, 0).numpy()
    plt.imshow(grid_image[:, :, 0], cmap="hot")
    plt.axis("off")
    plt.show()

    grid = torchvision.utils.make_grid(
        inv_super_fmap[256:512],
        nrow=16,
        padding=0,
    )

    # Assuming single-channel images; adjust if you have multiple channels
    # show(grid)
    grid_image = grid.permute(1, 2, 0).numpy()
    plt.imshow(grid_image[:, :, 0], cmap="hot")
    plt.axis("off")
    plt.show()

    grid = torchvision.utils.make_grid(
        inv_super_fmap[512:768],
        nrow=16,
        padding=0,
    )

    # Assuming single-channel images; adjust if you have multiple channels
    # show(grid)
    grid_image = grid.permute(1, 2, 0).numpy()
    plt.imshow(grid_image[:, :, 0], cmap="hot")
    plt.axis("off")
    plt.show()

    grid = torchvision.utils.make_grid(
        inv_super_fmap[768:],
        nrow=16,
        padding=0,
    )

    # Assuming single-channel images; adjust if you have multiple channels
    # show(grid)
    grid_image = grid.permute(1, 2, 0).numpy()
    plt.imshow(grid_image[:, :, 0], cmap="hot")
    plt.axis("off")
    plt.show()


def overlay_super_fmap_with_input(ckpt_path, img_path, save_path):
    model = get_pretrained_dino_head(ckpt_path)
    img = Image.open(img_path)
    x = transform(img).unsqueeze(0)
    _, super_fmap = model(x)
    super_fmap = super_fmap.squeeze(0).squeeze(0).detach().cpu()

    np_super_fmap = super_fmap.numpy()

    np_img = x.squeeze(0).permute(1, 2, 0).numpy()
    # Create figure and axes

    # Apply the styles
    plt.rcParams["text.usetex"] = True
    plt.rcParams["font.family"] = "serif"
    sns.set_theme(style="whitegrid", font_scale=1.2)  # Adjust font scale as needed

    fontsize = 15  # Adjust legend font size

    plt.figure(figsize=(8, 6))  # Adjust figure size as needed
    fig, ax = plt.subplots()

    # Display the image
    ax.imshow(np_img, cmap="gray")
    # Overlay the heatmap
    sns.heatmap(np_super_fmap, alpha=0.5, cmap="Reds", ax=ax)

    cbar = ax.collections[0].colorbar

    # Set the label and its font size
    cbar.set_label("Feature map value", fontsize=fontsize)
    cbar.ax.yaxis.label.set_fontfamily("serif")

    # Remove x and y axis labels and ticks
    ax.set_xticks([])
    ax.set_yticks([])

    plt.tight_layout()

    # Save the figure (optional)
    plt.savefig(save_path + ".png", dpi=300)  # High DPI for print quality
    plt.savefig(save_path + ".svg", format="svg", dpi=300)  # SVG for scalability


if __name__ == "__main__":
    # model, _, _ = get_pretrained_resnet50("dinov1", CHECKPOINT_PATH)
    # show_super_feature_map(CHECKPOINT_PATH, IMG_PATH)
    Path(PLOT_SAVE_DIR).mkdir(parents=True, exist_ok=True)
    plot_save_path = f"{PLOT_SAVE_DIR}/{Path(IMG_PATH).stem}"
    overlay_super_fmap_with_input(CHECKPOINT_PATH, IMG_PATH, plot_save_path)
