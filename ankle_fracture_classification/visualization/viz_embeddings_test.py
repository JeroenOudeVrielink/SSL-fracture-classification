import matplotlib.pyplot as plt
from pretrained.pretrained_resnet import get_pretrained_resnet50
from PIL import Image
from torchvision import transforms
from tqdm import tqdm
import os
from pathlib import Path
from torchvision.models import resnet50, ResNet50_Weights
import torch


SAVE_PATH = "/home/jvrielink/Thesis/fracture-attention-guidance/visualization/test"
IMG_DIR_PATH = (
    "/home/jvrielink/Thesis/fracture-attention-guidance/visualization/test_imgs"
)
MODEL_PATHS = [
    "/mnt/sdb1/Data_remote/models/spark_base_params_bs256_ep400_01-10_04:56:48/resnet50_1kpretrained_timm_style_ep300.pth",
    "/mnt/sdb1/Data_remote/models/dinov1_base_params_bs128_ep400_2024-01-09_09-31-49/checkpoint0300.pth",
    "/mnt/sdb1/Data_remote/models/vicregl_base_params_bs128_2024-01-10_05-05-33/model_resnet50_ep300.pth",
    "/mnt/sdb1/Data_remote/models/moco_base_params_bs256_ep400_2024-01-10_06-17-46/checkpoint_0299.pth.tar",
    "imgnet",
]


class Visualizer:
    def __init__(self, model, pretrained_method="Not given", n_feature_maps=500):
        self.model = model
        self.pretrained_method = pretrained_method
        self.transform = transforms.Compose(
            [
                transforms.Resize(448),
                transforms.Grayscale(num_output_channels=3),
                transforms.ToTensor(),
            ]
        )
        self.n_feature_maps = n_feature_maps

    def get_feature_maps(self, image):
        # image = Image.open(image)
        image = self.transform(image).unsqueeze(0)
        feature_maps = self.model(image)
        return self.processed_feature_maps(feature_maps)

    def processed_feature_maps(self, feature_maps):
        feature_maps = feature_maps.squeeze(0)
        processed = []
        for feature_map in feature_maps[0 : self.n_feature_maps]:
            # feature_map = feature_map / (feature_map.max() + 1e-5)
            processed.append(feature_map.data.cpu().numpy())
        return processed

    def extract_and_plot_feature_maps(self, img_path, save_path="debug.png"):
        image = Image.open(img_path)
        feature_maps = self.get_feature_maps(image)
        num_rows = 25
        num_cols = 20
        fig, ax = plt.subplots(num_rows, num_cols, figsize=(20, 25))
        # Flatten the axis array for easy indexing
        ax = ax.flatten()
        # Plot each image
        for i, img in enumerate(feature_maps):
            ax[i].imshow(img, cmap="hot", interpolation="nearest")
            ax[i].axis("off")

        # Hide remaining subplots
        # for i in range(self.n_feature_maps, num_rows * num_cols):
        #     ax[i].axis("off")

        plt.suptitle(
            f"First {self.n_feature_maps} embedding feature maps of {self.pretrained_method}",
            fontsize=36,
            fontweight="bold",
        )
        plt.tight_layout(
            rect=[0, 0.03, 1, 0.95]
        )  # Adjust subplot layout to accommodate the title
        plt.savefig(save_path)
        plt.close()

    def save_feature_maps(self, feature_maps, save_path, img_name):
        for i in range(len(feature_maps)):
            plt.imshow(feature_maps[i], cmap="gray")
            plt.axis("off")
            plt.savefig(
                os.path.join(
                    save_path, f"{self.pretrained_method}_{img_name}_layer_{i}.png"
                )
            )
            plt.close()


def str_to_pretrained_method(path):
    if "spark" in path:
        return "spark"
    elif "dinov1" in path:
        return "dinov1"
    elif "vicregl" in path:
        return "vicregl"
    elif "moco" in path:
        return "mocov3"


def main():
    for model_path in tqdm(MODEL_PATHS, total=len(MODEL_PATHS)):
        if model_path == "imgnet":
            pretrained_method = "imgnet"
            model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
            model = torch.nn.Sequential(*list(model.children())[:-2])
        else:
            pretrained_method = str_to_pretrained_method(model_path)
            model, _, _ = get_pretrained_resnet50(pretrained_method, model_path)
        visualizer = Visualizer(model, pretrained_method)
        save_dir = Path(SAVE_PATH) / pretrained_method
        save_dir.mkdir(parents=True, exist_ok=True)

        for i in tqdm([1, 3]):
            img_path = Path(IMG_DIR_PATH) / f"{i}.png"
            save_path = save_dir / f"img_{i}.png"
            visualizer.extract_and_plot_feature_maps(img_path, save_path)


if __name__ == "__main__":
    main()
