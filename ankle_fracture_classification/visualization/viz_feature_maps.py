import matplotlib.pyplot as plt
import torch
from torch import nn
from pretrained.pretrained_resnet import get_pretrained_resnet50
from PIL import Image
from torchvision import transforms
from tqdm import tqdm
import os
from torchvision.models import resnet50, ResNet50_Weights

SAVE_PATH = "/home/jvrielink/Thesis/fracture-attention-guidance/visualization/visualized_feature_maps"
IMG_DIR_PATH = (
    "/home/jvrielink/Thesis/fracture-attention-guidance/visualization/test_imgs"
)
MODEL_PATHS = [
    "/mnt/sdb1/Data_remote/models/spark_base_params_bs256_ep400_01-10_04:56:48/resnet50_1kpretrained_timm_style_ep300.pth",
    "/mnt/sdb1/Data_remote/models/dinov1_base_params_bs128_ep400_2024-01-09_09-31-49/checkpoint0300.pth",
    "/mnt/sdb1/Data_remote/models/vicregl_base_params_bs128_2024-01-10_05-05-33/model_resnet50_ep300.pth",
    "/mnt/sdb1/Data_remote/models/moco_base_params_bs256_ep400_2024-01-10_06-17-46/checkpoint_0299.pth.tar",
]


class FeatureMapVisuzalizer:
    def __init__(self, model, pretrained_method="Not given"):
        self.model = model
        self.pretrained_method = pretrained_method
        self.conv_layers = self.get_conv_layers()
        self.transform = transforms.Compose(
            [
                transforms.Resize(224),
                transforms.Grayscale(num_output_channels=3),
                transforms.ToTensor(),
            ]
        )

    def get_conv_layers(self):
        model_children = list(self.model.children())
        conv_layers = []

        for child in range(len(model_children)):
            if type(model_children[child]) == nn.Conv2d:
                conv_layers.append(model_children[child])
            elif type(model_children[child]) == nn.Sequential:
                for i in range(len(model_children[child])):
                    for c in model_children[child][i].children():
                        if type(c) == nn.Conv2d:
                            conv_layers.append(c)
        # print(f"Number of conv layers: {len(conv_layers)}")
        return conv_layers

    def processed_feature_maps(self, feature_maps):
        processed = []
        for feature_map in feature_maps:
            feature_map = feature_map.squeeze(0)
            gray_scale = torch.sum(feature_map, 0)
            gray_scale = gray_scale / feature_map.shape[0]
            processed.append(gray_scale.data.cpu().numpy())
        return processed

    def get_feature_maps(self, image):
        image = self.transform(image)
        outputs = []
        for layer in self.conv_layers:
            image = layer(image)
            outputs.append(image)
        outputs = self.processed_feature_maps(outputs)
        return outputs

    def extract_and_plot_feature_maps(self, img_path, save_path="debug.png"):
        image = Image.open(img_path)

        feature_maps = self.get_feature_maps(image)
        fig = plt.figure(figsize=(40, 120))
        for i in range(len(feature_maps)):
            a = fig.add_subplot(13, 4, i + 1)
            a.axis("off")
            a.set_title(f"Conv layer {i}", fontsize=30)

        fig.savefig(save_path)


class PlotTogetherWrapper:
    def __init__(self, model_paths, include_imgnet_pretrained=True):
        self.visualizers = []
        if include_imgnet_pretrained:
            model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
            model = nn.Sequential(*list(model.children())[:-2])
            self.visualizers.append(FeatureMapVisuzalizer(model, "imgnet"))

        for model_path in model_paths:
            pretrained_method = str_to_pretrained_method(model_path)
            model, _, _ = get_pretrained_resnet50(pretrained_method, model_path)
            self.visualizers.append(FeatureMapVisuzalizer(model, pretrained_method))

    def extract_and_plot_feature_maps(self, img_path, save_path="debug.png"):
        image = Image.open(img_path)
        outputs = []
        for visualizer in self.visualizers:
            # Append output as a tuple (out, name)
            outputs.append(
                (visualizer.get_feature_maps(image), visualizer.pretrained_method)
            )

        fig = plt.figure(figsize=(80, 320))
        for i, (feature_maps, name) in enumerate(outputs):
            for j, map in enumerate(feature_maps):
                a = fig.add_subplot(49, 5, i + 1 + (5 * j))
                imgplot = plt.imshow(map)
                a.axis("off")
                a.set_title(f"{name} Conv{j}", fontsize=30)
        fig.savefig(save_path)


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
    wrapper = PlotTogetherWrapper(MODEL_PATHS)
    for i in tqdm(range(1, 11)):
        img_path = os.path.join(IMG_DIR_PATH, f"{i}.png")
        save_path = os.path.join(SAVE_PATH, f"img_{i}.png")
        wrapper.extract_and_plot_feature_maps(
            img_path,
            save_path,
        )

    # for model_path in tqdm(MODEL_PATHS):
    #     pretrained_method = str_to_pretrained_method(model_path)
    #     model, in_features, adapt_layer = get_pretrained_resnet50(
    #         pretrained_method, model_path
    #     )
    #     save_dir = os.path.join(SAVE_PATH, pretrained_method)
    #     os.makedirs(save_dir, exist_ok=True)

    #     vis = FeatureMapVisuzalizer(model)

    #     for i in range(1, 11):
    #         img_path = os.path.join(IMG_DIR_PATH, f"{i}.png")
    #         save_path = os.path.join(save_dir, f"{i}.png")
    #         vis.extract_and_plot_feature_maps(
    #             img_path,
    #             save_path,
    #         )
