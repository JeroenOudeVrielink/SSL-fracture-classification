import sys

sys.path.append("/home/jvrielink/Thesis/fracture-attention-guidance")
sys.path.append("/home/jvrielink/Thesis/fracture-attention-guidance/modelling")
from torchvision import transforms
from modelling.dataset import get_loader_from_dataset
import argparse
from torch import nn
import torch
from tqdm import tqdm
from pretrained.pretrained_resnet import get_pretrained_resnet50, preprocess_state_dict
from torchvision.models import resnet50, ResNet50_Weights
from pathlib import Path
from pretrained.moco_v3 import get_pretrained_moco_v3
from pretrained.moco_v2 import get_pretrained_moco_v2
from pretrained.dino import get_pretrained_dino
from pretrained.vicregl import get_pretrained_vicregl, get_pretrained_vicreg
from pretrained.AIML_dataset import AIMLDataset
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder

# SAVE_PATHS = [
#     "/code/visualization/projected_feature_vectors/dino/dinov1_downstream_v2",
#     "/code/visualization/projected_feature_vectors/vicregl/vicregl_downstream_v2",
#     "/code/visualization/projected_feature_vectors/moco_v3/moco_v3_downstream_v2",
#     "/code/visualization/projected_feature_vectors/moco_v2/moco_v2_downstream_v2",
#     "/code/visualization/projected_feature_vectors/vicreg/vicreg_downstream_v2",
#     # "/code/visualization/projected_feature_vectors/vicregl_cov_coeff_50",
# ]

# MODEL_PATHS = [
#     "/models_hdd/dinov1_base_params_bs128_ep400_2024-01-09_09-31-49/checkpoint0300.pth",
#     "/models_hdd/vicregl_base_params_bs128_2024-01-10_05-05-33/model_resnet50_ep300.pth",
#     "/models_hdd/moco_base_params_bs256_ep400_2024-01-10_06-17-46/checkpoint_0299.pth.tar",
#     "/models_hdd/moco_v2_bs512_base_params_2024-03-23_00-26-32/checkpoint_0299.pth.tar",
#     # "/models_local/vicregl_bs128_cov_coeff_10_2024-05-31_08-07-27/model_resnet50_ep100.pth",
#     # "/models_local/vicregl_bs128_cov_coeff_25_2024-06-02_07-47-59/model_resnet50_ep100.pth",
#     # "/models_local/vicregl_bs128_cov_coeff_50_2024-06-03_05-11-02/model_resnet50_ep80.pth",
#     "/data/vicregl_models/vicregl_bs128_alpha1_2024-03-23_07-29-10/model_resnet50_ep300.pth",
# ]

SAVE_PATHS = [
    "visualization/projected_feature_vectors/dino/dinov1_luxry",
    # "visualization/projected_feature_vectors/vicregl_AIML",
    #     "visualization/projected_feature_vectors/moco_v3_AIML",
    #     "visualization/projected_feature_vectors/moco_v2_AIML",
    #     "visualization/projected_feature_vectors/vicregl_cov_coeff_10_AIML",
    #     "visualization/projected_feature_vectors/vicregl_cov_coeff_25_AIML",
    #     "visualization/projected_feature_vectors/vicregl_cov_coeff_50_AIML",
    # "visualization/projected_feature_vectors/vicregl/vicregl_luxry",
]

MODEL_PATHS = [
    "/home/jvrielink/data_hdd/models/dinov1_base_params_bs128_ep400_2024-01-09_09-31-49/checkpoint0300.pth",
    # "/home/jvrielink/data_hdd/models/vicregl_base_params_bs128_2024-01-10_05-05-33/model_resnet50_ep300.pth",
    #     "/home/jvrielink/data_hdd/models/moco_base_params_bs256_ep400_2024-01-10_06-17-46/checkpoint_0299.pth.tar",
    #     "/home/jvrielink/data_hdd/models/moco_v2_bs512_base_params_2024-03-23_00-26-32/checkpoint_0299.pth.tar",
    #     "/home/jvrielink/models/vicregl_bs128_cov_coeff_10_2024-05-31_08-07-27/model_resnet50_ep100.pth",
    #     "/home/jvrielink/models/vicregl_bs128_cov_coeff_25_2024-06-02_07-47-59/model_resnet50_ep100.pth",
    #     "/home/jvrielink/models/vicregl_bs128_cov_coeff_50_2024-06-03_05-11-02/model_resnet50_ep80.pth",
    # "/home/jvrielink/data/vicregl_models/vicregl_bs128_alpha1_2024-03-23_07-29-10/model_resnet50_ep300.pth",
]

# SAVE_PATHS = [
#     "visualization/projected_feature_vectors/vicreg_imagenet_train",
#     "visualization/projected_feature_vectors/vicregl_imagenet_train",
#     "visualization/projected_feature_vectors/moco_v2_imagenet_train",
#     "visualization/projected_feature_vectors/moco_v3_imagenet_train",
#     "visualization/projected_feature_vectors/dino/dinov1_imagenet_train"
# ]

# SAVE_PATHS = [
#     # "visualization/projected_feature_vectors/vicreg_imagenet_test",
#     # "visualization/projected_feature_vectors/vicregl_imagenet_test",
#     # "visualization/projected_feature_vectors/moco_v2_imagenet_test",
#     # "visualization/projected_feature_vectors/moco_v3_imagenet_test",
#     "visualization/projected_feature_vectors/dino/dinov1_imagenet_test",
# ]

# SAVE_PATHS = [
#     # "visualization/projected_feature_vectors/vicreg/vicreg_pascal",
#     # "visualization/projected_feature_vectors/vicregl/vicregl_pascal",
#     # "visualization/projected_feature_vectors/moco_v2/moco_v2_pascal",
#     # "visualization/projected_feature_vectors/moco_v3/moco_v3_pascal",
#     "visualization/projected_feature_vectors/dino/dinov1_coco",
# ]

# MODEL_PATHS = [
#     # "/home/jvrielink/models/imagenet_models/vicreg_resnet50_fullckpt.pth",
#     # "/home/jvrielink/models/imagenet_models/vicregl_resnet50_alpha0.75_fullckpt.pth",
#     # "/home/jvrielink/models/imagenet_models/moco_v2_200ep_pretrain.pth.tar",
#     # "/home/jvrielink/models/imagenet_models/mocov3-r-50-300ep.pth.tar",
#     "/home/jvrielink/models/imagenet_models/dino_rn50_checkpoint.pth",
# ]


# "downstream", "luxry", "imagenet_train", "imagenet_test", "coco", "pascal"
DATA_TYPE = "luxry"


def str_to_pretrained_method(path):
    if "spark" in path:
        return "spark"
    elif "dino" in path:
        return "dinov1"
    elif "vicreg" in path:
        if "vicregl" in path and "alpha1" not in path:
            return "vicregl"
        return "vicreg"
    elif "moco" in path:
        if "moco_v2" in path:
            return "mocov2"
        return "mocov3"
    elif "imgnet" or "baseline" in path:
        return "imgnet"


IMGNET = False
if DATA_TYPE == "downstream":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_size", type=int, default=224)
    parser.add_argument("--scribble_type", type=str, default="scribble")
    args = parser.parse_args()
    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ]
    )
    datasets = {
        "train",
        "val",
        "ai",  # 'weak_ai', 'weak'
        "test",
        "weak_test",
    }
    config = {
        "cls_tasks": {"fibula": 4},
        "cls_cols_dict": {
            "classification": 4
        },  # Here for legacy reasons to make coder work
        "binary_cls_cols": [],  # 'modality' # Here for legacy reasons to make coder work
        "regression_cols": [],  # Here for legacy reasons to make coder work
        "seg_cols_dict": {"fibula": 4},  # Here for legacy reasons to make coder work
        "ignore_value": -1e10,
        "args": vars(args),
    }
    data_loader = get_loader_from_dataset(
        data_root="/code/dataset/fracture/images",
        csv_path="/code/dataset/fracture/labels/master_v5_complete_no_radius_scribble.csv",
        config=config,
        target_translator=None,
        batch_size=128,
        transform=transform,
        set=datasets,
        use_random_crop=False,
        shuffle=False,
        num_workers=4,
        drop_last=False,
    )
elif DATA_TYPE == "downstream_v2":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_size", type=int, default=224)
    parser.add_argument("--scribble_type", type=str, default="scribble")
    args = parser.parse_args()
    transform = transforms.Compose(
        [
            transforms.Resize(224),
            transforms.CenterCrop(224),
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
        ]
    )
    datasets = {
        "train",
        "val",
        "ai",  # 'weak_ai', 'weak'
        "test",
        "weak_test",
    }
    config = {
        "cls_tasks": {"fibula": 4},
        "cls_cols_dict": {
            "classification": 4
        },  # Here for legacy reasons to make coder work
        "binary_cls_cols": [],  # 'modality' # Here for legacy reasons to make coder work
        "regression_cols": [],  # Here for legacy reasons to make coder work
        "seg_cols_dict": {"fibula": 4},  # Here for legacy reasons to make coder work
        "ignore_value": -1e10,
        "args": vars(args),
    }
    data_loader = get_loader_from_dataset(
        data_root="/code/dataset/fracture/images",
        csv_path="/code/dataset/fracture/labels/master_v5_complete_no_radius_scribble.csv",
        config=config,
        target_translator=None,
        batch_size=128,
        transform=transform,
        set=datasets,
        use_random_crop=False,
        shuffle=False,
        num_workers=4,
        drop_last=False,
        no_padding=True,
    )

if DATA_TYPE == "downstream_v3":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_size", type=int, default=224)
    parser.add_argument("--scribble_type", type=str, default="scribble")
    args = parser.parse_args()
    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ]
    )
    datasets = {
        "train",
        "val",
        "ai",  # 'weak_ai', 'weak'
        "test",
        "weak_test",
    }
    config = {
        "cls_tasks": {"fibula": 4},
        "cls_cols_dict": {
            "classification": 4
        },  # Here for legacy reasons to make coder work
        "binary_cls_cols": [],  # 'modality' # Here for legacy reasons to make coder work
        "regression_cols": [],  # Here for legacy reasons to make coder work
        "seg_cols_dict": {"fibula": 4},  # Here for legacy reasons to make coder work
        "ignore_value": -1e10,
        "args": vars(args),
    }
    data_loader = get_loader_from_dataset(
        data_root="/code/dataset/fracture/images",
        csv_path="/code/dataset/fracture/labels/master_v5_complete_no_radius_scribble.csv",
        config=config,
        target_translator=None,
        batch_size=128,
        transform=transform,
        set=datasets,
        use_random_crop=False,
        shuffle=False,
        num_workers=4,
        drop_last=False,
        no_padding=True,
    )

elif DATA_TYPE == "luxry":
    transform = transforms.Compose(
        [
            transforms.Resize(224),
            transforms.CenterCrop(224),
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
        ]
    )
    dataset = AIMLDataset(
        annotations_file="annotations/merged_classes_age/test.pkl",
        data_path="/home/jvrielink/data_hdd/AIML_rot_corrected",
        transform=transform,
    )
    data_loader = DataLoader(
        dataset=dataset,
        batch_size=128,
        shuffle=False,
        num_workers=4,
        drop_last=False,
    )
elif "imagenet" or "coco" or "pascal" in DATA_TYPE:
    IMGNET = True
    transform = transforms.Compose(
        [
            transforms.Resize(224),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ],
    )

    if DATA_TYPE == "imagenet_train":
        data_path = "/home/jvrielink/imagenet/imagenet-mini/train"
    elif DATA_TYPE == "imagenet_test":
        data_path = "/home/jvrielink/imagenet/ILSVRC2012_img_test_v10102019"
    elif DATA_TYPE == "coco":
        data_path = "/home/jvrielink/coco"
    elif DATA_TYPE == "pascal":
        data_path = "/home/jvrielink/PASCAL/VOCtest_06-Nov-2007/VOCdevkit/test_imgs"
    dataset = ImageFolder(root=data_path, transform=transform)
    data_loader = DataLoader(
        dataset=dataset,
        batch_size=512,
        shuffle=False,
        num_workers=4,
        drop_last=False,
    )


def gen_and_save_feature_vectors(model, save_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    test = torch.zeros(1, 3, 224, 224)
    x = model(test.to(device))
    embed_dim = x.shape[1]

    features_to_save = torch.empty(0, embed_dim)
    labels_body = torch.empty(0, 13)
    labels_view = torch.empty(0, 3)
    for out in tqdm(data_loader):
        with torch.no_grad():
            images = out[0]
            features = model(images.to(device))
            feature_vecs_cpu = features.to("cpu")
            features_to_save = torch.cat((features_to_save, feature_vecs_cpu), dim=0)
            labels_body = torch.cat((labels_body, out[1]), dim=0)
            labels_view = torch.cat((labels_view, out[2]), dim=0)

    # Save the feature maps
    torch.save(features_to_save, save_path / "feature_vectors.pt")
    torch.save(labels_body, save_path / "labels_body.pt")
    torch.save(labels_view, save_path / "labels_view.pt")


if __name__ == "__main__":
    for i, model_path in enumerate(MODEL_PATHS):
        pretrained_method = str_to_pretrained_method(model_path)
        print("Generating feature vectors for: ", pretrained_method)

        if pretrained_method == "mocov3":
            model = get_pretrained_moco_v3(model_path, imgnet=IMGNET)
        elif pretrained_method == "mocov2":
            model = get_pretrained_moco_v2(model_path, imgnet=IMGNET)
        elif pretrained_method == "dinov1":
            model = get_pretrained_dino(model_path, imgnet=IMGNET)
        elif pretrained_method == "vicregl":
            model = get_pretrained_vicregl(model_path, imgnet=IMGNET)
        elif pretrained_method == "vicreg":
            model = get_pretrained_vicreg(model_path, imgnet=IMGNET)

        save_dir = Path(SAVE_PATHS[i])
        save_dir.mkdir(parents=True, exist_ok=True)
        gen_and_save_feature_vectors(model, save_dir)
