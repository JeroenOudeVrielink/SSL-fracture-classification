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
from pretrained.AIML_dataset import AIMLDataset
from torch.utils.data import DataLoader


SAVE_PATH = (
    "/home/jvrielink/fracture-attention-guidance/visualization/luxry_resnet_features"
)

MODEL_PATHS = [
    # "/home/jvrielink/data_hdd/models/dinov1_base_params_bs128_ep400_2024-01-09_09-31-49/checkpoint0300.pth",
    # "/home/jvrielink/data_hdd/models/spark_base_params_bs256_ep400_01-10_04:56:48/resnet50_1kpretrained_timm_style_ep300.pth",
    # "/home/jvrielink/data_hdd/models/vicregl_base_params_bs128_2024-01-10_05-05-33/model_resnet50_ep300.pth",
    # "/home/jvrielink/data_hdd/models/moco_base_params_bs256_ep400_2024-01-10_06-17-46/checkpoint_0299.pth.tar",
    # "/home/jvrielink/data/vicregl_models/vicregl_bs128_alpha1_2024-03-23_07-29-10/model_resnet50_ep300.pth",
    # "/home/jvrielink/data_hdd/models/moco_v2_bs512_base_params_2024-03-23_00-26-32/checkpoint_0299.pth.tar",
    "/home/jvrielink/data/classification_models/Gen3_v1_imgnet_pretrained_color_jitter_rotation_lr_schedule_04-11_02:00:18/epoch12.ckpt",
    "/home/jvrielink/data/classification_models/Gen3_v3_imgnet_pretrained_color_jitter_rotation_lr_schedule_04-14_14:55:25/epoch08.ckpt",
]

# MODEL_PATHS = [
#     "/models_hdd/spark_base_params_bs256_ep400_01-10_04:56:48/resnet50_1kpretrained_timm_style_ep300.pth",
#     "/models_hdd/dinov1_base_params_bs128_ep400_2024-01-09_09-31-49/checkpoint0300.pth",
#     "/models_hdd/vicregl_base_params_bs128_2024-01-10_05-05-33/model_resnet50_ep300.pth",
#     "/models_hdd/moco_base_params_bs256_ep400_2024-01-10_06-17-46/checkpoint_0299.pth.tar",
#     "imgnet",
# ]
# MODEL_PATHS = [
#     "/mnt/sdb1/Data_remote/models/baseline_bs32_save_model_03-12_03:59:21/net_e100.ckpt",
#     "/mnt/sdb1/Data_remote/models/dinov1_save_model_03-12_02:28:07/net_e100.ckpt",
#     "/mnt/sdb1/Data_remote/models/mocov3_save_model_03-08_06:38:48/net_e100.ckpt",
#     "/mnt/sdb1/Data_remote/models/vicregl_save_model_03-08_09:40:21/net_e100.ckpt",
#     "/mnt/sdb1/Data_remote/models/spark_save_model_03-08_08:09:28/net_e100.ckpt",
# ]
FINETUNED = False

DATATYPE = "luxry"


def str_to_pretrained_method(path):
    if "spark" in path:
        return "spark"
    elif "dinov1" in path:
        return "dinov1"
    elif "vicregl" in path:
        return "vicregl"
    elif "moco" in path:
        if "moco_v2" in path:
            return "mocov2"
        return "mocov3"
    elif "classification" in path:
        return "classification"
    elif "imgnet" or "baseline" in path:
        return "imgnet"


if DATATYPE == "downstream":

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
        batch_size=32,
        transform=transform,
        set=datasets,
        use_random_crop=False,
        shuffle=False,
        num_workers=4,
        drop_last=False,
    )

elif DATATYPE == "luxry":
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


def gen_and_save_feature_vectors(model, save_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    avg_pool = nn.AdaptiveAvgPool2d((1, 1))
    avg_pool.to(device)

    if DATATYPE == "downstream":

        tensor_2048 = torch.empty(0, 2048)
        labels = torch.empty(0, 4)
        for images, targets, _, _ in tqdm(data_loader):
            with torch.no_grad():
                feature_maps = model(images.to(device))
                feature_vecs = avg_pool(feature_maps)
                feature_vecs_cpu = feature_vecs.to("cpu")
                tensor_2048 = torch.cat(
                    (tensor_2048, feature_vecs_cpu.squeeze(2).squeeze(2)), dim=0
                )
                labels = torch.cat((labels, targets), dim=0)
    if DATATYPE == "luxry":
        tensor_2048 = torch.empty(0, 2048)
        labels_body = torch.empty(0, 13)
        labels_view = torch.empty(0, 3)
        for images, labels_b, labels_v in tqdm(data_loader):
            with torch.no_grad():
                feature_maps = model(images.to(device))
                feature_vecs = avg_pool(feature_maps)
                feature_vecs_cpu = feature_vecs.to("cpu")
                tensor_2048 = torch.cat(
                    (tensor_2048, feature_vecs_cpu.squeeze(2).squeeze(2)), dim=0
                )
                labels_body = torch.cat((labels_body, labels_b), dim=0)
                labels_view = torch.cat((labels_view, labels_v), dim=0)

    # Save the feature maps
    torch.save(tensor_2048, save_path / "feature_vectors.pt")
    if DATATYPE == "luxry":
        torch.save(labels_body, save_path / "labels_body.pt")
        torch.save(labels_view, save_path / "labels_view.pt")
    if DATATYPE == "downstream":
        torch.save(labels, save_path / "labels.pt")


def load_finetuned_model(model_path):
    model = resnet50()
    model.fc = nn.Identity()
    # model = torch.nn.Sequential(*list(model.children())[:-2])
    state_dict = torch.load(model_path)
    state_dict = preprocess_state_dict(
        state_dict, remove=["classifiers"], replace="net.", replace_with=""
    )
    model.load_state_dict(state_dict)
    model = nn.Sequential(*list(model.children())[:-2])
    return model


if __name__ == "__main__":
    for model_path in MODEL_PATHS:
        pretrained_method = str_to_pretrained_method(model_path)
        print("Generating feature vectors for: ", pretrained_method)

        if model_path == "imgnet":
            model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
            model = torch.nn.Sequential(*list(model.children())[:-2])
        elif FINETUNED:
            model = load_finetuned_model(model_path)
        else:
            model, _, _ = get_pretrained_resnet50(pretrained_method, model_path)
        if "alpha1" in model_path:
            pretrained_method = "vicreg"
        if "classification" in model_path:
            if "v1" in model_path:
                pretrained_method = "classification_v1"
            elif "v3" in model_path:
                pretrained_method = "classification_v3"
        save_dir = Path(SAVE_PATH) / pretrained_method
        save_dir.mkdir(parents=True, exist_ok=True)
        gen_and_save_feature_vectors(model, save_dir)
