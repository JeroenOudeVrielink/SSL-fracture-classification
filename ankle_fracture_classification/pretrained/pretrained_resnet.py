import torch
import timm
from torch import nn
import torchvision
from torchvision.models import resnet50
import pretrained.optimizers as optimizers  # KEEP THIS! Necessary for vicregl loading
from modelling.utils import remove_named_children

# LOAD_PATH = "/mnt/sdb1/Data_remote/models/spark_base_params_bs256_ep400_01-10_04:56:48/resnet50_1kpretrained_timm_style_ep40.pth"
# LOAD_PATH = "/mnt/sdb1/Data_remote/models/dinov1_base_params_bs128_ep400_2024-01-09_09-31-49/checkpoint.pth"
# LOAD_PATH = "/mnt/sdb1/Data_remote/models/vicregl_base_params_bs128_2024-01-10_05-05-33/model_resnet50_ep90.pth"
# LOAD_PATH = "/mnt/sdb1/Data_remote/models/dinov1_bs128_conv_head_2024-03-05_03-10-02/checkpoint0060.pth"
# LOAD_PATH = "/mnt/sdb1/Data_remote/models/exp_merged_classes_v25_weighted_sampling_square_resize_rotation_correction_12-30_01:56:44/epoch=00.ckpt"
LOAD_PATH = "/home/jvrielink/data/moco_models/moco_v2_bs512_base_params_2024-03-20_02-36-24/checkpoint_0299.pth.tar"


def preprocess_state_dict(state_dict, remove=[], replace="", replace_with=""):
    filtered_state_dict = {
        key: value
        for key, value in state_dict.items()
        if not any(key.startswith(prefix) for prefix in remove)
    }
    renamed_state_dict = {
        key.replace(replace, replace_with): value
        for key, value in filtered_state_dict.items()
    }
    return renamed_state_dict


def get_pretrained_resnet50(pretrained_method, load_path):
    """Function thar loads the saved model from the respective pretraining method
    It then preporcesses the state dict to match resnet model, by filtering dict keys"""

    model = resnet50()
    in_features = model.fc.in_features
    adapt_layer = None

    if pretrained_method == "imagenet_pretrained":
        net = getattr(torchvision.models, "resnet50")(weights="IMAGENET1K_V1")
        in_features = net.fc.in_features
        model = remove_named_children(net, ["fc", "avgpool"])

    elif pretrained_method == "imagenet_pretrained_v2":
        net = getattr(torchvision.models, "resnet50")(weights="IMAGENET1K_V2")
        in_features = net.fc.in_features
        model = remove_named_children(net, ["fc", "avgpool"])

    elif pretrained_method == "classification":
        checkpoint = torch.load(load_path, "cpu")

        state_dict = checkpoint["state_dict"]
        state_dict = preprocess_state_dict(
            state_dict, remove=["decoder"], replace="encoder.", replace_with=""
        )
        model = nn.Sequential(*list(model.children())[:-2])
        model.load_state_dict(state_dict)

    elif pretrained_method == "classification_adapt_layer":
        checkpoint = torch.load(load_path, "cpu")

        state_dict = checkpoint["state_dict"]
        state_dict = preprocess_state_dict(
            state_dict,
            remove=["decoder", "adapt_layer"],
            replace="encoder.",
            replace_with="",
        )
        model = nn.Sequential(*list(model.children())[:-2])
        model.load_state_dict(state_dict)

        adapt_layer = nn.Sequential(
            nn.Conv2d(in_features, in_features, kernel_size=1), nn.ReLU()
        )
        state_dict = checkpoint["state_dict"]
        state_dict = preprocess_state_dict(
            state_dict,
            remove=["encoder", "decoder"],
            replace="adapt_layer.",
            replace_with="",
        )
        adapt_layer.load_state_dict(state_dict)

    elif pretrained_method == "spark" or pretrained_method == "spark_weighted_masking":
        model, timm.create_model("resnet50")
        checkpoint = torch.load(load_path, "cpu")
        model = remove_named_children(model, ["fc", "avgpool"])
        model.load_state_dict(checkpoint.get("module", checkpoint))

    elif pretrained_method == "dinov1":
        model = resnet50()
        in_features = model.fc.in_features
        checkpoint = torch.load(load_path, "cpu")
        state_dict = preprocess_state_dict(
            checkpoint["student"],
            remove=["module.head."],
            replace="module.backbone.",
            replace_with="",
        )
        try:
            regular_dino = remove_named_children(model, ["fc", "avgpool"])
            regular_dino.load_state_dict(state_dict)
            model = regular_dino
        except:
            print("Loading DINOv1 model with conv head...")
            dino_conv_head = nn.Sequential(*list(model.children())[:-2])
            dino_conv_head.load_state_dict(state_dict)
            model = dino_conv_head
    elif pretrained_method == "mocov3":
        model = resnet50()
        in_features = model.fc.in_features
        model = remove_named_children(model, ["fc", "avgpool"])
        checkpoint = torch.load(load_path, "cpu")
        state_dict = preprocess_state_dict(
            checkpoint["state_dict"],
            remove=["momentum_encoder", "predictor", "base_encoder.fc"],
            replace="base_encoder.",
            replace_with="",
        )
        model.load_state_dict(state_dict)
    elif pretrained_method == "mocov2":
        model = resnet50()
        in_features = model.fc.in_features
        model = remove_named_children(model, ["fc", "avgpool"])
        checkpoint = torch.load(load_path, "cpu")
        state_dict = preprocess_state_dict(
            checkpoint["state_dict"],
            remove=[
                "module.queue",
                "module.queue_ptr",
                "module.encoder_k",
                "module.encoder_q.fc",
            ],
            replace="module.encoder_q.",
            replace_with="",
        )
        model.load_state_dict(state_dict)

    elif pretrained_method == "vicregl":
        model = resnet50()
        in_features = model.fc.in_features
        model = remove_named_children(model, ["fc", "avgpool"])
        checkpoint = torch.load(load_path, "cpu")
        state_dict = preprocess_state_dict(
            checkpoint["model"],
            remove=["maps_projector.", "projector.", "classifier."],
            replace="backbone.",
            replace_with="",
        )
        model.load_state_dict(state_dict)
    else:
        print("Pretraining method not implemented")
        raise NotImplementedError

    return model, in_features, adapt_layer


if __name__ == "__main__":
    model, in_features, adapt_layer = get_pretrained_resnet50("mocov2", LOAD_PATH)
    # for name, child in model.named_modules():
    #     print(name)
    # print(model(torch.randn(1, 3, 512, 512)).shape)
    # if adapt_layer is not None:
    #     print(adapt_layer(model(torch.randn(1, 3, 512, 512))).shape)
