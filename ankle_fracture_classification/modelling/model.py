import torch
import torch.nn as nn
import torchvision.models as models
from torch.nn import functional as F
from pretrained.pretrained_resnet import get_pretrained_resnet50
from modelling.utils import remove_named_children

device = "cuda" if torch.cuda.is_available() else "cpu"


class Net(nn.Module):
    def __init__(self, config):
        super(Net, self).__init__()

        self.config = config
        self.adapt_layer = None

        if self.config["arch"] == "resnet50":
            if self.config["pretrained_method"] is not None:
                net, in_features, self.adapt_layer = get_pretrained_resnet50(
                    self.config["pretrained_method"], self.config["ckpt_path"]
                )
                print(
                    f"Running with {self.config['pretrained_method']} {self.config['arch']} from: {self.config['ckpt_path']}"
                )
            else:
                net = getattr(models, self.config["arch"])()
                in_features = net.fc.in_features
                net = remove_named_children(net, ["fc", "avgpool"])

        elif self.config["arch"] == "convnext_base":
            net = models.convnext_base(
                weights=models.ConvNeXt_Base_Weights.IMAGENET1K_V1
            )
            in_features = 1024
            net = remove_named_children(net, ["avgpool", "classifier"])

        elif self.config["arch"] == "vit_b_16":
            if self.config["enable_attn_loss"] is True:
                raise NotImplementedError(
                    "Attention guidance is not supported for ViT backbone"
                )
            net = models.vit_b_16(weights=models.ViT_B_16_Weights.IMAGENET1K_V1)
            in_features = net.heads.head.in_features
            net.heads = nn.Identity()
        else:
            raise NotImplementedError("Unknown backbone")

        # Set net
        self.net = net

        # classification module
        self.classifiers = nn.ModuleDict()
        for t_name in self.config["cls_tasks"]:
            modules = list()
            # modules.append(nn.AdaptiveAvgPool2d((1, 1)))
            # modules.append(nn.Flatten())
            modules.append(nn.Linear(in_features, self.config["cls_tasks"][t_name]))
            classifier = nn.Sequential(*modules)
            self.classifiers[t_name] = classifier

        self.gen_cam_map = self.config["gen_cam_map"]

    def forward(self, input):
        features = self.net(input)
        # if there is an adaptation layer, apply it
        if self.adapt_layer is not None:
            features = self.adapt_layer(features)
        # If attention loss is enabled
        if self.config["enable_attn_loss"]:
            return self.forward_attn_guidance(features, input)
        else:
            # If vit architecture (cannot use CAM)
            if "vit" in self.config["arch"]:
                return self.forward_regular(features)
            else:
                return self.foward_with_cam(features)

    def forward_regular(self, features):
        cls_logits = list()
        for c_name in self.classifiers:
            classifier = self.classifiers[c_name]
            logits = classifier(features)
            cls_logits.append(logits)
        return cls_logits, [], []

    def foward_with_cam(self, features):
        cls_logits = list()
        cls_maps = list()
        for c_name in self.classifiers:
            classifier = self.classifiers[c_name]
            # get CAM maps
            c = self.get_cam_faster(features, classifier)
            cls_maps.append(c)
            cls_logits.append(c.mean(dim=(2, 3)))
        return cls_logits, [], cls_maps

    def forward_attn_guidance(self, features, input):
        cls_logits = list()
        cls_maps = list()
        for c_name in self.classifiers:
            classifier = self.classifiers[c_name]
            # get CAM maps
            c = self.get_cam_faster(features, classifier)
            cls_maps.append(c)
            # attention guidance computes soft weights rather than AVG
            a = torch.softmax(c.reshape(c.shape[0], c.shape[1], -1), dim=2).reshape(
                c.shape
            )
            cls_logits.append((c.contiguous() * a).sum(dim=(2, 3)))

        attn_logits = list()  # []
        for a_name in self.config["attn_tasks"]:
            c_name = self.config["attn_tasks"][a_name]["attn_of"]
            cls_maps_idx = list(self.config["cls_tasks"].keys()).index(c_name)
            cls_map = cls_maps[cls_maps_idx]
            attn_logit = F.interpolate(
                cls_map, input.shape[2:], mode="bicubic", align_corners=True
            )
            attn_logits.append(attn_logit)

        return cls_logits, attn_logits, cls_maps

    def get_cam_faster(self, features, classifier):
        if not self.gen_cam_map:
            return None

        cls_weights = classifier[-1].weight
        cls_bias = classifier[-1].bias

        act_maps = F.conv2d(
            features,
            cls_weights.view(cls_weights.shape[0], cls_weights.shape[1], 1, 1),
            cls_bias,
            stride=1,
            padding=0,
            dilation=1,
            groups=1,
        )

        return act_maps


def get_network(config):
    return Net(config)
