from torch import nn
import resnet
import argparse
import torch
from pathlib import Path
from pretrained.pretrained_resnet import preprocess_state_dict


def get_arguments():
    parser = argparse.ArgumentParser(
        description="Pretraining with VICRegL", add_help=False
    )

    # Checkpoints and Logs
    parser.add_argument("--exp-dir", type=Path, required=False)
    parser.add_argument("--log-tensors-interval", type=int, default=60)
    parser.add_argument("--checkpoint-freq", type=int, default=1)

    # Data
    parser.add_argument("--dataset", type=str, default="imagenet1k")
    parser.add_argument("--dataset_from_numpy", action="store_true")
    parser.add_argument("--size-crops", type=int, nargs="+", default=[224, 96])
    parser.add_argument("--num-crops", type=int, nargs="+", default=[2, 6])
    parser.add_argument("--min_scale_crops", type=float, nargs="+", default=[0.4, 0.08])
    parser.add_argument("--max_scale_crops", type=float, nargs="+", default=[1, 0.4])
    parser.add_argument("--no-flip-grid", type=int, default=1)

    # Model
    parser.add_argument("--arch", type=str, default="resnet50")
    parser.add_argument("--drop-path-rate", type=float, default=0.1)
    parser.add_argument("--layer-scale-init-value", type=float, default=0.0)
    parser.add_argument("--mlp", default="8192-8192-8192")
    parser.add_argument("--maps-mlp", default="512-512-512")

    # Loss Function
    parser.add_argument("--alpha", type=float, default=0.75)
    parser.add_argument(
        "--num_matches",
        type=int,
        nargs="+",
        default=[20, 4],
        help="Number of spatial matches in a feature map",
    )
    parser.add_argument("--l2_all_matches", type=int, default=0)
    parser.add_argument("--inv-coeff", type=float, default=25.0)
    parser.add_argument("--var-coeff", type=float, default=25.0)
    parser.add_argument("--cov-coeff", type=float, default=1.0)
    parser.add_argument("--fast-vc-reg", type=int, default=0)

    # Optimization
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--warmup-epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--optimizer", default="adamw")
    parser.add_argument("--base-lr", type=float, default=0.0005)
    parser.add_argument("--end-lr-ratio", type=float, default=0.001)
    parser.add_argument("--weight-decay", type=float, default=0.05)

    # Evaluation
    parser.add_argument("--val-batch-size", type=int, default=-1)
    parser.add_argument("--evaluate", action="store_true")
    parser.add_argument("--evaluate-only", action="store_true")
    parser.add_argument("--eval-freq", type=int, default=10)
    parser.add_argument("--maps-lr-ratio", type=float, default=0.1)

    # Running
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--num-workers", type=int, default=10)
    parser.add_argument(
        "--device", default="cuda", help="device to use for training / testing"
    )

    # Distributed
    parser.add_argument(
        "--world-size", default=1, type=int, help="number of distributed processes"
    )
    parser.add_argument("--local_rank", default=-1, type=int)
    parser.add_argument(
        "--dist-url", default="env://", help="url used to set up distributed training"
    )

    # Own
    parser.add_argument("--data_path", type=str, required=False)
    parser.add_argument(
        "--annotations_file", type=str, default="annotations/img_paths.csv"
    )
    parser.add_argument("--datetime", type=str, default="")
    parser.add_argument("--exp_name", type=str, required=False)
    parser.add_argument("--num_classes", type=int, default=1000)

    return parser


def MLP(mlp, embedding, norm_layer):
    mlp_spec = f"{embedding}-{mlp}"
    layers = []
    f = list(map(int, mlp_spec.split("-")))
    for i in range(len(f) - 2):
        layers.append(nn.Linear(f[i], f[i + 1]))
        if norm_layer == "batch_norm":
            layers.append(nn.BatchNorm1d(f[i + 1]))
        elif norm_layer == "layer_norm":
            layers.append(nn.LayerNorm(f[i + 1]))
        layers.append(nn.ReLU(True))
    layers.append(nn.Linear(f[-2], f[-1], bias=False))
    return nn.Sequential(*layers)


class VICRegL(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.embedding_dim = int(args.mlp.split("-")[-1])

        self.backbone, self.representation_dim = resnet.__dict__[args.arch](
            zero_init_residual=True
        )
        norm_layer = "batch_norm"

        if self.args.alpha < 1.0:
            self.maps_projector = MLP(
                args.maps_mlp, self.representation_dim, norm_layer
            )

        if self.args.alpha > 0.0:
            self.projector = MLP(args.mlp, self.representation_dim, norm_layer)

        self.classifier = nn.Linear(self.representation_dim, self.args.num_classes)

    def forward_networks(self, inputs, is_val):
        outputs = {
            "representation": [],
            "embedding": [],
            "maps_embedding": [],
            "logits": [],
            "logits_val": [],
        }
        for x in inputs["views"]:
            maps, representation = self.backbone(x)
            outputs["representation"].append(representation)

            if self.args.alpha > 0.0:
                embedding = self.projector(representation)
                outputs["embedding"].append(embedding)

            if self.args.alpha < 1.0:
                batch_size, num_loc, _ = maps.shape
                maps_embedding = self.maps_projector(
                    maps.flatten(start_dim=0, end_dim=1)
                )
                maps_embedding = maps_embedding.view(batch_size, num_loc, -1)
                outputs["maps_embedding"].append(maps_embedding)

            logits = self.classifier(representation.detach())
            outputs["logits"].append(logits)

        if is_val:
            _, representation = self.backbone(inputs["val_view"])
            val_logits = self.classifier(representation.detach())
            outputs["logits_val"].append(val_logits)

        return outputs

    def forward(self, x, is_val=False, backbone_only=False):
        # outputs = self.forward_networks(x, is_val)
        maps, representation = self.backbone(x)
        embedding = self.projector(representation)
        return embedding


def exclude_bias_and_norm(p):
    return p.ndim == 1


def get_pretrained_vicreg(model_path, imgnet=False):
    parser = argparse.ArgumentParser(
        "Pretraining with VICRegL", parents=[get_arguments()]
    )
    args = parser.parse_args()
    args.alpha = 1.0

    model = VICRegL(args)
    if imgnet:
        model.classifier = nn.Identity()
        import __main__

        setattr(__main__, "exclude_bias_and_norm", exclude_bias_and_norm)
    checkpoint = torch.load(model_path, "cpu")
    state_dict = checkpoint["model"]
    if imgnet:
        state_dict = preprocess_state_dict(
            state_dict, replace="module.", replace_with=""
        )

    model.load_state_dict(state_dict)
    return model


def get_pretrained_vicregl(model_path, imgnet=False):
    parser = argparse.ArgumentParser(
        "Pretraining with VICRegL", parents=[get_arguments()]
    )
    args = parser.parse_args()
    model = VICRegL(args)
    if imgnet:
        import __main__

        setattr(__main__, "exclude_bias_and_norm", exclude_bias_and_norm)
    checkpoint = torch.load(model_path, "cpu")
    state_dict = checkpoint["model"]
    if imgnet:
        state_dict = preprocess_state_dict(
            state_dict, replace="module.", replace_with=""
        )

    model.load_state_dict(state_dict)
    return model


if __name__ == "__main__":
    model = get_pretrained_vicregl(
        "/home/jvrielink/Downloads/resnet50_alpha0.75_fullckpt.pth"
    )
