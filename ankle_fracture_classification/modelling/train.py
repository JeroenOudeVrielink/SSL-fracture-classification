import argparse
import numpy as np
import os
import torch
import torchvision.transforms.v2 as transforms
from torch.utils.data import DataLoader
from modelling.trainer import Trainer
from modelling.model import get_network
from modelling.dataset import get_loader_from_dataset
from modelling.recorder import Recorder
from toolbox.general_utils import str2bool
from toolbox.save_utils import save_mat
from toolbox.json_utils import save_json
from utils import set_learning_rate

device = "cuda" if torch.cuda.is_available() else "cpu"


def main(args):
    # Disable CAM display for ViT
    args.cam_interval = -1 if "vit" in args.arch else args.cam_interval

    print(args)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    os.makedirs(args.save_path, exist_ok=True)

    # WARNING, IMAGE TRANSFORMS ARE HARDCODED IN DATASET CLASS!
    train_transform = transforms.Compose(
        [
            transforms.Resize((args.input_size, args.input_size)),
            transforms.ToTensor(),
        ]
    )
    val_transform = transforms.Compose(
        [
            transforms.Resize((args.input_size, args.input_size)),
            transforms.ToTensor(),
        ]
    )

    if args.val_type == "aug":
        train_sets = {
            "train",
            "val",
            "ai",  # 'weak_ai', 'weak'
        }
        val_sets = {"test", "weak_test"}
    else:
        raise ValueError("Wrong set definition: {}".format(args.val_type))

    config = {
        "gen_cam_map": args.cam_interval != -1,
        "attn_loss_ver": args.attn_loss_ver,
        "arch": args.arch,
        "imgnet_pretrained": args.imgnet_pretrained,
        "pretrained_method": args.pretrained_method,
        "ckpt_path": args.ckpt_path,
        "enable_attn_loss": args.enable_attn_loss,
        "attn_loss_scalar": args.attn_loss_scalar,
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

    train_data_loader = get_loader_from_dataset(
        data_root=os.path.join(args.data_path, "images"),
        csv_path=os.path.join(args.data_path, "labels", args.master_csv + ".csv"),
        config=config,
        target_translator=None,
        batch_size=args.batch_size,
        transform=train_transform,
        set=train_sets,
        use_random_crop=True,
        shuffle=True,
        num_workers=args.num_workers,
        drop_last=True,
    )

    val_data_loader = get_loader_from_dataset(
        data_root=os.path.join(args.data_path, "images"),
        csv_path=os.path.join(args.data_path, "labels", args.master_csv + ".csv"),
        config=config,
        target_translator=train_data_loader.dataset.target_translator,
        batch_size=args.batch_size,
        transform=val_transform,
        set=val_sets,
        use_random_crop=False,
        shuffle=False,
        num_workers=args.num_workers,
        drop_last=False,
    )

    if args.enable_attn_loss:
        config["attn_tasks"] = {"attn": {"attn_of": "fibula", "num_channels": 4}}
    else:
        config["attn_tasks"] = dict()

    save_json(os.path.join(args.save_path, "config.json"), config)

    net = get_network(config)

    # reload trained model weights from a checkpoint
    if args.reload_from_checkpoint:
        print("Loading from checkpoint: {}".format(args.ckpt_path))
        if os.path.exists(args.ckpt_path):
            net.load_state_dict(torch.load(args.ckpt_path))
        else:
            print("File not exists in the reload path: {}".format(args.ckpt_path))

    if "cuda" in device:
        print("Multiple GPUS.......")
        net = torch.nn.DataParallel(net).to(device)

    params = list(net.parameters())
    trainable_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
    print(f"Trainable parameters: {trainable_params:,}")

    # compute learning rate by: bs / reference_bs * base_lr
    args.learning_rate = (
        args.batch_size / args.reference_batch_size
    ) * args.base_learning_rate
    args.learning_rate_end = (
        args.batch_size / args.reference_batch_size
    ) * args.base_learning_rate_end

    if args.optimizer == "sgd":
        optimizer = torch.optim.SGD(
            net.parameters(), lr=args.learning_rate, weight_decay=5e-4, momentum=0.9
        )
    elif args.optimizer == "adam":
        optimizer = torch.optim.Adam(params, lr=args.learning_rate, weight_decay=5e-4)
    else:
        raise ValueError("Unexpected optimizer: {}".format(args.optimizer))

    # schedule = torch.optim.lr_scheduler.StepLR(
    #     optimizer,
    #     step_size=args.learning_rate_step_size,
    #     gamma=args.learning_rate_gamma,
    # )
    schedule = set_learning_rate
    recorder = Recorder()

    trainer = Trainer(
        net,
        optimizer,
        schedule,
        recorder,
        train_data_loader,
        val_data_loader,
        config,
        args,
        device,
    )

    recorder = trainer.train()
    save_mat(args.save_path, recorder.master_dict)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    torch.autograd.set_detect_anomaly(True)

    # DATA params
    parser.add_argument(
        "--data_path",
        type=str,
        default="dataset/fracture",
        help="the path to data",
    )
    parser.add_argument(
        "--master_csv", type=str, default="master_v5_complete_no_radius_scribble"
    )
    parser.add_argument(
        "--save_path", type=str, default="data/output", help="the path to save results"
    )
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--val_type", type=str, default="aug")
    parser.add_argument("--save_interval", type=int, default=10, help="#epochs")
    parser.add_argument("--cam_interval", type=int, default=10, help="#batches")
    parser.add_argument("--display_interval", type=int, default=10, help="#batches")
    parser.add_argument(
        "--ckpt_path", type=str, default="NA", help="path for trained network"
    )
    parser.add_argument("--reload_from_checkpoint", type=str2bool, default="False")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--save_model_checkpoint", type=str2bool, default="False")

    # TRAINING params
    parser.add_argument("--num_epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--input_size", type=int, default=512)
    parser.add_argument("--optimizer", type=str, default="sgd")
    parser.add_argument("--learning_rate", type=float, default=None)
    parser.add_argument("--learning_rate_end", type=float, default=None)
    parser.add_argument("--base_learning_rate", type=float, default=1e-3)
    parser.add_argument("--base_learning_rate_end", type=float, default=1e-5)
    parser.add_argument("--reference_batch_size", type=int, default=16)
    parser.add_argument(
        "--lr_decay_policy", type=str, default="linear", choices=["exp", "linear"]
    )

    # NETWORK params
    parser.add_argument("--arch", type=str, default="resnet50")
    parser.add_argument("--imgnet_pretrained", type=str2bool, default="True")
    parser.add_argument("--enable_attn_loss", type=str2bool, default="True")
    parser.add_argument("--attn_loss_ver", type=int, default=1)
    parser.add_argument("--attn_loss_scalar", type=float, default=0.5)
    parser.add_argument("--scribble_type", type=str, default="scribble")

    # PRETRAINED params
    parser.add_argument("--pretrained_method", type=str, default=None)

    args = parser.parse_args()
    main(args)
