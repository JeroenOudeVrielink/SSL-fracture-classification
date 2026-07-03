import torch
import numpy as np
import seaborn as sns
import pandas as pd
import umap
import os
import matplotlib.pyplot as plt
from pathlib import Path
import json
from matplotlib.colors import BoundaryNorm, ListedColormap
import matplotlib.cm as cm
import matplotlib.ticker as mticker


LOAD_PATH = Path("visualization/luxry_resnet_features")
BODY_NAMES = "/home/jvrielink/data_hdd/AIML_rot_corrected/annotations/merged_classes_age/body_part_names.json"
VIEW_NAMES = "/home/jvrielink/data_hdd/AIML_rot_corrected/annotations/merged_classes_age/view_names.json"


def get_sub_dirs(dir):
    sub_dirs = [d for d in os.listdir(dir) if os.path.isdir(os.path.join(dir, d))]
    # sub_dirs.sort()
    return sub_dirs


def transform_int_labels_to_str(body_labels, view_labels):
    with open(BODY_NAMES, "r") as json_file:
        body_part_names = json.load(json_file)

    with open(VIEW_NAMES, "r") as json_file:
        view_names = json.load(json_file)

    # Remove body part prefix
    prefix = "body_part_"
    new_body_part_names = [
        s[len(prefix) :] for s in body_part_names if s.startswith(prefix)
    ]
    # Remove view prefix
    prefix = "view_"
    new_view_names = [s[len(prefix) :] for s in view_names if s.startswith(prefix)]

    body_labels_str = np.array(new_body_part_names)[body_labels]
    view_labels_str = np.array(new_view_names)[view_labels]

    return body_labels_str, view_labels_str


def get_categories(body_names, view_names):
    with open(body_names, "r") as json_file:
        body_part_names = json.load(json_file)

    with open(view_names, "r") as json_file:
        view_names = json.load(json_file)

    # Remove body part prefix
    prefix = "body_part_"
    new_body_part_names = [
        s[len(prefix) :] for s in body_part_names if s.startswith(prefix)
    ]
    new_body_part_names = [s.replace("_", " ") for s in new_body_part_names]
    # Remove view prefix
    prefix = "view_"
    new_view_names = [s[len(prefix) :] for s in view_names if s.startswith(prefix)]
    new_view_names = [s.replace("_", " ") for s in new_view_names]

    return new_body_part_names, new_view_names


def load_feature_vectors(path):
    feature_vectors = torch.load(path / "feature_vectors.pt")
    body_labels = torch.load(path / "labels_body.pt")
    view_labels = torch.load(path / "labels_view.pt")
    body_labels = body_labels.argmax(dim=1)
    view_labels = view_labels.argmax(dim=1)
    return feature_vectors.numpy(), body_labels.numpy(), view_labels.numpy()


def plot_umap(feature_vectors, labels, save_path, n_classes, categories):
    reducer = umap.UMAP(n_neighbors=40, min_dist=0.5)
    embedding = reducer.fit_transform(feature_vectors)
    df = pd.DataFrame(embedding, columns=["x", "y"])
    df["label"] = labels

    plt.rcParams["text.usetex"] = True
    plt.rcParams["font.family"] = "serif"
    sns.set_theme(style="whitegrid", font_scale=1.7)  # Adjust font scale as needed

    axis_font_size = 16  # Adjust axis font size

    fig, ax = plt.subplots(figsize=(7.5, 5.5))

    cax = ax.scatter(df.x, df.y, c=df.label, cmap="Spectral", s=5)
    # plt.gca().set_aspect("equal", "datalim")
    cbar = fig.colorbar(
        cax,
        ticks=np.arange(n_classes),
        format=mticker.FixedFormatter(categories),
        boundaries=np.arange(n_classes + 1) - 0.5,
    )
    cbar.ax.tick_params(labelsize=axis_font_size)

    plt.xlabel("Projected dimension 1", fontsize=axis_font_size)
    plt.ylabel("Projected dimension 2", fontsize=axis_font_size)
    plt.savefig(
        save_path.parent / (save_path.name + ".png"),
        dpi=300,
    )  # High DPI for print quality
    plt.savefig(
        save_path.parent / (save_path.name + ".svg"),
        format="svg",
        dpi=300,
    )  # SVG for scalability
    plt.close()


if __name__ == "__main__":
    sub_dirs = get_sub_dirs(LOAD_PATH)
    # sub_dirs = ["dinov1"]
    for sub_dir in sub_dirs:
        print("Plotting UMAP for: ", sub_dir)
        feature_vectors, body_labels, view_labels = load_feature_vectors(
            LOAD_PATH / sub_dir
        )
        # body_labels, view_labels = transform_int_labels_to_str(body_labels, view_labels)
        body_tick_labels, view_tick_labels = get_categories(BODY_NAMES, VIEW_NAMES)
        plot_umap(
            feature_vectors,
            body_labels,
            LOAD_PATH / sub_dir / f"{sub_dir}_umap_body",
            n_classes=13,
            categories=body_tick_labels,
        )
        plot_umap(
            feature_vectors,
            view_labels,
            LOAD_PATH / sub_dir / f"{sub_dir}_umap_view",
            n_classes=3,
            categories=view_tick_labels,
        )
