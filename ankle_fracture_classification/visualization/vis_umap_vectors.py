import torch
import numpy as np
import seaborn as sns
import pandas as pd
import umap
import os
import matplotlib.pyplot as plt
from pathlib import Path

sns.set_style("whitegrid")

LOAD_PATH = Path(
    "/home/jvrielink/Thesis/fracture-attention-guidance/visualization/finetuned_feature_vectors"
)


def get_sub_dirs(dir):
    sub_dirs = [d for d in os.listdir(dir) if os.path.isdir(os.path.join(dir, d))]
    # sub_dirs.sort()
    return sub_dirs


def load_feature_vectors(path):
    feature_vectors = torch.load(path / "feature_vectors.pt")
    labels = torch.load(path / "labels.pt")
    labels = labels.argmax(dim=1)
    return feature_vectors.numpy(), labels.numpy()


def plot_umap(feature_vectors, labels, save_path, sub_dir):
    reducer = umap.UMAP(n_neighbors=30, min_dist=0.3)
    embedding = reducer.fit_transform(feature_vectors)
    df = pd.DataFrame(embedding, columns=["x", "y"])
    df["label"] = labels
    plt.scatter(
        embedding[:, 0],
        embedding[:, 1],
        # label=df.label,
        c=df.label,
        cmap="Spectral",
        s=5,
    )
    plt.gca().set_aspect("equal", "datalim")
    plt.title(f"UMAP projection of {sub_dir} embedding", fontsize=16)
    plt.colorbar(boundaries=np.arange(5) - 0.5).set_ticks(np.arange(4))
    plt.xlabel("X")
    plt.ylabel("Y")
    # plt.legend()
    plt.savefig(save_path)
    plt.close()


if __name__ == "__main__":
    sub_dirs = get_sub_dirs(LOAD_PATH)
    for sub_dir in sub_dirs:
        print("Plotting UMAP for: ", sub_dir)
        feature_vectors, labels = load_feature_vectors(LOAD_PATH / sub_dir)
        plot_umap(
            feature_vectors,
            labels,
            LOAD_PATH / sub_dir / f"{sub_dir}_umap.png",
            sub_dir,
        )
        print("Done plotting UMAP for: ", sub_dir)
