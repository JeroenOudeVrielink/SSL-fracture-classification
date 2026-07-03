import numpy as np
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns


body_parts = [
    "ankle",
    "elbow",
    "femur",
    "finger",
    "foot",
    "forearm",
    "hand",
    "humerus",
    "knee",
    "pelvis_and_hip",
    "shoulder",
    "tibia_fibula",
    "wrist",
]

views = ["bilateral", "left", "right"]


def plot_confusion_matrix(y_true, y_pred, title, save_dir, x_labels, y_labels, view=False):
    conf_matrix = confusion_matrix(y_true, y_pred)
    norm_conf_matrix = (
        conf_matrix.astype("float") / conf_matrix.sum(axis=1)[:, np.newaxis]
    )

    task = "body"
    if view:
        task = "view"

    fig, ax = plt.subplots(figsize=(16, 16))
    sns.heatmap(
        norm_conf_matrix,
        annot=True,
        fmt=".2",
        cmap="Blues",
        xticklabels=x_labels,
        yticklabels=y_labels,
    )

    # plt.title(title)
    plt.suptitle(title, fontsize=32)
    plt.xlabel("Predicted Labels")
    plt.ylabel("True Labels")
    plt.savefig(save_dir / ("conf_matrix_" + task + ".png"))
