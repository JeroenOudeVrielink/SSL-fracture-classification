import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from calc_utils import compute_average_metric, get_sub_dirs
import re

sns.set_style("darkgrid")

RESULTS_DIRS = [
    # "/home/jvrielink/data_hdd/data_to_keep/phase_2/mocov3_01-16_23:50:25",
    # "data/moco_v2_bs512_base_params_03-26_02:05:48",
    # "/home/jvrielink/data_hdd/data_to_keep/phase_2/spark_01-16_23:59:31",
    # "data/spark_in224_bs64_weighted_masking_04-03_08:09:40",
    "data/vicregl_bs128_alpha1_03-26_13:58:18",
    "/home/jvrielink/data_hdd/data_to_keep/phase_2/vicregl_01-17_00:05:49",
]

Y_LABELS = [
    # "super_fmap_v2",
    # "dino_no_lg_no_mc",
    # "dino_no_aug_smooth",
    # "spark_laplace_recon",
    # "dino_no_aug",
    # "dino_no_lg",
    # "dino_no_mc",
    # "DINOv1",
    "VicRegL",
    "VicRegL_a1.0",
    # "SparK",
    # "SparK_wm",
    # "MoCo_v3",
    # "MoCo_v2",
]
SAVE_FILE_PATH = "data/vicregl_vs_vicregl_a1.png"

# RESULTS_DIRS = [
#     "data_to_keep/model_per_epoch_baseline_11-06_02:02:18",
#     "data_to_keep/model_per_epoch_lr05_11-09_00:52:15",
# ]
# Y_LABELS = ["pretrained_baseline", "pretrained_lr05"]
# SAVE_FILE_PATH = "data_to_keep/model_per_epoch.png"


CLAS_BASELINE = 0.636
CLAS_BASELINE_LR05 = 0.7279999996821086


def plot_per_epoch_together(
    results_dirs, y_labels, clas_baseline, clas_baseline_lr05, save_file_path
):
    # For every results dir
    plt.axhline(y=clas_baseline, label="baseline", color="red")
    # plt.axhline(y=clas_baseline_lr05, label="baseline_lr05", color="green")
    plt.xlabel("N pretrain epochs")
    plt.ylabel("Accuracy clas")
    plt.title("Test clas accuracy after 100 train epochs per pretrained epoch")

    for i, result_dir in enumerate(results_dirs):
        x = np.load(os.path.join(result_dir, "x.npy"))
        accuracies_clas = np.load(os.path.join(result_dir, "accu_clas.npy"))
        plt.plot(x, accuracies_clas, label=y_labels[i], marker=".")

    plt.legend()
    plt.savefig(save_file_path)


if __name__ == "__main__":
    plot_per_epoch_together(
        RESULTS_DIRS, Y_LABELS, CLAS_BASELINE, CLAS_BASELINE_LR05, SAVE_FILE_PATH
    )
