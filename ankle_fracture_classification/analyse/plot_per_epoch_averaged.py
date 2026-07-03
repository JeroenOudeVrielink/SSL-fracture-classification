import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from calc_utils import compute_average_metric, get_sub_dirs
import re

sns.set_style("darkgrid")

# RESULTS_DIRS = [
#     "data_to_keep/model_per_epoch_baseline_11-06_02:02:18",
#     "data_to_keep/model_per_epoch_lr05_11-09_00:52:15",
# ]
# Y_LABELS = ["pretrained_baseline", "pretrained_lr05"]
RESULTS_DIRS = [
    "/home/jvrielink/fracture-attention-guidance/data/test/Gen3_v1_imgnet_pretrained_color_jitter_rotation_lr_schedule_04-11_07:41:53",
]
Y_LABELS = [
    "test"
    # "SparK_wm",
    # "MoCo_v2",
    # "VicRegL_a1.0",
    # "dino_no_lg_no_mc",
    # "dino_no_aug_smooth",
    # "spark_laplace_recon",
    #     "dino_no_aug",
    #     "dino_no_lg",
    #     "dino_no_mc",
]


CLAS_BASELINE = 0.636
CLAS_BASELINE_LR05 = 0.7279999996821086


# def get_epochs_from_file_names(dirs):
#     x = []
#     for dir in dirs:
#         split = dir.split("_")
#         res = [i for i in split if "epoch" in i]
#         numbers = [int(s) for s in re.findall(r"\b\d+\b", res[0])]
#         x.append(numbers[0] + 1)
#     return x


def get_epochs_from_file_names(dirs):
    x = []
    for dir in dirs:
        split = dir.split("_")
        res = [i for i in split if "epoch" in i]
        if len(res) == 0:
            res = split[-1]
        pattern = r"\d+"
        match = re.search(pattern, res[0])
        x.append(int(match.group()))
    x.sort()
    return x


def make_plot(
    x, y, b1, b2, xlabel, ylabel, y_label, b1_label, b2_label, title, save_path
):
    plt.axhline(y=b1, label=b1_label, color="red")
    # plt.axhline(y=b2, label=b2_label, color="green")
    plt.plot(x, y, label=y_label, marker=".")
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.title(title)
    plt.savefig(save_path)
    # plt.show()
    plt.close()


def plot_per_epoch_avg(results_dirs, y_labels, clas_baseline, clas_baseline_lr05):
    # For every results dir
    for i, result_dir in enumerate(results_dirs):
        print(f"{i}: Computing {result_dir} ...")
        seed_dirs = get_sub_dirs(result_dir)

        x = get_epochs_from_file_names(
            get_sub_dirs(os.path.join(result_dir, seed_dirs[0]))
        )
        accuracies_fibu = np.array([0.0] * len(x))

        # For every seed dir
        for seed_dir in seed_dirs:
            run_dirs = get_sub_dirs(os.path.join(result_dir, seed_dir))
            # Go over each run and sum it over the number of seeds
            for j, run_dir in enumerate(run_dirs):
                file = os.path.join(result_dir, seed_dir, run_dir, "master_dict.json")
                _, accu_fibu = compute_average_metric(file)
                accuracies_fibu[j] += accu_fibu
        accuracies_fibu /= len(seed_dirs)

        make_plot(
            x=x,
            y=accuracies_fibu,
            b1=clas_baseline,
            b2=clas_baseline_lr05,
            xlabel="N pretrain epochs",
            ylabel="Accuracy clas",
            y_label=y_labels[i],
            b1_label="baseline",
            b2_label="baseline_lr05",
            title="Test clas accuracy after 100 train epochs per pretrained epoch",
            save_path=os.path.join(result_dir, "accu_clas.png"),
        )
        np.save(os.path.join(result_dir, "accu_clas.npy"), accuracies_fibu)
        np.save(os.path.join(result_dir, "x.npy"), x)


if __name__ == "__main__":
    plot_per_epoch_avg(RESULTS_DIRS, Y_LABELS, CLAS_BASELINE, CLAS_BASELINE_LR05)
