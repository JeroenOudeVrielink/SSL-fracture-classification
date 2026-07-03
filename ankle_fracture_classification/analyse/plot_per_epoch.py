import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from calc_utils import compute_average_metric, get_sub_dirs

sns.set_style("darkgrid")

RESULTS_DIR = "data/model_per_epoch_10-30_06:31:52"
CLAS_BASELINE = 0.6559999998410542
CLAS_BASELINE_LR05 = 0.5873333334128061
FIBU_BASELINE = 0.60173619333903
FIBU_BASELINE_LR05 = 0.5840226253668468


def make_plot(
    x, y, b1, b2, xlabel, ylabel, y_label, b1_label, b2_label, title, save_path
):
    plt.axhline(y=b1, label=b1_label, color="red")
    plt.axhline(y=b2, label=b2_label, color="green")
    plt.plot(x, y, label=y_label)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.title(title)
    plt.savefig(save_path)
    # plt.show()
    plt.close()


def plot_model_per_epoch(results_dir):
    sub_dirs = get_sub_dirs(results_dir)
    sub_dirs.sort()

    accuracies_clas = []
    accuracies_fibu = []
    for sub_dir in sub_dirs:
        file = os.path.join(results_dir, sub_dir, "master_dict.json")
        accu_clas, accu_fibu = compute_average_metric(file)
        accuracies_clas.append(accu_clas)
        accuracies_fibu.append(accu_fibu)

    x = np.arange(0, len(sub_dirs) * 5, 5)
    make_plot(
        x=x,
        y=accuracies_clas,
        b1=CLAS_BASELINE,
        b2=CLAS_BASELINE_LR05,
        xlabel="N pretrain epochs",
        ylabel="Accuracy clas",
        y_label="pretrained_lr05",
        b1_label="baseline",
        b2_label="baseline_lr05",
        title="Test clas accuracy after 100 train epochs per pretrained epoch",
        save_path=os.path.join(results_dir, "accu_clas.png"),
    )
    make_plot(
        x=x,
        y=accuracies_fibu,
        b1=FIBU_BASELINE,
        b2=FIBU_BASELINE_LR05,
        xlabel="N pretrain epochs",
        ylabel="Accuracy fibu",
        y_label="pretrained_lr05",
        b1_label="baseline",
        b2_label="baseline_lr05",
        title="Test fibu accuracy after 100 train epochs per pretrained epoch",
        save_path=os.path.join(results_dir, "accu_fibu.png"),
    )


if __name__ == "__main__":
    plot_model_per_epoch(RESULTS_DIR)
