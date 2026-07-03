from calc_utils import get_sub_dirs, compute_average_metric
import os
import numpy as np

RESULTS_DIRS = [
    "data/baseline_08:03:09",
    "data/baseline_convnext_base_in224_04-24_12:26:50",
    "data/baseline_in224_04-24_16:27:12",
    "data/baseline_in224_04-24_20:24:49",
    "data/baseline_vit_b_16_04-25_00:20:05",
]
EXTRA_DEPTH = [False, False, False, False, False]


def lst_avg(lst):
    return sum(lst) / len(lst)


def comp_and_write(f, sub_dir, results_dir):
    file = os.path.join(results_dir, sub_dir, "master_dict.json")
    accu_clas, accu_fibu = compute_average_metric(file)
    f.write(f"{sub_dir}: accu_clas:{accu_clas}, accu_fibu:{accu_fibu}\n")
    return accu_clas, accu_fibu


def compute_metrics_and_write(results_dir, extra_depth=False):
    sub_dirs = get_sub_dirs(results_dir)
    sub_dirs.sort()

    accuracies_clas = []
    accuracies_fibu = []
    f = open(os.path.join(results_dir, "metrics_summary.txt"), "w")
    for sub_dir in sub_dirs:
        if extra_depth:
            sub_sub_dirs = get_sub_dirs(os.path.join(results_dir, sub_dir))
            sub_sub_dir = sub_sub_dirs[0]
            accu_clas, accu_fibu = comp_and_write(
                f, os.path.join(sub_dir, sub_sub_dir), results_dir
            )
        else:
            accu_clas, accu_fibu = comp_and_write(f, sub_dir, results_dir)
        accuracies_clas.append(accu_clas)
        accuracies_fibu.append(accu_fibu)

    avg_accu_clas = None
    std_accu_clas = None
    avg_accu_fibu = None
    std_accu_fibu = None
    if None not in accuracies_clas:
        avg_accu_clas = np.average(accuracies_clas)
        std_accu_clas = np.std(accuracies_clas)
    if None not in accuracies_fibu:
        avg_accu_fibu = np.average(accuracies_fibu)
        std_accu_fibu = np.std(accuracies_fibu)
    f.write("\n")
    f.write(f"average accu_clas: {avg_accu_clas}\n")
    f.write(f"Std. dev. accu_clas: {std_accu_clas}\n")
    f.write(f"average accu_fibu: {avg_accu_fibu}\n")
    f.write(f"Std. dev. accu_fibu: {std_accu_fibu}\n")
    f.close()


if __name__ == "__main__":
    for idx, dir in enumerate(RESULTS_DIRS):
        print(f"{idx}: Computing {dir} ...")
        compute_metrics_and_write(dir, extra_depth=EXTRA_DEPTH[idx])
