from calc_utils import get_sub_dirs, compute_average_metric
import os
import numpy as np

RESULTS_DIRS = [
    "data/dinov1_bs128_ep1000_06-22_07:49:45",
]


def lst_avg(lst):
    return sum(lst) / len(lst)


def avg_acc_nan(accs):
    avg_accs = []
    for acc in np.transpose(accs):
        count = 0
        acc_sum = 0
        for a in acc:
            if a is not None:
                acc_sum += a
                count += 1
            else:
                pass
        avg_accs.append(acc_sum / count)
    return avg_accs


def comp_and_write(f, sub_dir, results_dir):
    try:
        file = os.path.join(results_dir, sub_dir, "master_dict.json")
        accu_clas, accu_fibu = compute_average_metric(file)
    except:
        print(f"Error: {results_dir}/{sub_dir}/master_dict.json not found")
        accu_clas = None
        accu_fibu = None

    f.write(f"{sub_dir}:\n accu_clas:{accu_clas}, accu_fibu:{accu_fibu}\n")
    return accu_clas, accu_fibu


def write_in_copy_format(f, avg_acc_clas, std_acc_clas, avg_acc_fibu, std_acc_fibu):
    f.write("avg_acc_clas:\n")
    for value in avg_acc_clas:
        f.write(f"{value}\n")

    f.write("\n")
    f.write("std_acc_clas:\n")
    for value in std_acc_clas:
        f.write(f"{value}\n")

    f.write("\n")
    f.write("avg_acc_fibu:\n")
    for value in avg_acc_fibu:
        f.write(f"{value}\n")

    f.write("\n")
    f.write("std_acc_fibu:\n")
    for value in std_acc_fibu:
        f.write(f"{value}\n")


def compute_metrics_and_write(results_dir):
    seed_dirs = get_sub_dirs(results_dir)
    seed_dirs.sort()

    n_seed_dirs = len(seed_dirs)
    run_dirs = get_sub_dirs(os.path.join(results_dir, seed_dirs[0]))
    n_run_dirs = len(run_dirs)

    accuracies_clas = np.zeros((n_seed_dirs, n_run_dirs))
    accuracies_fibu = np.zeros((n_seed_dirs, n_run_dirs))
    f = open(os.path.join(results_dir, "metrics_summary.txt"), "w")
    for i, seed_dir in enumerate(seed_dirs):
        result_seed_dir_path = os.path.join(results_dir, seed_dir)
        run_dirs = get_sub_dirs(result_seed_dir_path)
        run_dirs.sort()
        for j, run_dir in enumerate(run_dirs):
            accu_clas, accu_fibu = comp_and_write(f, run_dir, result_seed_dir_path)
            accuracies_clas[i, j] = accu_clas
            accuracies_fibu[i, j] = accu_fibu

    avg_accu_clas = np.nanmean(accuracies_clas, axis=0, keepdims=False)
    std_accu_clas = np.nanstd(accuracies_clas, axis=0, keepdims=False)
    avg_accu_fibu = np.nanmean(accuracies_fibu, axis=0, keepdims=False)
    std_accu_fibu = np.nanstd(accuracies_fibu, axis=0, keepdims=False)
    for i, run_dir in enumerate(run_dirs):
        f.write("\n")
        f.write(f"{run_dir}\n")
        f.write("(Seed not accurate, values listed below is a mean of all seeds)\n")
        f.write(f"avg accu_clas:{avg_accu_clas[i]}\n")
        f.write(f"std_clas:{std_accu_clas[i]}\n")
        f.write(f"avg accu_fibu:{avg_accu_fibu[i]}\n")
        f.write(f"std_fibu:{std_accu_fibu[i]}\n")

    write_in_copy_format(f, avg_accu_clas, std_accu_clas, avg_accu_fibu, std_accu_fibu)
    f.close()


if __name__ == "__main__":
    for idx, dir in enumerate(RESULTS_DIRS):
        print(f"{idx}: Computing {dir} ...")
        compute_metrics_and_write(dir)
