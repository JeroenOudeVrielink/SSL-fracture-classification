import json
import numpy as np
import os


def get_sub_dirs(dir):
    sub_dirs = [d for d in os.listdir(dir) if os.path.isdir(os.path.join(dir, d))]
    sub_dirs.sort()
    return sub_dirs


def compute_average_metric(file, epoch="e100", mode="val"):
    master_dict = None
    with open(file) as f:
        master_dict = json.load(f)
    batch_size = master_dict[epoch][mode]["batch_size"]
    avg_accu_clas = None
    if "accu_clas" in master_dict[epoch][mode]:
        accu_clas = master_dict[epoch][mode]["accu_clas"]
        avg_accu_clas = np.average(accu_clas, weights=batch_size)
    avg_accu_fibu = None
    if "accu_fibu" in master_dict[epoch][mode]:
        accu_fibu = master_dict[epoch][mode]["accu_fibu"]
        avg_accu_fibu = np.average(accu_fibu, weights=batch_size)
    return avg_accu_clas, avg_accu_fibu
