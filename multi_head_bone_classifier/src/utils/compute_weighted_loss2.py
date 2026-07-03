import numpy as np
import pandas as pd
import torch


TRAIN_ANNOTATIONS_PATH = (
    "~/data_hdd/AIML_rot_corrected/annotations/merged_classes_age/train.pkl"
)

body_num_classes = 24
view_num_classes = 4
beta = 0.99


def argmax(lst):
    return lst.index(max(lst))


df = pd.read_pickle(TRAIN_ANNOTATIONS_PATH)
df["body_part_int"] = df.body_part_encoded.apply(argmax)
df["view_int"] = df.view_encoded.apply(argmax)

body_samples_per_class = df.body_part_int.value_counts()
view_samples_per_class = df.view_int.value_counts()

body_effective_num = 1.0 - np.power(beta, body_samples_per_class)
body_weights = (1.0 - beta) / np.array(body_effective_num)
body_weights = body_weights / np.sum(body_weights) * body_num_classes

view_effective_num = 1.0 - np.power(beta, view_samples_per_class)
view_weights = (1.0 - beta) / np.array(view_effective_num)
view_weights = view_weights / np.sum(view_weights) * view_num_classes

np.save("src/utils/weights/body_loss_weights2.npy", body_weights)
np.save("src/utils/weights/view_loss_weights2.npy", view_weights)

body_weights_pt = torch.from_numpy(body_weights).float()
view_weights_pt = torch.from_numpy(view_weights).float()

torch.save(body_weights_pt, "src/utils/weights/body_loss_weights3.pt")
torch.save(view_weights_pt, "src/utils/weights/view_loss_weights3.pt")
