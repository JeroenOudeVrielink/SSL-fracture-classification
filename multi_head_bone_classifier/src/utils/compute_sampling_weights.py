import pandas as pd
import numpy as np
import torch


TRAIN_ANNOTATIONS_PATH = (
    "~/data_hdd/AIML_rot_corrected/annotations/merged_classes_age/train.pkl"
)


def argmax(lst):
    return lst.index(max(lst))


df = pd.read_pickle(TRAIN_ANNOTATIONS_PATH)
df["body_part_int"] = df.body_part_encoded.apply(argmax)

body_counts = df.body_part_int.value_counts()
body_sample_weights = [1 / body_counts[i] for i in df.body_part_int.values]
body_sample_weights = np.array(body_sample_weights)
np.save("src/utils/weights/body_sample_weights.npy", body_sample_weights)

body_sample_weights_pt = torch.from_numpy(body_sample_weights).float()
torch.save(body_sample_weights_pt, "src/utils/weights/body_sample_weights.pt")
