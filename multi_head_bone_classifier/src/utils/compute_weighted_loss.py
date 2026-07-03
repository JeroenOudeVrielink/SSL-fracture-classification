import pandas as pd
import numpy as np


max_body_samples = 30907
max_view_samples = 57453


def compute_weight_1(x):
    w = 0
    if x < 100:
        w = 10
    elif x < 1000:
        w = 5
    elif x < 10000:
        w = 2
    else:
        w = 1
    return w


def compute_weight_2(x, view=False):
    if view:
        max_samples = max_body_samples
    else:
        max_samples = max_view_samples
    w = np.sqrt(max_samples / x)
    return w


df = pd.read_pickle("/mnt/sdb1/Data_remote/AIML_half_size/train.pkl")
body_freq = df["body_part_encoded"].value_counts()
df_body_counts = body_freq.to_frame()
df_body_sorted = df_body_counts.sort_values(by=["body_part_encoded"], ascending=False)
np.save(
    "src/utils/body_samples_per_class.npy",
    df_body_sorted["count"].to_numpy(),
)

df_body_sorted["weight1"] = df_body_sorted["count"].apply(compute_weight_1)
df_body_sorted["weight2"] = df_body_sorted["count"].apply(compute_weight_2)
print(df_body_sorted)

body_weights = df_body_sorted["weight2"].to_numpy()
np.save("src/utils/body_weights.npy", body_weights)

view_freq = df["view_encoded"].value_counts()
df_view_counts = view_freq.to_frame()
df_view_sorted = df_view_counts.sort_values(by=["view_encoded"], ascending=False)
np.save(
    "src/utils/view_samples_per_class.npy",
    df_view_sorted["count"].to_numpy(),
)

df_view_sorted["weight1"] = df_view_sorted["count"].apply(compute_weight_1)
df_view_sorted["weight2"] = df_view_sorted["count"].apply(compute_weight_2)
print(df_view_sorted)

view_weights = df_view_sorted["weight2"].to_numpy()
np.save("src/utils/view_weights.npy", view_weights)
