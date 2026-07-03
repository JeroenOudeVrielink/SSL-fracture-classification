import pandas as pd
import numpy as np
import json
import os
import random
import cv2
import collections
import matplotlib.pyplot as plt


UNKNOWN_FILES = "/home/jvrielink/data_hdd/AIML_rot_corrected/annotations/merged_classes_age/unknown.pkl"
BODY_PREDS = "results/Gen3_v1_imgnet_pretrained_color_jitter_rotation_lr_schedule_prediction_square_resize_06-05_02:42:14/body_preds.npy"
VIEW_PREDS = "results/Gen3_v1_imgnet_pretrained_color_jitter_rotation_lr_schedule_prediction_square_resize_06-05_02:42:14/view_preds.npy"
BODY_NAMES = "/home/jvrielink/data_hdd/AIML_rot_corrected/annotations/merged_classes_age/body_part_names.json"
VIEW_NAMES = "/home/jvrielink/data_hdd/AIML_rot_corrected/annotations/merged_classes_age/view_names.json"

# Load unknown files and convert to dataframe
unknown_files = pd.read_pickle(UNKNOWN_FILES)
unknown_files = unknown_files.to_frame(name="path").reset_index()

# Get body and view predictions
body_preds = np.load(BODY_PREDS)
view_preds = np.load(VIEW_PREDS)

# Load body names
with open(BODY_NAMES, "r") as json_file:
    body_part_names = json.load(json_file)

with open(VIEW_NAMES, "r") as json_file:
    view_names = json.load(json_file)

# Remove body part prefix
prefix = "body_part_"
new_body_part_names = [
    s[len(prefix) :] for s in body_part_names if s.startswith(prefix)
]
# Remove view prefix
prefix = "view_"
new_view_names = [s[len(prefix) :] for s in view_names if s.startswith(prefix)]

body_preds_str = np.array(new_body_part_names)[np.argmax(body_preds, axis=-1)]
view_preds_str = np.array(new_view_names)[np.argmax(view_preds, axis=-1)]

unknown_files["predicted_body_part"] = body_preds_str
unknown_files["predicted_view"] = view_preds_str

print(unknown_files.head())

unknown_files.to_csv(
    "/home/jvrielink/multi-head-bone-classifier/src/preprocessing/label_files/unknown_files_classified.csv",
    index=False,
)
