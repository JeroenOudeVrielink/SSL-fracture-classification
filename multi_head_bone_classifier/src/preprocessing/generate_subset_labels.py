import pandas as pd
import numpy as np
import json
import os
import random
import cv2
import collections
import matplotlib.pyplot as plt


UNKNOWN_FILES = "/home/jvrielink/data_hdd/AIML_rot_corrected/annotations/merged_classes_age/unknown.pkl"
KNOWN_FILES = "/home/jvrielink/data_hdd/AIML_rot_corrected/annotations/merged_classes_age/all_labels_as.csv"
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

known_files = pd.read_csv(KNOWN_FILES)
print(known_files)


body_preds_str = np.array(new_body_part_names)[np.argmax(body_preds, axis=-1)]
view_preds_str = np.array(new_view_names)[np.argmax(view_preds, axis=-1)]

# unknown_files["body_part_encoded"] = body_preds
# unknown_files["view_encoded"] = view_preds

unknown_files["body_part"] = body_preds_str
unknown_files["view"] = view_preds_str
# print(unknown_files.head())

keys = ["ankle", "foot", "tibia_fibula"]
# keys = ["hip", "shoulder"]
known_filtered = known_files[known_files["body_part"].isin(keys)]
unknown_filtered = unknown_files[unknown_files["body_part"].isin(keys)]

merged_df = known_filtered[["path"]]
# print(merged_df)

print(merged_df.iloc[0, -1])
merged_df.to_csv(
    "/home/jvrielink/data_hdd/AIML_rot_corrected/annotations/img_paths_subset.csv"
)
print(len(merged_df))


# image_paths = unknown_filtered["path"].tolist()
# keys = ["ankle"]
# f = unknown_files[unknown_files["body_part"].isin(keys)]
image_paths = unknown_files["path"].tolist()
body_parts = unknown_files["body_part"].tolist()
views = unknown_files["view"].tolist()

combined = list(zip(image_paths, body_parts, views))
random.shuffle(combined)
image_paths, body_parts, views = zip(*combined)

# Count the frequency of each string
counter = collections.Counter(body_parts)

# Get the unique strings and their frequencies
labels, values = zip(*counter.items())

# Create the histogram
plt.bar(labels, values)

# Add labels and title
plt.xlabel("String")
plt.ylabel("Frequency")
plt.title("Histogram of Strings")

# Show the plot
plt.show()

image_paths = known_filtered["path"].tolist()
body_parts = known_filtered["body_part"].tolist()
views = known_filtered["view"].tolist()


# for i, path in enumerate(image_paths):
for path, body_str, view_str in zip(image_paths, body_parts, views):
    image = cv2.imread("/home/jvrielink/data_hdd/AIML_rot_corrected/" + path)

    # body_idx = np.argmax(body_preds[i])
    # view_idx = np.argmax(view_preds[i])
    # body_str = new_body_part_names[body_idx]
    # view_str = new_view_names[view_idx]
    print(f"Body: {body_str}, View: {view_str}")

    if image is None:
        print(f"Error loading image: {image_paths[current_index]}")
        break

    # Resize to 1/4 resolution
    resized_image = cv2.resize(image, (image.shape[1] // 4, image.shape[0] // 4))

    cv2.imshow("Image Viewer", resized_image)  # Display resized image

    key = cv2.waitKey(0)  # Wait for a key press

    if key == ord("q"):  # Press 'q' to quit
        break
    elif key == ord("n"):  # Press 'n' for next image
        current_index = (current_index + 1) % len(image_paths)

    os.system("cls" if os.name == "nt" else "clear")


cv2.destroyAllWindows()
