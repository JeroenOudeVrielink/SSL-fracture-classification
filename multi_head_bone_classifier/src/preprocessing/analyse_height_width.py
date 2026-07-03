import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns  # For enhanced plot styling
from tqdm import tqdm  # Import tqdm for progress bar
import pandas as pd
from PIL import Image
import os
from torch.utils.data import Dataset
from pathlib import Path
import numpy as np
import matplotlib.ticker as mticker  # Import for tick formatting

RUN_ANALYSIS = True

DATASET_PATH = "/home/jvrielink/data_hdd/AIML_rot_corrected"
ANNOTATIONS_FILE = "annotations/img_paths.pkl"
CSV_DIR = "src/preprocessing/hw_analyses"


class AIMLDataset(Dataset):
    def __init__(
        self,
        annotations_file: str,
        data_path: str,
        transform=None,
        age_prediction=False,
    ):
        self.img_paths_labels = pd.read_pickle(
            os.path.join(data_path, annotations_file)
        )
        self.data_path = data_path
        self.transform = transform
        self.age_prediction = age_prediction

    def __len__(self):
        return len(self.img_paths_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.data_path, self.img_paths_labels.iloc[idx, 0])
        image = Image.open(img_path)

        height, width = image.size
        aspect_ratio = width / height
        return width, height, aspect_ratio


if RUN_ANALYSIS:
    dataset = AIMLDataset(ANNOTATIONS_FILE, DATASET_PATH)
    dataloader = DataLoader(
        dataset, batch_size=256, shuffle=False, num_workers=14
    )  # Set shuffle to False for analysis

    heights = []
    widths = []
    aspect_ratios = []

    for batch in tqdm(dataloader, desc="Analyzing Images"):
        h, w, a = batch  # Extract images (assuming labels are also present)

        heights.extend(h.tolist())
        widths.extend(w.tolist())
        aspect_ratios.extend(a.tolist())  # Calculate aspect ratio

    # Create DataFrame and save to CSV
    data = {"Height": heights, "Width": widths, "Aspect Ratio": aspect_ratios}
    df = pd.DataFrame(data)
    df.to_csv(Path(CSV_DIR) / "image_dimensions.csv", index=False)

else:
    # Load data from CSV
    df = pd.read_csv(Path(CSV_DIR) / "image_dimensions.csv")
    heights = df["Height"]
    widths = df["Width"]
    aspect_ratios = df["Aspect Ratio"]

matplotlib.rcParams["text.usetex"] = True  # Enable LaTeX rendering
matplotlib.rcParams["font.family"] = "serif"

# Plotting histograms with enhanced styling
sns.set_theme(
    style="whitegrid", font_scale=2.0
)  # Set Seaborn theme for professional look

plt.figure(figsize=(15, 5))  # Larger figure size for better readability

axis_font_size = 20
title_font_size = 26


# Height distribution
ax1 = plt.subplot(1, 3, 1)
sns.histplot(heights, bins=20, kde=True, color="skyblue")
plt.xlabel("Height (pixels)", fontsize=axis_font_size)
plt.ylabel("Density", fontsize=axis_font_size)
plt.title("X-ray Height Distribution", fontsize=title_font_size)
# Format y-axis ticks to scientific notation
ax1.yaxis.set_major_formatter(mticker.ScalarFormatter(useMathText=True))
ax1.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))

# Width distribution
ax2 = plt.subplot(1, 3, 2)
sns.histplot(widths, bins=20, kde=True, color="lightcoral")
plt.xlabel("Width (pixels)", fontsize=axis_font_size)
plt.ylabel("Density", fontsize=axis_font_size)
plt.title("X-ray Width Distribution", fontsize=title_font_size)
# Format y-axis ticks to scientific notation
ax2.yaxis.set_major_formatter(mticker.ScalarFormatter(useMathText=True))
ax2.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))

# Aspect ratio distribution
ax3 = plt.subplot(1, 3, 3)
sns.histplot(aspect_ratios, bins=20, kde=True, color="gold")
plt.xlabel("Aspect Ratio (Width / Height)", fontsize=axis_font_size)
plt.ylabel("Density", fontsize=axis_font_size)
plt.title("Aspect Ratio Distribution", fontsize=title_font_size)
# No need to format aspect ratio y-axis
ax3.yaxis.set_major_formatter(mticker.ScalarFormatter(useMathText=True))
ax3.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))

plt.tight_layout()
plt.savefig(
    Path(CSV_DIR) / "image_dimensions_analysis.svg", format="svg", dpi=300
)  # Save high-resolution figure
plt.savefig(
    Path(CSV_DIR) / "image_dimensions_analysis.png", dpi=300
)  # Save high-resolution figure
plt.show()
