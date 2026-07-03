import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.ticker as mticker  # Import for tick formatting


# File paths (replace with your actual paths)
csv_file_path = "/home/jvrielink/data_hdd/AIML_rot_corrected/annotations/merged_classes_age/all_labels_as.csv"
output_figure_path = "src/preprocessing/data_distribution/distribution_plots.png"

# Read the CSV file
df = pd.read_csv(csv_file_path)

# Calculate frequency counts
body_part_counts = df["body_part"].value_counts()
view_counts = df["view"].value_counts()

# Apply the styles
plt.rcParams["text.usetex"] = True
plt.rcParams["font.family"] = "serif"
sns.set_theme(style="whitegrid", font_scale=2.2)  # Adjust font scale as needed

# Create figure
fig, axs = plt.subplots(1, 3, figsize=(20, 6))  # Adjust figure size
axis_font_size = 26  # Adjust axis font size
title_font_size = 32  # Adjust title font size

# Body Part Distribution
sns.barplot(
    x=body_part_counts.index, y=body_part_counts.values, color="skyblue", ax=axs[0]
)
axs[0].set_title("Body Part Distribution", fontsize=title_font_size)
axs[0].set_xlabel("Body Part", fontsize=axis_font_size)
axs[0].set_ylabel("Frequency", fontsize=axis_font_size)
plt.setp(
    axs[0].get_xticklabels(), rotation=45, ha="right", fontsize=axis_font_size
)  # Rotate x-labels
axs[0].ticklabel_format(axis="y", style="sci", scilimits=(0, 0))


# View Distribution
sns.barplot(x=view_counts.index, y=view_counts.values, color="lightcoral", ax=axs[1])
axs[1].set_title("View Distribution", fontsize=title_font_size)
axs[1].set_xlabel("View", fontsize=axis_font_size)
axs[1].set_ylabel("Frequency", fontsize=axis_font_size)
plt.setp(axs[1].get_xticklabels(), rotation=45, ha="right", fontsize=axis_font_size)
axs[1].ticklabel_format(axis="y", style="sci", scilimits=(0, 0))


# Age Scalar Distribution
sns.histplot(df["age_scalar"], bins=10, kde=True, color="gold", ax=axs[2])
axs[2].set_title("Age Scalar Distribution", fontsize=title_font_size)
axs[2].set_xlabel("Age Scalar", fontsize=axis_font_size)
axs[2].set_ylabel("Density", fontsize=axis_font_size)
axs[2].yaxis.set_major_formatter(mticker.ScalarFormatter(useMathText=True))
axs[2].ticklabel_format(axis="y", style="sci", scilimits=(0, 0))

# Adjust layout and save
plt.tight_layout()
plt.savefig(output_figure_path, dpi=300)  # High-resolution PNG
plt.savefig(
    output_figure_path.replace(".png", ".svg"), format="svg", dpi=300
)  # SVG for scalability
plt.show()
