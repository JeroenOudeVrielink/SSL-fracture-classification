import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

SAVE_PATH = "visualization/collapse_plots/dino"
LOG_SCALE = True
EXTRA = False
Y_LIM = None

title = "Singular Value Decomposition of Projected Feature Vectors"

paths = [
    # "visualization/projected_feature_vectors/moco_v2/moco_v2_luxry",
    # "visualization/projected_feature_vectors/moco_v2/moco_v2_downstream_v2",
    # "visualization/projected_feature_vectors/moco_v2/moco_v2_imagenet_train",
    # "visualization/projected_feature_vectors/moco_v2/moco_v2_coco",
    # "visualization/projected_feature_vectors/moco_v3/moco_v3_luxry",
    # "visualization/projected_feature_vectors/moco_v3/moco_v3_downstream_v2",
    # "visualization/projected_feature_vectors/moco_v3/moco_v3_imagenet_train",
    # "visualization/projected_feature_vectors/moco_v3/moco_v3_coco",
    # "visualization/projected_feature_vectors/vicreg/vicreg_luxry",
    # "visualization/projected_feature_vectors/vicreg/vicreg_downstream_v2",
    # "visualization/projected_feature_vectors/vicreg/vicreg_imagenet_train",
    # "visualization/projected_feature_vectors/vicreg/vicreg_coco",
    # "visualization/projected_feature_vectors/vicregl/vicregl_luxry",
    # "visualization/projected_feature_vectors/vicregl/vicregl_downstream_v2",
    # "visualization/projected_feature_vectors/vicregl/vicregl_imagenet_train",
    # "visualization/projected_feature_vectors/vicregl/vicregl_coco",
    "visualization/projected_feature_vectors/dino/dinov1_luxry",
    "visualization/projected_feature_vectors/dino/dinov1_downstream_v2",
    "visualization/projected_feature_vectors/dino/dinov1_imagenet_train",
    "visualization/projected_feature_vectors/dino/dinov1_coco",
    # "visualization/projected_feature_vectors/vicregl/vicregl_cov_coeff_10_AIML",
    # "visualization/projected_feature_vectors/vicregl/vicregl_cov_coeff_10",
    # "visualization/projected_feature_vectors/vicregl/vicregl_cov_coeff_25_AIML",
    # "visualization/projected_feature_vectors/vicregl/vicregl_cov_coeff_25",
    # "visualization/projected_feature_vectors/vicregl/vicregl_cov_coeff_50_AIML",
    # "visualization/projected_feature_vectors/vicregl/vicregl_cov_coeff_50",
    # "visualization/projected_feature_vectors/dino/dinov1_AIML",
    # "visualization/projected_feature_vectors/dino/dinov1",
]

if EXTRA:
    labels = [
        "LuXry (train) c=10",
        "Downstream (external) c=10",
        "LuXry (train) c=25",
        "Downstream (external) c=25",
        "LuXry (train) c=50",
        "Downstream (external) c=50",
    ]
else:
    labels = [
        "LuXry (train)",
        "Downstream (external)",
        "ImageNet (train)",
        "COCO (external)",
    ]

y1 = np.load(paths[0] + "/singular_values.npy")
y2 = np.load(paths[1] + "/singular_values.npy")
y3 = np.load(paths[2] + "/singular_values.npy")
y4 = np.load(paths[3] + "/singular_values.npy")
if EXTRA:
    y5 = np.load(paths[4] + "/singular_values.npy")
    y6 = np.load(paths[5] + "/singular_values.npy")

x = np.arange(0, len(y1))

if LOG_SCALE:
    y1 = np.log(y1)
    y2 = np.log(y2)
    y3 = np.log(y3)
    y4 = np.log(y4)
    if EXTRA:
        y5 = np.log(y5)
        y6 = np.log(y6)


# Apply the styles
plt.rcParams["text.usetex"] = True
plt.rcParams["font.family"] = "serif"
sns.set_theme(style="whitegrid", font_scale=1.7)  # Adjust font scale as needed

axis_font_size = 20  # Adjust axis font size
title_font_size = 32  # Adjust title font size
legend_font_size = 14  # Adjust legend font size

plt.figure(figsize=(8, 6))  # Adjust figure size as needed
# Create the figure and axes
if EXTRA:
    plt.plot(x, y1, label=labels[0], color="darkgreen", linewidth=3)
    plt.plot(x, y2, label=labels[1], color="mediumseagreen", linewidth=3)
    plt.plot(x, y3, label=labels[2], color="darkorange", linewidth=3)
    plt.plot(x, y4, label=labels[3], color="orange", linewidth=3)
    plt.plot(x, y5, label=labels[4], color="royalblue", linewidth=3)
    plt.plot(x, y6, label=labels[5], color="skyblue", linewidth=3)
else:
    plt.plot(x, y1, label=labels[0], color="royalblue", linewidth=3)
    plt.plot(x, y2, label=labels[1], color="skyblue", linewidth=3)
    plt.plot(x, y3, label=labels[2], color="firebrick", linewidth=3)
    plt.plot(x, y4, label=labels[3], color="lightcoral", linewidth=3)
# plt.plot(x, y4, label=labels[4], color="seagreen", linewidth=3)
# if EXTRA:
#     plt.plot(x, y5, label=labels[4], color="seagreen", linewidth=3)
#     plt.plot(x, y6, label=labels[5], color="lightgreen", linewidth=3)

if Y_LIM is not None:
    ymin, ymax = plt.ylim()

    # Increase the upper limit by a desired amount (e.g., 10%)
    new_ymax = Y_LIM

    # Set the new y-axis limits
    plt.ylim(ymin, new_ymax)

# Customize the plot
plt.xlabel("Singular Value Rank Index", fontsize=axis_font_size)
y_label = "Logarithm of Singular Value" if LOG_SCALE else "Singular Value"
plt.ylabel(y_label, fontsize=axis_font_size)
# plt.title("Example Plot for Scientific Paper", fontsize=title_font_size)
plt.legend(fontsize=legend_font_size, loc="upper right")  # Add a legend
# plt.grid(axis="y", linestyle="--")  # Add a subtle grid

# Adjust layout to prevent overlapping elements
plt.tight_layout()

# Save the figure (optional)
plt.savefig(SAVE_PATH + ".png", dpi=300)  # High DPI for print quality
plt.savefig(SAVE_PATH + ".svg", format="svg", dpi=300)  # SVG for scalability

# Show the plot
plt.show()
