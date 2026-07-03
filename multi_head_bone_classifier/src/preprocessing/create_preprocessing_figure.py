import numpy as np
from scipy.stats import mode
import cv2
from image_preprocessor import ImagePreprocessor
import matplotlib.pyplot as plt
import matplotlib

# import seaborn as sns
from matplotlib.gridspec import GridSpec


if __name__ == "__main__":
    matplotlib.rcParams["text.usetex"] = True  # Enable LaTeX rendering
    matplotlib.rcParams["font.family"] = "serif"
    # sns.set_theme(font_scale=2.0)

    preprocessor = ImagePreprocessor(debug=True)

    img_path = f"src/preprocessing/test_imgs/2.png"
    img = cv2.imread(img_path)
    p_img, border_img, bin_img, img_edges, rotated_img, angles, dists, thresh = (
        preprocessor.preprocess(img)
    )
    bin_img = bin_img.astype(np.uint8) * 255
    # thresh = thresh.astype(np.uint8) * 255
    cv2.imwrite(img_path[:-4] + "_preprocessed.png", p_img)
    cv2.imwrite(img_path[:-4] + "_border.png", border_img)
    cv2.imwrite(img_path[:-4] + "_binarized.png", bin_img)
    cv2.imwrite(img_path[:-4] + "_otsu.png", thresh)

    angles_mode = mode(angles)
    # print("Angles Mode: ", angles_mode)
    idx = np.where(angles == angles_mode[0])[0]

    fig, ax = plt.subplots()
    ax.imshow(bin_img, cmap="gray")
    origin = np.array((0, img.shape[1]))
    # for i in idx:
    y0, y1 = (dists[0] - origin * np.cos(angles[0])) / np.sin(angles[0])
    ax.plot(origin, (y0, y1), "-r", linewidth=5, label="Selected Hough line")
    # ax.plot(origin, (y0, y1), "-r")
    ax.set_xlim(origin)
    ax.set_ylim((img.shape[0], 0))
    ax.set_axis_off()
    # ax.set_title("Detected lines")
    # plt.legend()
    plt.tight_layout()
    plt.savefig(
        img_path[:-4] + "_selected_hough_line.png", bbox_inches="tight", pad_inches=0
    )
    plt.cla()

    # Plot Image and Lines
    fig, ax = plt.subplots()
    ax.imshow(img, cmap="gray")
    origin = np.array((0, img.shape[1]))
    for angle, dist in zip(angles, dists):
        y0, y1 = (dist - origin * np.cos(angle)) / np.sin(angle)
        ax.plot(origin, (y0, y1), "-r")
    ax.set_xlim(origin)
    ax.set_ylim((img.shape[0], 0))
    ax.set_axis_off()
    # ax.set_title("Detected lines")
    plt.tight_layout()
    plt.savefig(img_path[:-4] + "_hough_lines.png", bbox_inches="tight", pad_inches=0)
    plt.cla()

# # Create a figure and a grid layout
# fig = plt.figure(figsize=(12, 5))  # Adjusted size for titles
# grid = GridSpec(1, 4, wspace=0.2)

# # Load and display the images
# images = [img, bin_img, thresh, p_img]
# titles = ["Cat", "Dog", "Bird", "Fish"]

# for i in range(4):
#     ax = plt.subplot(grid[0, i])
#     ax.imshow(images[i], cmap="gray")
#     ax.axis("off")
#     ax.set_title(titles[i])  # Add title to each image

#     # Custom origin and starting points for the line
#     if i == 1:
#         origin_x, origin_y = np.array((0, img.shape[1]))
#         # for i in idx:
#         end_x, end_y = (dists[0] - origin * np.cos(angles[0])) / np.sin(angles[0])
#         ax.plot([origin_x, end_x], [origin_y, end_y], "r-", linewidth=2)

# # Show the plot
# plt.tight_layout()
# plt.show()
