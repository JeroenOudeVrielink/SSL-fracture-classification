import numpy as np
import matplotlib.pyplot as plt
from skimage.transform import hough_line, hough_line_peaks
from scipy.stats import mode
from skimage.filters import threshold_otsu, sobel
import cv2


class ImagePreprocessor:
    def __init__(
        self,
        resize=None,
        border_size=200,
        verbose=False,
        verbose_save_dir="img_preprocessing",
    ):
        self.resize = resize
        self.border_size = border_size
        self.verbose = verbose
        self.verbose_save_dir = verbose_save_dir

    def get_otsu_threshold(self, img):
        threshold = threshold_otsu(img)
        if self.verbose:
            print("Otsu Threshold: ", threshold)
        return threshold

    def binarize_img(self, img):
        threshold = self.get_otsu_threshold(img)
        bin_img = img < 1
        # if self.verbose:
        #     cv2.imwrite(
        #         self.verbose_save_dir + "/binarized_image.png", bin_img.astype(int)
        #     )
        return bin_img

    def find_edges(self, img):
        img_edges = sobel(img)
        # if self.verbose:
        # cv2.imwrite(self.verbose_save_dir + "/edge_img.png", img_edges.astype(int))
        return img_edges

    def find_tilt_angle(self, img, img_save_path):
        # Compute high lines
        h, theta, d = hough_line(img)
        # Compute peaks
        _, angles, dists = hough_line_peaks(h, theta, d)
        # find most common value
        angles_mode = mode(angles)
        # print("Angles Mode: ", angles_mode)
        idx = np.where(angles == angles_mode[0])[0]
        angle = np.rad2deg(mode(angles)[0])

        test = mode(angles)

        # if angle < 0:
        #     r_angle = angle + 90
        # else:
        #     r_angle = angle - 90
        # Take the difference between the detected angle and the 4 right angles
        if angle < 0:
            diff = np.array([0, -90, -180, -270]) - angle
        else:
            diff = np.array([0, 90, 180, 270]) - angle
        # Find wich one is closest
        idx_smallest_diff = np.argmin(np.absolute(diff))
        # Rotate the the image in the direction of the clostes right angle
        r_angle = -diff[idx_smallest_diff]

        if self.verbose:
            print("Angle: ", angle)
            print("Rotated Angle: ", r_angle)

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
            ax.set_title("Detected lines")
            plt.savefig(self.verbose_save_dir + "/hough_lines.png")
            plt.cla()

            fig, ax = plt.subplots()
            ax.imshow(img, cmap="gray")
            origin = np.array((0, img.shape[1]))
            for i in idx:
                y0, y1 = (dists[i] - origin * np.cos(angles[i])) / np.sin(angles[i])
                ax.plot(origin, (y0, y1), "-r")
            ax.set_xlim(origin)
            ax.set_ylim((img.shape[0], 0))
            ax.set_axis_off()
            ax.set_title("Detected lines")
            plt.savefig(self.verbose_save_dir + "/selected_hough_line.png")
            plt.cla()

        fig, ax = plt.subplots()
        ax.imshow(img, cmap="gray")
        origin = np.array((0, img.shape[1]))
        for i in idx:
            y0, y1 = (dists[i] - origin * np.cos(angles[i])) / np.sin(angles[i])
            ax.plot(origin, (y0, y1), "-r")
        ax.set_xlim(origin)
        ax.set_ylim((img.shape[0], 0))
        ax.set_axis_off()
        ax.set_title(f"angle:{angle} r_angle:{r_angle}")
        plt.savefig(img_save_path[:-4] + "selected_hugh_lines.png")
        plt.close()

        return r_angle

    def increase_black_borders(self, img):
        height, width = img.shape
        if height > width:
            top = bottom = self.border_size
            left = right = int((height - width) / 2) + self.border_size
        else:
            top = bottom = int((width - height) / 2) + self.border_size
            left = right = self.border_size
        border_img = cv2.copyMakeBorder(
            img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=[0, 0, 0]
        )
        if self.verbose:
            cv2.imwrite(self.verbose_save_dir + "/border_image.png", border_img)
        return border_img

    def shrink_black_borders(self, img):
        # make in binary image for threshold value of 1
        threshold = self.get_otsu_threshold(img)
        _, thresh = cv2.threshold(img, threshold, 255, cv2.THRESH_BINARY)

        if self.verbose:
            cv2.imwrite(self.verbose_save_dir + "/border_thresh.png", thresh)
        x, y, w, h = cv2.boundingRect(thresh)
        cropped_img = img[y : y + h, x : x + w]
        if self.verbose:
            cv2.imwrite(self.verbose_save_dir + "/border_shrink.png", cropped_img)
        return cropped_img

    def rotate_img(self, img, angle):
        img_center = tuple(np.array(img.shape[1::-1]) / 2)
        rot_mat = cv2.getRotationMatrix2D(img_center, angle, 1.0)
        result = cv2.warpAffine(img, rot_mat, img.shape[1::-1], flags=cv2.INTER_CUBIC)
        if self.verbose:
            cv2.imwrite(self.verbose_save_dir + "/rotated_image.png", result)
        return result

    def preprocess(self, img, img_save_path):
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        border_img = self.increase_black_borders(gray_img)
        bin_img = self.binarize_img(border_img)
        # bin_img = self.binarize_img(gray_img)
        img_edges = self.find_edges(bin_img)
        angle = self.find_tilt_angle(img_edges, img_save_path)
        # border_img = self.increase_black_borders(gray_img)
        rotated_img = self.rotate_img(border_img, angle)
        preprocessed_img = self.shrink_black_borders(rotated_img)
        if self.resize is not None:
            preprocessed_img = cv2.resize(
                preprocessed_img,
                (self.resize, self.resize),
                interpolation=cv2.INTER_CUBIC,
            )
        return preprocessed_img


if __name__ == "__main__":
    # img = cv2.imread(
    #     "img_preprocessing/0698c405-6903-4c92-91b0-23e72367bbb5_x_tib-fib_ap_Win.png"
    # )
    preprocessor = ImagePreprocessor(verbose=False)
    for i in range(1, 8):
        img_path = f"img_preprocessing/debug_imgs/{i}/{i}.png"
        img = cv2.imread(img_path)
        # cv2.imshow("img", img)
        # cv2.waitKey(0)
        p_img = preprocessor.preprocess(img, img_path)
        cv2.imwrite(img_path[:-4] + "_preprocessed.png", p_img)
