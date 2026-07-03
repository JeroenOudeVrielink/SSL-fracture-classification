import numpy as np
from skimage.transform import hough_line, hough_line_peaks
from scipy.stats import mode
from skimage.filters import threshold_otsu, sobel
import cv2


class ImagePreprocessor:
    def __init__(
        self, resize=None, border_size=200, resize_intp_method=None, debug=False
    ):
        self.resize = resize
        self.border_size = border_size
        self.resize_intp_method = resize_intp_method
        self.debug = debug

    def get_otsu_threshold(self, img):
        threshold = threshold_otsu(img)
        return threshold

    def binarize_img_strict(self, img):
        bin_img = img < 1
        return bin_img

    def get_sobel_edges(self, img):
        img_edges = sobel(img)
        return img_edges

    def find_tilt_angle(self, img):
        # Compute hough lines
        h, theta, d = hough_line(img)
        # Compute peaks
        _, angles, dists = hough_line_peaks(h, theta, d)
        # find most common value
        angles_mode = mode(angles)
        # Convert to degrees
        angle = np.rad2deg(angles_mode[0])

        # Take the difference between the detected angle and the 4 right angles
        if angle < 0:
            diff = np.array([0, -90, -180, -270]) - angle
        else:
            diff = np.array([0, 90, 180, 270]) - angle
        # Find wich one is closest
        idx_smallest_diff = np.argmin(np.absolute(diff))
        # Rotate the the image in the direction of the clostes right angle
        r_angle = -diff[idx_smallest_diff]
        return r_angle, angles, dists

    def increase_black_borders(self, img):
        height, width = img.shape
        # Make image into a square with black borders
        if height > width:
            top = bottom = self.border_size
            left = right = int((height - width) / 2) + self.border_size
        else:
            top = bottom = int((width - height) / 2) + self.border_size
            left = right = self.border_size
        border_img = cv2.copyMakeBorder(
            img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=[0, 0, 0]
        )
        return border_img

    def shrink_black_borders(self, img):
        # Make binary image with otsu threshold
        threshold = self.get_otsu_threshold(img)
        _, thresh = cv2.threshold(img, threshold, 255, cv2.THRESH_BINARY)
        # Get boudingbox and crop
        x, y, w, h = cv2.boundingRect(thresh)
        cropped_img = img[y : y + h, x : x + w]
        return cropped_img, thresh

    def rotate_img(self, img, angle):
        # Find image center
        img_center = tuple(np.array(img.shape[1::-1]) / 2)
        # Rotate
        rot_mat = cv2.getRotationMatrix2D(img_center, angle, 1.0)
        result = cv2.warpAffine(img, rot_mat, img.shape[1::-1], flags=cv2.INTER_CUBIC)
        return result

    def preprocess(self, img):
        # Convert image to single channel
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # Binarize image
        bin_img = self.binarize_img_strict(gray_img)
        # Find edges
        img_edges = self.get_sobel_edges(bin_img)
        # Find tilt angle
        angle, angles, dists = self.find_tilt_angle(img_edges)
        # Increase black borders
        border_img = self.increase_black_borders(gray_img)
        # Rotate image
        rotated_img = self.rotate_img(border_img, angle)
        # Shrink black borders
        preprocessed_img, thresh = self.shrink_black_borders(rotated_img)
        # Resize if specified
        if self.resize is not None:
            preprocessed_img = cv2.resize(
                preprocessed_img,
                (self.resize, self.resize),
                interpolation=self.resize_intp_method,
            )
        if self.debug:
            return (
                preprocessed_img,
                border_img,
                bin_img,
                img_edges,
                rotated_img,
                angles,
                dists,
                thresh,
            )
        return preprocessed_img


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    preprocessor = ImagePreprocessor(debug=True)
