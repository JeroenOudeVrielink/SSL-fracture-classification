import os
import pandas as pd
from tqdm import tqdm
import cv2
from image_preprocessor import ImagePreprocessor
from pathlib import Path
from torch.utils.data import Dataset, DataLoader

MASTER_CV_PATH = "src/preprocessing/label_files/all_files.csv"
SOURCE_DIR = "/home/jvrielink/data_hdd/AIML"
NEW_DATA_DIR = "/home/jvrielink/data_hdd/test"
RESIZE = None
RESIZE_INTP_METHOD = cv2.INTER_AREA


class ProcessingDataset(Dataset):
    def __init__(
        self,
        master_cv_path: str,
        source_dir: str,
        new_data_dir: str,
        resize: int,
        resize_intp_method,
    ):
        self.master_df = pd.read_csv(master_cv_path)
        self.source_dir = source_dir
        self.new_data_dir = new_data_dir
        # Setup the image preprocessor
        self.image_preprocessor = ImagePreprocessor(
            resize=resize, resize_intp_method=resize_intp_method
        )
        self.faulty_images = []

    def __len__(self):
        return len(self.master_df)

    def __getitem__(self, idx):
        # Read in image
        img_path = self.master_df.iloc[idx, -1]
        sub_dir, upper_lower, file = img_path.split("/")[-3:]
        img = cv2.imread(os.path.join(self.source_dir, img_path))
        # Try to preprocess the image
        try:
            preprocessed_img = self.image_preprocessor.preprocess(img)
            cv2.imwrite(os.path.join(self.new_data_dir, img_path), preprocessed_img)
        except:
            print(f"Error with image: {img_path}")
            self.faulty_images.append((idx, img_path))
            cv2.imwrite(
                os.path.join(self.new_data_dir, "faulty", upper_lower, file), img
            )
        return 0


def get_sub_dirs(dir):
    # Get a list of all entries in the directory
    entries = os.listdir(dir)

    # Filter out only the directories
    dirs = [entry for entry in entries if os.path.isdir(os.path.join(dir, entry))]
    return dirs


def setup_dirs(source_dir, new_data_dir):
    # Create the new data dir
    os.mkdir(new_data_dir)
    sub_dirs = get_sub_dirs(source_dir)
    sub_dirs.append("faulty")
    # Create the sub dirs
    for sub_dir in sub_dirs:
        os.mkdir(os.path.join(new_data_dir, sub_dir))
        os.mkdir(os.path.join(new_data_dir, sub_dir, "Lower"))
        os.mkdir(os.path.join(new_data_dir, sub_dir, "Upper"))


def write_to_txt(string_list, new_data_dir):
    file_path = os.path.join(new_data_dir, "faulty", "paths_of_faulty_imgs.txt")
    # Open the file in write mode
    with open(file_path, "w") as file:
        # Write each string to the file
        for idx, string in string_list:
            file.write(f"{idx},{string}\n")


def preprocess(source_dir, new_data_dir, master_cv_path, resize, resize_intp_method):
    setup_dirs(source_dir, new_data_dir)
    # Read the master csv
    data = ProcessingDataset(
        master_cv_path, source_dir, new_data_dir, resize, resize_intp_method
    )
    loader = DataLoader(dataset=data, batch_size=4, shuffle=False, num_workers=4)
    for _ in tqdm(loader):
        pass
    write_to_txt(data.faulty_images, new_data_dir)


if __name__ == "__main__":
    preprocess(SOURCE_DIR, NEW_DATA_DIR, MASTER_CV_PATH, RESIZE, RESIZE_INTP_METHOD)
