import os
import shutil
import argparse
from pathlib import Path


# Function to move files from sub-subdirectories to subdirectories
def move_files_to_subdirectories(root_dir):
    for root, dirs, _ in os.walk(root_dir):
        for sub_dir in dirs:
            sub_dir_path = os.path.join(root, sub_dir)

            sub_sub_dirs = [
                d
                for d in os.listdir(sub_dir_path)
                if os.path.isdir(os.path.join(sub_dir_path, d))
            ]
            for sub_sub_dir in sub_sub_dirs:
                path_sub_sub_dir = root + "/" + sub_dir + "/" + sub_sub_dir
                for file in os.listdir(path_sub_sub_dir):
                    # Rename the file with sub-sub-subdirectory name
                    old_file_path = path_sub_sub_dir + "/" + file
                    new_file_path = path_sub_sub_dir + "_" + file

                    # Move the file to the subdirectory
                    shutil.move(old_file_path, new_file_path)
                    print(f"Moved {old_file_path} to {new_file_path}")
                os.rmdir(path_sub_sub_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Move files from sub-subdirectories to subdirectories with renaming."
    )
    parser.add_argument(
        "--root_dir",
        help="The root directory to start the operation.",
        type=str,
        default="test_move",
    )

    args = parser.parse_args()
    move_files_to_subdirectories(args.root_dir)
