import argparse
import sys
import pandas as pd
import cv2
from torchvision import transforms
from PIL import Image
import json
import os

sys.path.insert(1, "/home/jvrielink/multi-head-bone-classifier/src")
from model.get_model import get_model

UNKNOWN_FILES = "/home/jvrielink/data_hdd/AIML_rot_corrected/annotations/merged_classes_age/unknown.pkl"
BODY_NAMES = "/home/jvrielink/data_hdd/AIML_rot_corrected/annotations/merged_classes_age/body_part_names.json"
VIEW_NAMES = "/home/jvrielink/data_hdd/AIML_rot_corrected/annotations/merged_classes_age/view_names.json"

transform = transforms.Compose(
    [
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
    ]
)


def test_classifier(args):
    model = get_model(args)
    model.eval()
    model.to("cuda")

    unknown_files = pd.read_pickle(UNKNOWN_FILES)
    unknown_files = unknown_files.to_frame(name="path").reset_index()
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

    image_paths = unknown_files["path"].tolist()

    for path in image_paths:
        img_path = "/home/jvrielink/data_hdd/AIML_rot_corrected/" + path

        pil_image = Image.open(img_path)
        x = transform(pil_image)
        x = x.unsqueeze(0).to("cuda")
        body_pred, view_pred = model(x)

        body_pred = body_pred.argmax(dim=-1).item()
        view_pred = view_pred.argmax(dim=-1).item()

        body_pred_str = new_body_part_names[body_pred]
        view_pred_str = new_view_names[view_pred]

        print(f"Body part: {body_pred_str}, View: {view_pred_str}")

        image = cv2.imread(img_path)

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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Mutlitask bone classification.")

    parser.add_argument(
        "--checkpoint_load_path",
        type=str,
        default="~/models/Gen3_v1_imgnet_pretrained_color_jitter_rotation_lr_schedule_04-05_22:42:44/epoch99_summedf1_1.64.ckpt",
        help="The path to the directory where checkpoint will be loaded from.",
    )

    parser.add_argument(
        "--data_path",
        type=str,
        default="/AIML_rot_corrected",
        help="The path to the AIML dataset.",
    )

    parser.add_argument(
        "--annotations_path",
        type=str,
        default="annotations/merged_classes_age",
        help="The path to the annotations directory.",
    )

    parser.add_argument(
        "--body_part_names_path",
        type=str,
        default="/home/jvrielink/data_hdd/AIML_rot_corrected/annotations/merged_classes_age/body_part_names.json",
        help="Path to .json with body part names.",
    )

    parser.add_argument(
        "--view_names_path",
        type=str,
        default="/home/jvrielink/data_hdd/AIML_rot_corrected/annotations/merged_classes_age/view_names.json",
        help="Path to .json with view names.",
    )

    parser.add_argument(
        "--run_name",
        type=str,
        default="Test",
        help="The name of the run.",
    )

    parser.add_argument(
        "--model_version", type=str, default="v1", help="Which model version to use."
    )

    parser.add_argument(
        "--checkpoint_save_path",
        type=str,
        default="/classification_models",
        help="The path to the directory where checkpoints will be saved.",
    )

    parser.add_argument(
        "--results_save_path",
        type=str,
        default="/code/results",
        help="The path to the directory where results will be saved.",
    )

    parser.add_argument(
        "--num_classes1",
        type=int,
        default=13,
        help="The number of classes for decoder 1.",
    )

    parser.add_argument(
        "--num_classes2",
        type=int,
        default=3,
        help="The number of classes for decoder 2.",
    )

    parser.add_argument(
        "--input_size",
        type=int,
        default=(3, 244, 244),
        nargs="+",
        help="Pixel size of the input image.",
    )

    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-3,
        help="The learning rate to use for training.",
    )

    parser.add_argument(
        "--train_batch_size",
        type=int,
        default=64,
        help="The batch size to use for training.",
    )

    parser.add_argument(
        "--eval_batch_size",
        type=int,
        default=64,
        help="The batch size to use for evaluation.",
    )

    parser.add_argument(
        "--max_epochs",
        type=int,
        default=20,
        help="The maximum number of epochs to train for.",
    )

    parser.add_argument(
        "--log_every_n_steps",
        type=int,
        default=10,
        help="The number of trainig steps to log after.",
    )

    parser.add_argument(
        "--val_check_interval",
        type=int,
        default=500,
        help="The number of training steps to do a validation run after.",
    )

    parser.add_argument(
        "--limit_train_batches",
        type=float,
        default=1.0,
        help="The proportion of the training data to use.",
    )

    parser.add_argument(
        "--limit_val_batches",
        type=float,
        default=1.0,
        help="The proportion of the validation data to use for validation.",
    )

    parser.add_argument(
        "--limit_test_batches",
        type=int,
        default=None,
        help="The number of batches to use for testing.",
    )

    parser.add_argument(
        "--limit_predict_batches",
        type=int,
        default=None,
        help="The number of batches to use for prediction.",
    )

    parser.add_argument(
        "--save_top_k",
        type=int,
        default=6,
        help="Number of models to save during checkpointing",
    )

    parser.add_argument(
        "--monitor",
        type=str,
        default="val/summed_f1",
        help="The metric to monitor for model checkpointing.",
    )

    parser.add_argument(
        "--num_workers",
        type=int,
        default=10,
        help="The number of workers to use for data loading.",
    )

    parser.add_argument(
        "--lr_start_factor",
        type=float,
        default=1.0,
        help="The factor to start the learning rate at.",
    )

    parser.add_argument(
        "--lr_end_factor",
        type=float,
        default=None,
        help="The factor to end the learning rate at.",
    )

    parser.add_argument(
        "--predict",
        type=bool,
        default=True,
        help="Whether to use predict function in evaluate.py.",
    )

    parser.add_argument(
        "--model_output_savepath",
        type=str,
        default="/home/jvrielink/Thesis/multi-head-bone-classifier/results",
        help="Save directory for model output when predicting",
    )

    parser.add_argument(
        "--body_loss_weights_path",
        type=str,
        default=None,
        help="Weights for the body part classification loss.",
    )

    parser.add_argument(
        "--view_loss_weights_path",
        type=str,
        default=None,
        help="Weights for the view classification loss.",
    )

    parser.add_argument(
        "--sample_weights_path",
        type=None,
        default=None,
        help="Weights path for weighted random sampling",
    )

    parser.add_argument(
        "--clip_p",
        type=float,
        default=1.0,
        help="The probability boundary value for clipped cross entropy loss.",
    )

    parser.add_argument(
        "--imgnet_pretrained",
        type=bool,
        default=False,
        help="Whether to use an ImageNet pretrained model.",
    )

    parser.add_argument(
        "--aug_rotation",
        type=bool,
        default=False,
        help="Whether to rotation on the training images.",
    )

    parser.add_argument(
        "--aug_color_jitter",
        type=bool,
        default=False,
        help="Whether to use color jitter on the training images.",
    )

    parser.add_argument(
        "--metric_in_filename",
        type=bool,
        default=False,
        help="Whether to include the metric score in the filename.",
    )

    parser.add_argument(
        "--optimizer_type",
        type=str,
        default="adam",
        help="The type of optimizer to use.",
    )

    args = parser.parse_args()
    test_classifier(args)
