import os
from pathlib import Path
import numpy as np
import pandas as pd
from utils.plot_confusion_matrix import plot_confusion_matrix
import argparse

from sklearn.metrics import (
    f1_score,
    accuracy_score,
    precision_score,
    recall_score,
)


def compute_and_save_metrics(
    true_labels_path,
    body_preds,
    view_preds,
    checkpoint_path,
    save_dir,
    body_part_names,
    view_names,
):
    true_labels = pd.read_pickle(true_labels_path)
    # true_labels = true_labels.head(64)

    body_parts_one_hot = np.array(true_labels["body_part_encoded"].tolist())
    views_one_hot = np.array(true_labels["view_encoded"].tolist())
    body_true = np.argmax(body_parts_one_hot, axis=-1)
    view_true = np.argmax(views_one_hot, axis=-1)

    body_preds = np.array(body_preds).argmax(axis=-1)
    view_preds = np.array(view_preds).argmax(axis=-1)

    body_scores = pd.DataFrame(dtype=np.float64)
    view_scores = pd.DataFrame(dtype=np.float64)

    body_scores["model"] = ""
    view_scores["model"] = ""
    body_scores.at[0, "model"] = checkpoint_path
    view_scores.at[0, "model"] = checkpoint_path

    body_scores["macro_f1"] = np.nan
    view_scores["macro_f1"] = np.nan
    body_scores.at[0, "macro_f1"] = f1_score(body_true, body_preds, average="macro")
    view_scores.at[0, "macro_f1"] = f1_score(view_true, view_preds, average="macro")

    body_scores["micro_f1"] = np.nan
    view_scores["micro_f1"] = np.nan
    body_scores.at[0, "micro_f1"] = f1_score(body_true, body_preds, average="micro")
    view_scores.at[0, "micro_f1"] = f1_score(view_true, view_preds, average="micro")

    body_scores["accuracy"] = np.nan
    view_scores["accuracy"] = np.nan
    body_scores.at[0, "accuracy"] = accuracy_score(body_true, body_preds)
    view_scores.at[0, "accuracy"] = accuracy_score(view_true, view_preds)

    body_scores["macro_precision"] = np.nan
    view_scores["macro_precision"] = np.nan
    body_scores.at[0, "macro_precision"] = precision_score(
        body_true, body_preds, average="macro"
    )
    view_scores.at[0, "macro_precision"] = precision_score(
        view_true, view_preds, average="macro"
    )

    body_scores["macro_recall"] = np.nan
    view_scores["macro_recall"] = np.nan
    body_scores.at[0, "macro_recall"] = recall_score(
        body_true, body_preds, average="macro"
    )
    view_scores.at[0, "macro_recall"] = recall_score(
        view_true, view_preds, average="macro"
    )

    plot_confusion_matrix(
        body_true,
        body_preds,
        title="body_conf_matrix",
        save_dir=save_dir,
        x_labels=body_part_names,
        y_labels=body_part_names,
        view=False,
    )
    plot_confusion_matrix(
        view_true,
        view_preds,
        title="view_conf_matrix",
        save_dir=save_dir,
        x_labels=view_names,
        y_labels=view_names,
        view=True,
    )
    body_scores.to_csv(save_dir / "body_scores.csv", index=False)
    view_scores.to_csv(save_dir / "view_scores.csv", index=False)


def compute_evaluation(true_labels_path, results_dir):
    results_dir = results_dir
    sub_dirs = [
        dir
        for dir in os.listdir(results_dir)
        if os.path.isdir(os.path.join(results_dir, dir))
    ]

    true_labels_df = pd.read_pickle(true_labels_path)
    body_parts_one_hot = np.array(true_labels_df["body_part_encoded"].tolist())
    views_one_hot = np.array(true_labels_df["view_encoded"].tolist())
    body_true = np.argmax(body_parts_one_hot, axis=-1)
    view_true = np.argmax(views_one_hot, axis=-1)

    for sub_dir in sub_dirs:
        path = Path(results_dir) / sub_dir
        body_preds = np.load(path / "body_preds.npy").argmax(axis=-1)
        view_preds = np.load(path / "view_preds.npy").argmax(axis=-1)

        body_scores = pd.DataFrame(dtype=np.float64)
        view_scores = pd.DataFrame()

        body_scores["model"] = ""
        view_scores["model"] = ""
        body_scores.at[0, "model"] = sub_dir
        view_scores.at[0, "model"] = sub_dir

        body_scores["macro_f1"] = np.nan
        view_scores["macro_f1"] = np.nan
        body_scores.at[0, "macro_f1"] = f1_score(body_true, body_preds, average="macro")
        view_scores.at[0, "macro_f1"] = f1_score(view_true, view_preds, average="macro")

        body_scores["micro_f1"] = np.nan
        view_scores["micro_f1"] = np.nan
        body_scores.at[0, "micro_f1"] = f1_score(body_true, body_preds, average="micro")
        view_scores.at[0, "micro_f1"] = f1_score(view_true, view_preds, average="micro")

        body_scores["accuracy"] = np.nan
        view_scores["accuracy"] = np.nan
        body_scores.at[0, "accuracy"] = accuracy_score(body_true, body_preds)
        view_scores.at[0, "accuracy"] = accuracy_score(view_true, view_preds)

        body_scores["macro_precision"] = np.nan
        view_scores["macro_precision"] = np.nan
        body_scores.at[0, "macro_precision"] = precision_score(
            body_true, body_preds, average="macro"
        )
        view_scores.at[0, "macro_precision"] = precision_score(
            view_true, view_preds, average="macro"
        )

        body_scores["macro_recall"] = np.nan
        view_scores["macro_recall"] = np.nan
        body_scores.at[0, "macro_recall"] = recall_score(
            body_true, body_preds, average="macro"
        )
        view_scores.at[0, "macro_recall"] = recall_score(
            view_true, view_preds, average="macro"
        )

        plot_confusion_matrix(
            body_true, body_preds, title=sub_dir + " Body", save_dir=path, view=False
        )
        plot_confusion_matrix(
            view_true, view_preds, title=sub_dir + " View", save_dir=path, view=True
        )
        body_scores.to_csv(path / "body_scores.csv", index=False)
        view_scores.to_csv(path / "view_scores.csv", index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Move files from sub-subdirectories to subdirectories with renaming."
    )
    parser.add_argument(
        "--true_labels_path",
        help="File path to the ground truth labels.",
        type=str,
        default="/AIML_half_size/annotations/merged_classes/test.pkl",
    )
    parser.add_argument(
        "--results_dir",
        help="Path to the directory containing the results.",
        type=str,
        default="/code/results",
    )

    args = parser.parse_args()
    compute_evaluation(args.true_labels_path, args.results_dir)
