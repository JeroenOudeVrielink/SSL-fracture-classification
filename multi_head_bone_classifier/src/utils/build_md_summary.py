import pandas as pd
import numpy as np
import os
from pathlib import Path
import glob
import json

RESULTS_DIR = "results"
SUB_DIRS = [
    "Gen3_v1_vanilla_epoch63_04-09_00:35:56",
    "Gen3_v1_imgnet_pretrained_epoch83_04-09_00:38:12",
    "Gen3_v1_imgnet_pretrained_color_jitter_epoch83_04-09_00:39:17",
    "Gen3_v1_imgnet_pretrained_color_jitter_rotation_epoch73_04-09_00:39:57",
    "Gen3_v1_imgnet_pretrained_color_jitter_rotation_lr_schedule_epoch99_04-09_00:40:37",
    "Gen3_v1_imgnet_pretrained_color_jitter_rotation_lr_schedule_oversampling_epoch99_04-09_00:41:18",
    "Gen3_v1_imgnet_pretrained_color_jitter_rotation_lr_schedule_oversampling_loss_weighting_epoch_98_04-09_00:41:58",
    "Gen3_v1_imgnet_pretrained_color_jitter_rotation_lr_schedule_loss_weighting_epoch91_04-10_00:35:56",
    "Gen3_v1_imgnet_pretrained_color_jitter_rotation_lr_schedule_loss_weighting_epoch67_04-10_00:53:50",
    "Gen3_v1_imgnet_pretrained_color_jitter_rotation_lr_schedule_loss_weighting_epoch85_04-10_00:51:06",
    "Gen3_v1_imgnet_pretrained_color_jitter_rotation_lr_schedule_loss_weighting_epoch99_04-10_00:50:26",
    "Gen3_v1_imgnet_pretrained_color_jitter_rotation_lr_schedule_loss_weighting099_epoch95_04-11_00:46:02",
    "Gen3_v1_sgd_epoch_97_04-11_00:48:01",
]
MODEL_DESCRIPTIONS_DIR = "src/model/model_descriptions"

CONFIG_KEYS = [
    "model_version",
    "input_size",
    "learning_rate",
    "train_batch_size",
    "max_epochs",
    "imgnet_pretrained",
    "aug_color_jitter",
    "aug_rotation",
    "lr_end_factor",
    "body_loss_weight_path",
    "view_loss_weight_path",
    "sample_weight_path",
    "monitor",
]


def get_subdirs(results_dir):
    sub_dirs = [
        dir
        for dir in os.listdir(results_dir)
        if os.path.isdir(os.path.join(results_dir, dir))
    ]
    return sub_dirs


def get_general_description_md(results_dir):
    md = "### General description \n\n"
    txt_file = Path(results_dir) / "general_description.txt"
    with open(txt_file, "r", encoding="utf-8") as file:
        md += file.read()
    md += "\n\n"
    return md


def get_metrics_md(results_dir, sub_dirs):
    # Read all evalation metrics scores and append to df
    body_df, view_df = pd.DataFrame(), pd.DataFrame()
    for sub_dir in sub_dirs:
        body_file = Path(results_dir) / sub_dir / "body_scores.csv"
        body_df = pd.concat([body_df, pd.read_csv(body_file)], ignore_index=True)
        view_file = Path(results_dir) / sub_dir / "view_scores.csv"
        view_df = pd.concat([view_df, pd.read_csv(view_file)], ignore_index=True)

    body_df = body_df.sort_values(by=["macro_f1"], ascending=False)
    view_df = view_df.sort_values(by=["macro_f1"], ascending=False)

    md = "### Body part classification test results \n\n"
    md += body_df.to_markdown(index=False, floatfmt=".4f") + "\n\n"
    md += "### View classification test results \n\n"
    md += view_df.to_markdown(index=False, floatfmt=".4f") + "\n\n"
    return md


def get_model_description_md(model_description_dir):
    models_df = pd.DataFrame(dtype=str, columns=["model", "description"])
    txt_files = glob.glob(model_description_dir + "/*.txt")
    txt_files.sort()

    # Loop through the list of .txt files and read their contents
    for idx, txt_file in enumerate(txt_files):
        with open(txt_file, "r", encoding="utf-8") as file:
            model_name = txt_file.split("/")[-1][:-4]
            content = file.read()
            models_df.at[idx, "model"] = model_name
            models_df.at[idx, "description"] = content

    md = "### Model descriptions \n\n"
    md += models_df.to_markdown(index=False) + "\n\n"
    return md


def get_conf_matrix_md(sub_dirs):
    md = "### Confusion matrices \n\n"
    for sub_dir in sub_dirs:
        md += f"#### {sub_dir} \n\n"
        body_img_file = sub_dir + "/" + "conf_matrix_body.png"
        md += f"![Image]({body_img_file})\n\n"
        view_img_file = sub_dir + "/" + "conf_matrix_view.png"
        md += f"![Image]({view_img_file})\n\n"
    md += "\n\n"
    return md


def get_config_md(results_dir, sub_dirs, keys_to_extract):
    # Create an empty DataFrame to store the extracted values
    params = []

    # Loop through JSON files in the directory
    for sub_dir in sub_dirs:
        json_path = Path(results_dir) / sub_dir / "config.json"
        with open(json_path, "r") as file:
            config_data = json.load(file)
            extracted_values = {}
            extracted_values["model"] = sub_dir
            for key in keys_to_extract:
                if key in config_data:
                    extracted_values[key] = config_data[key]
                else:
                    # Fill in with None if the key does not exist
                    extracted_values[key] = None
            params.append(extracted_values)

    params_df = pd.DataFrame(params)
    md = "### Model parameters \n\n"
    md += params_df.to_markdown(index=False) + "\n\n"
    return md


def build_md_summary(results_dir, sub_dirs, model_description_dir, config_keys):

    md = "# Test results summary \n\n"
    md += get_general_description_md(results_dir)
    md += get_model_description_md(model_description_dir)
    md += get_config_md(results_dir, sub_dirs, config_keys)
    md += get_metrics_md(results_dir, sub_dirs)
    md += get_conf_matrix_md(sub_dirs)

    with open(Path(results_dir) / "summary.md", "w") as f:
        f.write(md)


if __name__ == "__main__":
    build_md_summary(RESULTS_DIR, SUB_DIRS, MODEL_DESCRIPTIONS_DIR, CONFIG_KEYS)
