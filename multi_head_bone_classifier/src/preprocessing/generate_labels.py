import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
import pickle
from pathlib import Path
import matplotlib.pyplot as plt
import json

ANNOTATIONS_DIR = (
    "/home/jvrielink/data_hdd/AIML_rot_corrected/annotations/merged_labels_age"
)

UNKNOWN_FILES = "src/preprocessing/label_files/unknown_files.csv"
KNOWN_FILES = "src/preprocessing/label_files/known_files.csv"
CLASSES = "src/preprocessing/label_files/merged_classes.csv"
RANDOM_STATE = 42
TRAIN_SIZE = 0.8
VAL_SIZE = 0.1
TEST_SIZE = 0.1


def assign_to_class(known_files_df, classes_df):
    """
    Function to transfrom the various different classes to the 13 body parts and 3 views
    """
    results = {"path": [], "body_part": [], "view": [], "age_scalar": []}
    # Loop over the rows in the first DataFrame
    for index, row in known_files_df.iterrows():
        # Get the values for 'Path' and 'StudyDescription'
        path = row["Path"]
        study_description = row["StudyDescription"]

        # Find the value of 'StudyDescription' in the classes DataFrame
        matching_row = classes_df[classes_df["StudyDescription"] == study_description]

        # If a match is found, extract 'body_part' and 'view' values
        if not matching_row.empty:
            body_part = matching_row.iloc[0]["body_part"]
            if body_part != "REMOVE":
                view = matching_row.iloc[0]["view"]
                age_scalar = float(row["AgeYears"]) / 100

                # Add the values to the result
                results["path"].append(path)
                results["body_part"].append(body_part)
                results["view"].append(view)
                results["age_scalar"].append(age_scalar)
        else:
            print("WARNING: unidentified row")

    # return the results as a df
    return pd.DataFrame(results)


def generate_labels(known_files_df, classes_df):
    # Generate the 13 different body part classes and 3 different view classes
    labels = assign_to_class(known_files_df, classes_df)

    # Initialize OneHotEncoder for "body_part" and "view" columns
    encoder = OneHotEncoder(sparse_output=False)

    # Fit and transform "body_part" and "view" columns from AIML data
    body_part_encoded = encoder.fit_transform(labels[["body_part"]])
    body_part_names = encoder.get_feature_names_out().tolist()

    view_encoded = encoder.fit_transform(labels[["view"]])
    view_names = encoder.get_feature_names_out().tolist()

    # Append the one-hot encoded columns to the AIML DataFrame
    labels.insert(1, "body_part_encoded", body_part_encoded.tolist())
    labels.insert(2, "view_encoded", view_encoded.tolist())
    return labels, body_part_names, view_names


def split_train_val_test(labels, random_state, train_size, val_size, test_size):
    # Split the DataFrame into train (80%), validation (10%), and test (10%) sets
    train_df, temp_df = train_test_split(
        labels,
        test_size=(1 - train_size),
        random_state=random_state,
        stratify=labels["body_part"],
    )
    validation_df, test_df = train_test_split(
        temp_df,
        test_size=val_size / (1 - train_size),
        random_state=random_state,
        stratify=temp_df["body_part"],
    )

    return train_df, validation_df, test_df


def plot(df, collumn, split, save_dir, tail=False):
    # Plot a histogram for "body_part" counts
    df_counts = df[collumn].value_counts().reset_index()
    plt.figure(figsize=(10, 6))
    if tail:
        plt.bar(df_counts[collumn].tail(10), df_counts["count"].tail(10))
    else:
        plt.bar(df_counts[collumn], df_counts["count"])

    plt.xlabel(collumn)
    plt.ylabel("Frequency Count")
    title = split + "_Frequency_Count_of_" + collumn
    plt.title("Frequency Count of Body Parts")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    histogram_file = save_dir + "/" + (title + ".png")  # Use data_dir as a prefix
    plt.savefig(histogram_file)
    plt.close()


def analyse(df, name, save_dir):
    size = f"{name} dataset size: {len(df)}"
    print(size)
    file = open(save_dir + f"/size_{name}.txt", "w")
    file.write(size)
    file.close()

    plot(df, "body_part", name, save_dir)
    plot(train_df, "view", name, save_dir)
    plot(train_df, "body_part", name + "_tail", save_dir, tail=True)


if __name__ == "__main__":
    Path(ANNOTATIONS_DIR).mkdir(parents=True, exist_ok=True)

    known_files_df = pd.read_csv(KNOWN_FILES)
    classes_df = pd.read_csv(CLASSES)
    labels, body_part_names, view_names = generate_labels(known_files_df, classes_df)
    train_df, validation_df, test_df = split_train_val_test(
        labels, RANDOM_STATE, TRAIN_SIZE, VAL_SIZE, TEST_SIZE
    )
    analyse(train_df, "train", ANNOTATIONS_DIR)
    analyse(validation_df, "validation", ANNOTATIONS_DIR)
    analyse(test_df, "test", ANNOTATIONS_DIR)

    save_dir = Path(ANNOTATIONS_DIR)
    labels.to_csv(save_dir / "all_labels_as.csv", index=False)

    train_df = train_df.drop(["body_part", "view"], axis=1)
    validation_df = validation_df.drop(["body_part", "view"], axis=1)
    test_df = test_df.drop(["body_part", "view"], axis=1)

    file = open((save_dir / "train.pkl"), "wb")
    pickle.dump(train_df, file)
    file.close()

    file = open((save_dir / "val.pkl"), "wb")
    pickle.dump(validation_df, file)
    file.close()

    file = open((save_dir / "test.pkl"), "wb")
    pickle.dump(test_df, file)
    file.close()

    with open(save_dir / "body_part_names.json", "w") as outfile:
        json.dump(body_part_names, outfile)

    with open(save_dir / "view_names.json", "w") as outfile:
        json.dump(view_names, outfile)

    unknown_files_df = pd.read_csv(UNKNOWN_FILES)
    unknonwn_paths = unknown_files_df["Path"]
    file = open((save_dir / "unknown.pkl"), "wb")
    pickle.dump(unknonwn_paths, file)
    file.close()
