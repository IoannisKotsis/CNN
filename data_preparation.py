import json
import os
import pandas as pd
from pathlib import Path
import ast

#opens json and returns annotations
def load_annotations(json_path):
    with open(json_path, "r", encoding="utf-8") as file:
        annotations = json.load(file)
    return annotations

#builds the dataframes
def build_dataframe(annotations, images_folder_path):
    images_folder_path = Path(images_folder_path)

    # (προαιρετικό) βρες questionnaire_id απλά για print
    online_ad = None
    for z in annotations:
        if z.get("questionnaire_id") == "online-ad-2-part-2":
            online_ad = z.get("questionnaire_id")
            break

    rows = []
    for i in annotations:
        path = i.get("image_filepath")
        full_path = images_folder_path / path
        answers = i.get("answers", [])

        relevant_value = None
        social_media_channel_value = None
        creator_value = None
        logo_value = None

        for a in answers:
            if a.get("variable") == "is-relevant":
                relevant_value = a.get("answer")
                for b in answers:
                    if b.get("variable") == "creator":
                        creator_value = b.get("answer")
                    elif b.get("variable") == "social-media-channel":
                        social_media_channel_value = b.get("answer")
                    elif b.get("variable") == "shows-logo":
                        logo_value = b.get("answer")

        rows.append({
            "image_filepath": str(full_path),
            "is-relevant": relevant_value,
            "social-media-channel": social_media_channel_value,
            "creator": creator_value,
            "logo": logo_value
        })

    rows_df = pd.DataFrame(rows)
    yes_df = rows_df[rows_df["is-relevant"] == "Yes"]
    new_df = yes_df[["image_filepath", "social-media-channel", "creator", "logo"]]

    return new_df, online_ad

#creates the label maps of the 3 tasks
def create_label_maps(new_df):
    social_set = set(new_df["social-media-channel"])
    logo_set = set(new_df["logo"])

    creator_flattened_set = set()
    for x in new_df["creator"]:
        if isinstance(x, list):
            creator_flattened_set.update(x)
        else:
            creator_flattened_set.add(x)

    sorted_social = sorted(social_set)
    sorted_creator = sorted(creator_flattened_set)
    sorted_logo = sorted(logo_set)

    social_map = {s: i for i, s in enumerate(sorted_social)}
    creator_map = {s: i for i, s in enumerate(sorted_creator)}
    logo_map = {s: i for i, s in enumerate(sorted_logo)}

    label_maps = {
        "social": social_map,
        "creator": creator_map,
        "logo": logo_map
    }
    return label_maps

#dataframe shuffle->split->save csv->csv paths
def split_and_save_csv(new_df, csv_dir, train_split_pct, validation_split_pct, seed=18):
    csv_dir = Path(csv_dir)
    csv_dir.mkdir(parents=True, exist_ok=True)

    df_length = len(new_df)
    train_split = int(train_split_pct * df_length)
    validation_split = int(validation_split_pct * df_length)

    shuffled_df = new_df.sample(frac=1, random_state=seed)

    train_rows = shuffled_df[:train_split]
    validation_rows = shuffled_df[train_split:train_split + validation_split]
    test_rows = shuffled_df[train_split + validation_split:]

    train_csv = csv_dir / "train_csv.csv"
    val_csv = csv_dir / "validation_csv.csv"
    test_csv = csv_dir / "test_csv.csv"

    train_rows.to_csv(train_csv, index=False)
    validation_rows.to_csv(val_csv, index=False)
    test_rows.to_csv(test_csv, index=False)

    return str(train_csv), str(val_csv), str(test_csv)

#calls the previous functions
def prepare_data(json_path, images_folder_path, csv_dir,
                 train_split_pct, validation_split_pct, seed=18,
                 print_stats=True):
    annotations = load_annotations(json_path)
    new_df, online_ad = build_dataframe(annotations, images_folder_path)

    if print_stats:
        print(f"Online ad: {online_ad}")
        print(f"Number of relevant images: {len(new_df)}")

    label_maps = create_label_maps(new_df)
    train_csv, val_csv, test_csv = split_and_save_csv(
        new_df, csv_dir, train_split_pct, validation_split_pct, seed=seed
    )

    return train_csv, val_csv, test_csv, label_maps


