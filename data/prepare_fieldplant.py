"""
Prepare FieldPlant as an external compatible classification test set.

FieldPlant is downloaded as a YOLO/object-detection style dataset:

    data/raw/fieldplant/
        data.yaml
        train/images/
        train/labels/

This script uses the manually reviewed mapping file:

    data/metadata/fieldplant_to_plantguard_mapping.csv

and creates:

    data/splits/fieldplant_test.csv
    data/metadata/fieldplant_split_summary.csv

FieldPlant is NOT used for training. It is only used as an external final
evaluation dataset.
"""

import json
from pathlib import Path

import pandas as pd
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]

FIELDPLANT_DIR = PROJECT_ROOT / "data" / "raw" / "fieldplant"
FIELDPLANT_YAML = FIELDPLANT_DIR / "data.yaml"
FIELDPLANT_IMAGES_DIR = FIELDPLANT_DIR / "train" / "images"
FIELDPLANT_LABELS_DIR = FIELDPLANT_DIR / "train" / "labels"

METADATA_DIR = PROJECT_ROOT / "data" / "metadata"
SPLITS_DIR = PROJECT_ROOT / "data" / "splits"

EXPANDED_CLASS_NAMES_JSON = METADATA_DIR / "plantguard_expanded_class_names.json"
FIELDPLANT_MAPPING_CSV = METADATA_DIR / "fieldplant_to_plantguard_mapping.csv"

FIELDPLANT_TEST_CSV = SPLITS_DIR / "fieldplant_test.csv"
FIELDPLANT_SUMMARY_CSV = METADATA_DIR / "fieldplant_split_summary.csv"

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
EXPECTED_NUM_CLASSES = 132

def load_expanded_class_names():
    """
    Load the final expanded PlantGuard class list.

    The expanded list should contain the original 38 PlantVillage labels first,
    followed by the new PlantWild labels.
    """
    if not EXPANDED_CLASS_NAMES_JSON.exists():
        raise FileNotFoundError(
            f"Expanded class names file not found: {EXPANDED_CLASS_NAMES_JSON}"
        )

    with EXPANDED_CLASS_NAMES_JSON.open("r", encoding="utf-8") as file:
        class_names = json.load(file)

    if not isinstance(class_names, list):
        raise TypeError("Expanded class names JSON must contain a list.")

    if len(class_names) != EXPECTED_NUM_CLASSES:
        raise ValueError(
            f"Expected {EXPECTED_NUM_CLASSES} expanded PlantGuard labels, "
            f"found {len(class_names)} labels."
        )

    return class_names


def load_fieldplant_class_names():
    """
    Load FieldPlant class names from data.yaml.

    Returns:
        List of class names where list index == YOLO class id.
    """
    if not FIELDPLANT_YAML.exists():
        raise FileNotFoundError(f"FieldPlant data.yaml not found : {FIELDPLANT_YAML}")
    
    with FIELDPLANT_YAML.open('r', encoding='utf-8') as file:
        data = yaml.safe_load(file)

    names = data.get("names")

    if isinstance(names,dict):
        names = [names[index] for index in sorted(names)]

    if not isinstance(names,list):
        raise ValueError("Could not parse FieldPlant class names from data.yaml.")
    
    expected_nc = data.get("nc")

    if expected_nc is not None and int(expected_nc) != len(names):
        raise ValueError(
            f"data.yaml says nc={expected_nc}, but found {len(names)} class names."
        )

    return names


def load_and_validate_mapping(fieldplant_class_names, expanded_class_names):
    """
    Load and validate the manually reviewed FieldPlant mapping CSV.

    Required columns:
        fieldplant_class_id
        fieldplant_label
        mapping_status
        mapped_label
        mapping_note

    mapping_status must be either:
        mapped
        unmapped
    """
    if not FIELDPLANT_MAPPING_CSV.exists():
        raise FileNotFoundError(
            f"Mapping CSV not found: {FIELDPLANT_MAPPING_CSV}\n"
            "Create this file manually before running prepare_fieldplant.py."
        )
    
    mapping_df = pd.read_csv(
        FIELDPLANT_MAPPING_CSV,
        dtype=str,
        keep_default_na=False,
    )

    required_columns = {
        "fieldplant_class_id",
        "fieldplant_label",
        "mapping_status",
        "mapped_label",
        "mapping_note",
    }

    missing_columns = required_columns - set(mapping_df.columns)

    if missing_columns:
        raise ValueError(
            f"Mapping CSV is missing required columns: {sorted(missing_columns)}"
        )

    for column in mapping_df.columns:
        mapping_df[column] = mapping_df[column].astype(str).str.strip()

    if len(mapping_df) != len(fieldplant_class_names):
        raise ValueError(
            f"Mapping CSV has {len(mapping_df)} rows, but FieldPlant data.yaml has "
            f"{len(fieldplant_class_names)} classes."
        )
    
    valid_statuses = {"mapped", "unmapped"}
    invalid_status_rows = mapping_df[
        ~mapping_df["mapping_status"].isin(valid_statuses)
    ]

    if not invalid_status_rows.empty:
        raise ValueError(
            "Invalid mapping_status values found:\n"
            f"{invalid_status_rows.to_string(index=False)}"
        )
    
    expanded_set = set(expanded_class_names)

    for _,row in mapping_df.iterrows():
        class_id = int(row["fieldplant_class_id"])
        fieldplant_label = row["fieldplant_label"]
        mapping_status = row["mapping_status"]
        mapped_label = row["mapped_label"]

        if class_id < 0 or class_id >= len(fieldplant_class_names):
            raise ValueError(f"Invalid FieldPlant class id: {class_id}")
        
        expected_label = fieldplant_class_names[class_id]

        if fieldplant_label != expected_label:
            raise ValueError(
                "Mapping CSV class id/label mismatch:\n"
                f"CSV row says id {class_id} = {fieldplant_label}\n"
                f"data.yaml says id {class_id} = {expected_label}"
            )
        
        if mapping_status == "mapped":
            if not mapped_label:
                raise ValueError(
                    f"Class id {class_id} is marked mapped but mapped_label is empty."
                )

            if mapped_label not in expanded_set:
                raise ValueError(
                    f"Mapped label not found in expanded PlantGuard labels:\n"
                    f"{mapped_label}"
                )

        if mapping_status == "unmapped" and mapped_label:
            raise ValueError(
                f"Class id {class_id} is marked unmapped but mapped_label is not empty:\n"
                f"{mapped_label}"
            )

    return mapping_df


def find_image_files():
    """
    Find all FieldPlant images under train/images.
    """
    if not FIELDPLANT_IMAGES_DIR.exists():
        raise FileNotFoundError(
            f"FieldPlant images directory not found: {FIELDPLANT_IMAGES_DIR}"
        )

    image_paths = [
        path
        for path in FIELDPLANT_IMAGES_DIR.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    ]

    return sorted(image_paths)


def read_yolo_class_ids(label_path):
    """
    Read YOLO class ids from a label file.

    YOLO rows look like:
        class_id x_center y_center width height

    Returns:
        List of integer class ids.
    """
    class_ids = []

    if not label_path.exists():
        return class_ids
    
    with label_path.open("r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()

            if not line:
                continue

            parts = line.split()

            if len(parts)<5:
                continue

            class_id = int(float(parts[0]))
            class_ids.append(class_id)

    return class_ids


def create_fieldplant_test_csv(mapping_df, fieldplant_class_names, expanded_class_names):
    """
    Create a classification-style CSV from FieldPlant YOLO annotations.

    Rules:
        - Missing label file -> skip
        - Empty label file -> skip
        - Multiple different classes in one image -> skip
        - Multiple boxes of the same class -> keep
        - Unmapped class -> skip
        - Mapped class -> keep as external test image
    """
    SPLITS_DIR.mkdir(parents=True,exist_ok=True)
    METADATA_DIR.mkdir(parents=True, exist_ok=True)

    class_to_idx = {
        class_name : index 
        for index,class_name in enumerate(expanded_class_names)
    }

    mapping_by_id = {}

    for _,row in mapping_df.iterrows():
        class_id = int(row["fieldplant_class_id"])

        mapping_by_id[class_id] = {
            "fieldplant_label": row["fieldplant_label"],
            "mapping_status": row["mapping_status"],
            "mapped_label": row["mapped_label"],
            "mapping_note": row["mapping_note"],
        }

    image_paths = find_image_files()

    rows = []

    skipped = {
        "missing_label_file": 0,
        "empty_label_file": 0,
        "mixed_classes": 0,
        "invalid_class_id": 0,
        "unmapped_class": 0,
    }

    for image_path in image_paths:
        label_path = FIELDPLANT_LABELS_DIR / f"{image_path.stem}.txt"

        if not label_path.exists():
            skipped["missing_label_file"]+=1
            continue
        
        class_ids = read_yolo_class_ids(label_path)

        if not class_ids:
            skipped["empty_label_file"]+=1
            continue

        unique_class_ids = sorted(set(class_ids))

        if len(unique_class_ids) != 1:
            skipped["mixed_classes"] += 1
            continue

        fieldplant_class_id = unique_class_ids[0]

        if fieldplant_class_id < 0 or fieldplant_class_id >= len(fieldplant_class_names):
            skipped["invalid_class_id"] += 1
            continue

        mapping = mapping_by_id[fieldplant_class_id]

        if mapping["mapping_status"] != "mapped":
            skipped["unmapped_class"] += 1
            continue

        plantguard_label = mapping["mapped_label"]
        label_index = class_to_idx[plantguard_label]

        rows.append(
            {
                "image_path": image_path.relative_to(PROJECT_ROOT).as_posix(),
                "label_index": label_index,
                "class_name": plantguard_label,
                "source_dataset": "fieldplant",
                "fieldplant_class_id": fieldplant_class_id,
                "fieldplant_label": mapping["fieldplant_label"],
                "mapping_note": mapping["mapping_note"],
                "num_yolo_objects": len(class_ids),
            }
        )

    fieldplant_df = pd.DataFrame(rows)

    if fieldplant_df.empty:
        raise RuntimeError(
            "No compatible FieldPlant images were kept. "
            "Check fieldplant_to_plantguard_mapping.csv."
        )

    fieldplant_df.to_csv(FIELDPLANT_TEST_CSV, index=False)

    summary_rows = [
        {
            "category": "total_images",
            "count": len(image_paths),
        },
        {
            "category": "kept_compatible_images",
            "count": len(fieldplant_df),
        },
    ]

    for category, count in skipped.items():
        summary_rows.append(
            {
                "category": category,
                "count": count,
            }
        )

    mapped_class_counts = (
        fieldplant_df["class_name"]
        .value_counts()
        .sort_index()
        .reset_index()
    )
    mapped_class_counts.columns = ["class_name", "count"]

    for _, row in mapped_class_counts.iterrows():
        summary_rows.append(
            {
                "category": f"class::{row['class_name']}",
                "count": int(row["count"]),
            }
        )

    pd.DataFrame(summary_rows).to_csv(FIELDPLANT_SUMMARY_CSV, index=False)

    print("\nFieldPlant preparation complete.")
    print(f"Saved test CSV: {FIELDPLANT_TEST_CSV}")
    print(f"Saved summary:  {FIELDPLANT_SUMMARY_CSV}")

    print("\nSummary:")
    print(f"Total images:           {len(image_paths)}")
    print(f"Kept compatible images: {len(fieldplant_df)}")

    for category, count in skipped.items():
        print(f"Skipped {category}: {count}")

    print("\nKept class distribution:")
    print(mapped_class_counts.to_string(index=False))


def main():
    expanded_class_names = load_expanded_class_names()
    fieldplant_class_names = load_fieldplant_class_names()

    print(f"Expanded PlantGuard labels: {len(expanded_class_names)}")
    print(f"FieldPlant labels:          {len(fieldplant_class_names)}")

    mapping_df = load_and_validate_mapping(
        fieldplant_class_names=fieldplant_class_names,
        expanded_class_names=expanded_class_names,
    )

    create_fieldplant_test_csv(
        mapping_df=mapping_df,
        fieldplant_class_names=fieldplant_class_names,
        expanded_class_names=expanded_class_names,
    )


if __name__ == "__main__":
    main()
    