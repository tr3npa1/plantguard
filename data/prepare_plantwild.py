"""
Prepare PlantWild_v2 for expanded PlantGuard training.

This script does NOT train a model.

It does four things:
1. Reads the manually reviewed PlantWild -> PlantGuard mapping CSV.
2. Validates that the mapping is complete and clean.
3. Creates PlantWild train/val/test CSV splits.
4. Creates the expanded PlantGuard class list.

Expected PlantWild folder structure:
    data/raw/plantwild_v2/
        apple scab/
        tomato early blight/
        zucchini powdery mildew/
        ...

Expected mapping CSV:
    data/metadata/plantwild_v2_to_plantvillage_mapping.csv

Required mapping CSV columns:
    plantwild_label
    mapping_status
    mapped_label

Important:
    plantwild_label should keep underscore format, for example:
        tomato_early_blight

    The actual PlantWild folders use spaces, for example:
        tomato early blight

    This script preserves the underscore label names in CSV outputs, but uses
    label.replace("_", " ") only to locate folders on disk.
"""

import json
import random
from pathlib import Path

import pandas as pd


# ---------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

PLANTWILD_DIR = PROJECT_ROOT / "data" / "raw" / "plantwild_v2"
PLANTVILLAGE_TRAIN_DIR = PROJECT_ROOT / "data" / "train"

METADATA_DIR = PROJECT_ROOT / "data" / "metadata"
SPLITS_DIR = PROJECT_ROOT / "data" / "splits"

MAPPING_CSV = METADATA_DIR / "plantwild_v2_to_plantvillage_mapping.csv"

PLANTWILD_TRAIN_CSV = SPLITS_DIR / "plantwild_train.csv"
PLANTWILD_VAL_CSV = SPLITS_DIR / "plantwild_val.csv"
PLANTWILD_TEST_CSV = SPLITS_DIR / "plantwild_test.csv"

EXPANDED_CLASS_NAMES_JSON = METADATA_DIR / "plantguard_expanded_class_names.json"
EXPANDED_LABELS_CSV = METADATA_DIR / "plantguard_expanded_labels.csv"
SPLIT_SUMMARY_CSV = METADATA_DIR / "plantwild_split_summary.csv"

# ---------------------------------------------------------------------
# Split settings
# ---------------------------------------------------------------------

SEED = 42

PLANTWILD_TRAIN_RATIO = 0.80
PLANTWILD_VAL_RATIO = 0.10
PLANTWILD_TEST_RATIO = 0.10

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

VALID_MAPPING_STATUSES = {
    "existing_plantguard_label",
    "new_label",
}


# ---------------------------------------------------------------------
# Loading and validation
# ---------------------------------------------------------------------


def get_plantvillage_classes():
    """
    Load the original 38 PlantGuard/PlantVillage class names.

    Returns:
        Sorted list of PlantVillage class names.

    Why sorted:
        torchvision.datasets.ImageFolder uses alphabetical sorting, and this
        should match the class order used during the original training.
    """
    if not PLANTVILLAGE_TRAIN_DIR.exists():
        raise FileNotFoundError(
            f"PlantVillage train folder not found: {PLANTVILLAGE_TRAIN_DIR}"
        )
    
    class_names = sorted(
        path.name
        for path in PLANTVILLAGE_TRAIN_DIR.iterdir()
        if path.is_dir()
    )

    if not class_names:
        raise RuntimeError(
            f"No class folders found inside: {PLANTVILLAGE_TRAIN_DIR}"
        )

    return class_names


def load_mapping_csv():
    """
    Load the manually reviewed PlantWild mapping CSV.

    Returns:
        pandas DataFrame.
    """
    if not MAPPING_CSV.exists():
        raise FileNotFoundError(
            f"Mapping CSV not found: {MAPPING_CSV}\n"
            "Place your final CSV at data/metadata/"
            "plantwild_v2_to_plantvillage_mapping.csv"
        )
    
    mapping_df = pd.read_csv(MAPPING_CSV, dtype=str, keep_default_na=False)

     # Strip accidental spaces around all text fields.
    for column in mapping_df.columns:
        mapping_df[column] = mapping_df[column].astype(str).str.strip()

    return mapping_df


def validate_mapping_df(mapping_df, plantvillage_classes):
    """
    Validate the mapping CSV before creating splits.

    Checks:
        - required columns exist
        - 115 unique PlantWild labels
        - no missing values in key columns
        - no duplicate PlantWild labels
        - no unsure rows
        - existing mappings point to real PlantVillage labels
        - new labels map to themselves

    Args:
        mapping_df:
            Mapping DataFrame.
        plantvillage_classes:
            Original PlantGuard class list.
    """
    required_columns = {
        "plantwild_label",
        "mapping_status",
        "mapped_label",
    }
    missing_columns = required_columns - set(mapping_df.columns)

    if missing_columns:
        raise ValueError(
            f"Mapping CSV is missing required columns: {sorted(missing_columns)}"
        )

    if len(mapping_df) != 115:
        raise ValueError(
            f"Expected 115 PlantWild rows, found {len(mapping_df)}."
        )
    
    duplicate_labels = mapping_df[
        mapping_df["plantwild_label"].duplicated(keep=False)
    ]

    if not duplicate_labels.empty:
        raise ValueError(
            "Duplicate plantwild_label values found:\n"
            f"{duplicate_labels.to_string(index=False)}"
        )

    for column in ["plantwild_label", "mapping_status", "mapped_label"]:
        empty_rows = mapping_df[mapping_df[column] == ""]

        if not empty_rows.empty:
            raise ValueError(
                f"Empty values found in column '{column}':\n"
                f"{empty_rows.to_string(index=False)}"
            )
        
    invalid_status_rows = mapping_df[
        ~mapping_df["mapping_status"].isin(VALID_MAPPING_STATUSES)
    ]

    if not invalid_status_rows.empty:
        raise ValueError(
            "Invalid mapping_status values found:\n"
            f"{invalid_status_rows.to_string(index=False)}"
        )
    
    plantvillage_set = set(plantvillage_classes)

    existing_rows = mapping_df[
        mapping_df["mapping_status"] == "existing_plantguard_label"
    ]

    bad_existing_rows = existing_rows[
        ~existing_rows["mapped_label"].isin(plantvillage_set)
    ]

    if not bad_existing_rows.empty:
        raise ValueError(
            "Some existing_plantguard_label rows map to labels that are not "
            "in the original PlantVillage class list:\n"
            f"{bad_existing_rows.to_string(index=False)}"
        )
    
    new_rows = mapping_df[mapping_df["mapping_status"] == "new_label"]

    bad_new_rows = new_rows[
        new_rows["mapped_label"] != new_rows["plantwild_label"]
    ]

    if not bad_new_rows.empty:
        raise ValueError(
            "Some new_label rows do not map to themselves:\n"
            f"{bad_new_rows.to_string(index=False)}"
        )


# ---------------------------------------------------------------------
# Label-space creation
# ---------------------------------------------------------------------

def build_expanded_class_names(mapping_df, plantvillage_classes):
    """
    Build expanded PlantGuard class list.

    Rule:
        - original 38 PlantGuard labels remain at indices 0-37
        - new PlantWild labels are appended from index 38 onward

    Args:
        mapping_df:
            Validated mapping DataFrame.
        plantvillage_classes:
            Original 38 PlantGuard labels.

    Returns:
        Expanded class-name list.
    """
    new_labels = sorted(
        mapping_df.loc[
            mapping_df["mapping_status"] == "new_label",
            "mapped_label",
        ].unique()
    )

    expanded_class_names = list(plantvillage_classes) + new_labels

    return expanded_class_names


def save_expanded_label_files(expanded_class_names, plantvillage_classes):
    """
    Save expanded class-name files.

    Args:
        expanded_class_names:
            Full expanded class list.
        plantvillage_classes:
            Original 38 labels.
    """
    METADATA_DIR.mkdir(parents=True, exist_ok=True)

    with EXPANDED_CLASS_NAMES_JSON.open("w", encoding="utf-8") as file:
        json.dump(expanded_class_names, file, indent=2)

    rows = []

    plantvillage_set = set(plantvillage_classes)

    for index, class_name in enumerate(expanded_class_names):
        if class_name in plantvillage_set:
            label_origin = "original_plantguard"
        else:
            label_origin = "plantwild_new"

        rows.append(
            {
                "label_index": index,
                "class_name" : class_name,
                "label_origin" : label_origin,
            }
        )

        pd.DataFrame(rows).to_csv(EXPANDED_LABELS_CSV, index=False)



# ---------------------------------------------------------------------
# Image collection and splitting
# ---------------------------------------------------------------------

def resolve_plantwild_folder(plantwild_label):
    """
    Resolve a PlantWild label key to the actual folder on disk.

    The mapping CSV uses underscore labels:
        tomato_early_blight

    PlantWild folders use spaces:
        tomato early blight

    Args:
        plantwild_label:
            Label key from mapping CSV.

    Returns:
        Path to the class folder.
    """

    candidates = [
        PLANTWILD_DIR / plantwild_label.replace("_", " "),
        PLANTWILD_DIR / plantwild_label,
    ]

    for candidate in candidates:
        if candidate.exists() and candidate.is_dir():
            return candidate
        
    raise FileNotFoundError(
        f"Could not find PlantWild folder for label: {plantwild_label}\n"
        "Tried:\n"
        + "\n".join(f"- {candidate}" for candidate in candidates)
    )


def collect_images_for_folder(folder_path):
    """
    Collect image paths inside one PlantWild class folder.

    Args:
        folder_path:
            PlantWild class folder.

    Returns:
        Sorted list of image paths.
    """
    image_paths = [
        path
        for path in folder_path.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    ]

    return sorted(image_paths)


def split_image_paths(image_paths, rng):
    """
    Split image paths into train/val/test.

    This is class-wise splitting, so every PlantWild source class is split
    independently.

    Small-class handling:
        - 1 image: train only
        - 2 images: 1 train, 1 test
        - 3+ images: at least 1 val and 1 test

    Args:
        image_paths:
            List of image paths for one class.
        rng:
            random.Random instance.

    Returns:
        train_paths, val_paths, test_paths
    """
    image_paths = list(image_paths) 
    rng.shuffle(image_paths)

    total = len(image_paths)

    if total == 0:
        return [],[],[]
    
    if total == 1:
        return image_paths, [] , []
    
    if total == 2:
        return image_paths[:1], [], image_paths[1:]
    
    val_count = max(1, int(round(total * PLANTWILD_VAL_RATIO)))
    test_count = max(1, int(round(total * PLANTWILD_TEST_RATIO)))

    if val_count + test_count >= total:
        val_count = 1
        test_count = 1

    train_count = total - val_count - test_count

    train_paths = image_paths[:train_count]
    val_paths = image_paths[train_count:train_count+val_count]
    test_paths = image_paths[train_count + val_count:]

    return train_paths, val_paths, test_paths


def make_split_rows(
    image_paths,
    plantwild_label,
    mapped_label,
    mapping_status,
    class_to_idx,
):
    """
    Convert image paths into CSV rows.

    Args:
        image_paths:
            Image paths belonging to one split.
        plantwild_label:
            Original PlantWild label key.
        mapped_label:
            Final expanded PlantGuard label.
        mapping_status:
            existing_plantguard_label or new_label.
        class_to_idx:
            Expanded class-name to index dictionary.

    Returns:
        List of dictionaries.
    """
    label_index = class_to_idx[mapped_label]

    rows = []

    for image_path in image_paths:
        rows.append(
            {
                "image_path": image_path.relative_to(PROJECT_ROOT).as_posix(),
                "source_dataset": "plantwild_v2",
                "plantwild_label": plantwild_label,
                "plantguard_label": mapped_label,
                "label_index": label_index,
                "mapping_status": mapping_status,
            }
        )

    return rows


def create_plantwild_splits(mapping_df, expanded_class_names):
    """
    Create train/val/test split DataFrames for PlantWild_v2.

    Args:
        mapping_df:
            Validated mapping DataFrame.
        expanded_class_names:
            Expanded PlantGuard class list.

    Returns:
        train_df, val_df, test_df, summary_df
    """
    rng = random.Random(SEED)

    class_to_idx = {
        class_name : index
        for index, class_name in enumerate(expanded_class_names)
    }

    train_rows = []
    val_rows = []
    test_rows = []
    summary_rows = []

    for _, row in mapping_df.sort_values("plantwild_label").iterrows():
        plantwild_label = row["plantwild_label"]
        mapped_label = row["mapped_label"]
        mapping_status = row["mapping_status"]

        folder_path = resolve_plantwild_folder(plantwild_label)
        image_paths = collect_images_for_folder(folder_path)

        train_paths, val_paths, test_paths = split_image_paths(
            image_paths = image_paths,
            rng=rng,
        )

        train_rows.extend(
            make_split_rows(
                image_paths=train_paths,
                plantwild_label=plantwild_label,
                mapped_label=mapped_label,
                mapping_status=mapping_status,
                class_to_idx=class_to_idx,
            )
        )

        val_rows.extend(
            make_split_rows(
                image_paths=val_paths,
                plantwild_label=plantwild_label,
                mapped_label=mapped_label,
                mapping_status=mapping_status,
                class_to_idx=class_to_idx,
            )
        )

        test_rows.extend(
            make_split_rows(
                image_paths=test_paths,
                plantwild_label=plantwild_label,
                mapped_label=mapped_label,
                mapping_status=mapping_status,
                class_to_idx=class_to_idx,
            )
        )

        summary_rows.append(
            {
                "plantwild_label": plantwild_label,
                "plantguard_label": mapped_label,
                "mapping_status": mapping_status,
                "total_images": len(image_paths),
                "train_images": len(train_paths),
                "val_images": len(val_paths),
                "test_images": len(test_paths),
            }
        )

    train_df = pd.DataFrame(train_rows)
    val_df = pd.DataFrame(val_rows)
    test_df = pd.DataFrame(test_rows)
    summary_df = pd.DataFrame(summary_rows)

    return train_df, val_df, test_df, summary_df


def save_split_files(train_df, val_df, test_df, summary_df):
    """
    Save PlantWild split CSV files.

    Args:
        train_df:
            Training split DataFrame.
        val_df:
            Validation split DataFrame.
        test_df:
            Test split DataFrame.
        summary_df:
            Per-class split summary.
    """
    SPLITS_DIR.mkdir(parents=True, exist_ok=True)
    METADATA_DIR.mkdir(parents=True, exist_ok=True)

    train_df.to_csv(PLANTWILD_TRAIN_CSV, index=False)
    val_df.to_csv(PLANTWILD_VAL_CSV, index=False)
    test_df.to_csv(PLANTWILD_TEST_CSV, index=False)
    summary_df.to_csv(SPLIT_SUMMARY_CSV, index=False)


# ---------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------

def print_summary(
    mapping_df,
    plantvillage_classes,
    expanded_class_names,
    train_df,
    val_df,
    test_df,
    summary_df,
):
    """
    Print preparation summary.

    Args:
        mapping_df:
            Mapping DataFrame.
        plantvillage_classes:
            Original class list.
        expanded_class_names:
            Expanded class list.
        train_df:
            PlantWild train split.
        val_df:
            PlantWild validation split.
        test_df:
            PlantWild test split.
        summary_df:
            Per-class split summary.
    """
    existing_count = int(
        (mapping_df["mapping_status"] == "existing_plantguard_label").sum()
    )
    new_count = int((mapping_df["mapping_status"] == "new_label").sum())

    print("\nPlantWild_v2 preparation complete")
    print("=" * 80)

    print("\nLabel mapping:")
    print(f"PlantWild labels:              {len(mapping_df)}")
    print(f"Mapped to original PlantGuard: {existing_count}")
    print(f"New expanded labels:           {new_count}")

    print("\nExpanded label space:")
    print(f"Original PlantGuard labels: {len(plantvillage_classes)}")
    print(f"Expanded PlantGuard labels: {len(expanded_class_names)}")
    print("Old label indices:          0-37")
    print(f"New label indices:          38-{len(expanded_class_names) - 1}")

    print("\nPlantWild split sizes:")
    print(f"Train images: {len(train_df)}")
    print(f"Val images:   {len(val_df)}")
    print(f"Test images:  {len(test_df)}")
    print(f"Total images: {len(train_df) + len(val_df) + len(test_df)}")

    print("\nUnique final labels in PlantWild splits:")
    print(f"Train labels: {train_df['plantguard_label'].nunique()}")
    print(f"Val labels:   {val_df['plantguard_label'].nunique()}")
    print(f"Test labels:  {test_df['plantguard_label'].nunique()}")

    print("\nSmallest classes after split:")
    print(
        summary_df.sort_values("total_images", ascending=True)
        .head(15)
        .to_string(index=False)
    )

    print("\nSaved files:")
    print(f"- {PLANTWILD_TRAIN_CSV}")
    print(f"- {PLANTWILD_VAL_CSV}")
    print(f"- {PLANTWILD_TEST_CSV}")
    print(f"- {SPLIT_SUMMARY_CSV}")
    print(f"- {EXPANDED_CLASS_NAMES_JSON}")
    print(f"- {EXPANDED_LABELS_CSV}")


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main():
    """
    Prepare PlantWild_v2 split CSVs and expanded PlantGuard labels.
    """
    plantvillage_classes = get_plantvillage_classes()

    mapping_df = load_mapping_csv()

    validate_mapping_df(
        mapping_df=mapping_df,
        plantvillage_classes=plantvillage_classes,
    )

    expanded_class_names = build_expanded_class_names(
        mapping_df=mapping_df,
        plantvillage_classes=plantvillage_classes
    )

    save_expanded_label_files(
        expanded_class_names=expanded_class_names,
        plantvillage_classes=plantvillage_classes,
    )

    train_df, val_df, test_df, summary_df = create_plantwild_splits(
        mapping_df=mapping_df,
        expanded_class_names=expanded_class_names,
    )

    save_split_files(
        train_df=train_df,
        val_df=val_df,
        test_df=test_df,
        summary_df=summary_df,
    )

    print_summary(
        mapping_df=mapping_df,
        plantvillage_classes=plantvillage_classes,
        expanded_class_names=expanded_class_names,
        train_df=train_df,
        val_df=val_df,
        test_df=test_df,
        summary_df=summary_df,
    )


if __name__ == "__main__":
    main()