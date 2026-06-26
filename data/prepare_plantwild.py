"""
Prepare PlantWild_v2 for expanded PlantGuard training.

This script does not train a model.

It prepares PlantWild_v2 for Stage C expanded-label training by:

1. Reading the manually reviewed PlantWild -> PlantGuard mapping CSV.
2. Validating that the mapping is complete and clean.
3. Creating PlantWild train/validation/test CSV split files.
4. Creating the expanded PlantGuard class list.

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

Important label convention:
    plantwild_label should use underscore format, for example:

        tomato_early_blight

    The actual PlantWild folders may use spaces, for example:

        tomato early blight

    This script preserves underscore labels in CSV outputs, but uses
    label.replace("_", " ") when locating folders on disk.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

import pandas as pd


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

SEED = 42

PLANTWILD_TRAIN_RATIO = 0.80
PLANTWILD_VAL_RATIO = 0.10
PLANTWILD_TEST_RATIO = 0.10

EXPECTED_PLANTWILD_CLASSES = 115

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

VALID_MAPPING_STATUSES = {
    "existing_plantguard_label",
    "new_label",
}

REQUIRED_MAPPING_COLUMNS = {
    "plantwild_label",
    "mapping_status",
    "mapped_label",
}

SPLIT_COLUMNS = [
    "image_path",
    "source_dataset",
    "plantwild_label",
    "plantguard_label",
    "label_index",
    "mapping_status",
]


def is_image_file(path: Path) -> bool:
    """
    Check whether a path is a supported image file.

    Args:
        path:
            File path.

    Returns:
        True if the path is a supported image file.
    """
    return path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS


def validate_split_ratios() -> None:
    """
    Validate that PlantWild train/val/test ratios sum to 1.
    """
    ratio_sum = PLANTWILD_TRAIN_RATIO + PLANTWILD_VAL_RATIO + PLANTWILD_TEST_RATIO

    if abs(ratio_sum - 1.0) > 1e-6:
        raise ValueError(
            "PlantWild split ratios must sum to 1. "
            f"Got: {ratio_sum}"
        )


def get_plantvillage_classes() -> list[str]:
    """
    Load the original 38 PlantGuard/PlantVillage class names.

    Returns:
        Sorted list of PlantVillage class names.

    Why sorted:
        The original PlantDiseaseDataset uses alphabetical sorting. Keeping the
        same order here ensures that expanded labels 0-37 match the original
        PlantVillage checkpoint heads.
    """
    if not PLANTVILLAGE_TRAIN_DIR.exists():
        raise FileNotFoundError(
            f"PlantVillage train folder not found: {PLANTVILLAGE_TRAIN_DIR}\n"
            "Run: python data/download.py --dataset plantvillage"
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


def load_mapping_csv() -> pd.DataFrame:
    """
    Load the manually reviewed PlantWild mapping CSV.

    Returns:
        Mapping dataframe with all string columns stripped.
    """
    if not MAPPING_CSV.exists():
        raise FileNotFoundError(
            f"Mapping CSV not found: {MAPPING_CSV}\n"
            "Place your final CSV at:\n"
            "data/metadata/plantwild_v2_to_plantvillage_mapping.csv"
        )

    mapping_df = pd.read_csv(
        MAPPING_CSV,
        dtype=str,
        keep_default_na=False,
    )

    for column in mapping_df.columns:
        mapping_df[column] = mapping_df[column].astype(str).str.strip()

    return mapping_df


def validate_mapping_df(
    mapping_df: pd.DataFrame,
    plantvillage_classes: list[str],
) -> None:
    """
    Validate the PlantWild mapping CSV before creating splits.

    Checks:
        - required columns exist
        - expected number of PlantWild labels exists
        - key columns are non-empty
        - PlantWild labels are unique
        - mapping_status values are valid
        - existing mappings point to real PlantVillage labels
        - new labels map to themselves
        - new labels do not collide with original PlantVillage labels

    Args:
        mapping_df:
            Mapping dataframe.
        plantvillage_classes:
            Original PlantGuard/PlantVillage class list.
    """
    missing_columns = REQUIRED_MAPPING_COLUMNS - set(mapping_df.columns)

    if missing_columns:
        raise ValueError(
            f"Mapping CSV is missing required columns: {sorted(missing_columns)}"
        )

    if len(mapping_df) != EXPECTED_PLANTWILD_CLASSES:
        raise ValueError(
            f"Expected {EXPECTED_PLANTWILD_CLASSES} PlantWild rows, "
            f"found {len(mapping_df)}."
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

    colliding_new_rows = new_rows[
        new_rows["mapped_label"].isin(plantvillage_set)
    ]

    if not colliding_new_rows.empty:
        raise ValueError(
            "Some new_label rows collide with existing PlantVillage labels:\n"
            f"{colliding_new_rows.to_string(index=False)}"
        )


def build_expanded_class_names(
    mapping_df: pd.DataFrame,
    plantvillage_classes: list[str],
) -> list[str]:
    """
    Build the expanded PlantGuard class list.

    Rule:
        - original PlantGuard/PlantVillage labels stay at indices 0-37
        - new PlantWild labels are appended from index 38 onward

    Args:
        mapping_df:
            Validated mapping dataframe.
        plantvillage_classes:
            Original PlantVillage labels.

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

    if len(expanded_class_names) != len(set(expanded_class_names)):
        duplicates = pd.Series(expanded_class_names)
        duplicates = duplicates[duplicates.duplicated(keep=False)].tolist()

        raise RuntimeError(
            "Expanded class-name list contains duplicates:\n"
            + "\n".join(f"- {label}" for label in sorted(set(duplicates)))
        )

    return expanded_class_names


def save_expanded_label_files(
    expanded_class_names: list[str],
    plantvillage_classes: list[str],
) -> None:
    """
    Save expanded class-name metadata files.

    Files created:
        data/metadata/plantguard_expanded_class_names.json
        data/metadata/plantguard_expanded_labels.csv

    Args:
        expanded_class_names:
            Full expanded class list.
        plantvillage_classes:
            Original PlantVillage labels.
    """
    METADATA_DIR.mkdir(parents=True, exist_ok=True)

    with EXPANDED_CLASS_NAMES_JSON.open("w", encoding="utf-8") as file:
        json.dump(expanded_class_names, file, indent=2)

    plantvillage_set = set(plantvillage_classes)

    rows = []

    for index, class_name in enumerate(expanded_class_names):
        label_origin = (
            "original_plantguard"
            if class_name in plantvillage_set
            else "plantwild_new"
        )

        rows.append(
            {
                "label_index": index,
                "class_name": class_name,
                "label_origin": label_origin,
            }
        )

    pd.DataFrame(rows).to_csv(EXPANDED_LABELS_CSV, index=False)


def resolve_plantwild_folder(plantwild_label: str) -> Path:
    """
    Resolve a PlantWild label key to the actual folder on disk.

    The mapping CSV uses underscore labels:

        tomato_early_blight

    PlantWild folders may use spaces:

        tomato early blight

    Args:
        plantwild_label:
            Label key from mapping CSV.

    Returns:
        Path to the class folder.
    """
    if not PLANTWILD_DIR.exists():
        raise FileNotFoundError(
            f"PlantWild directory not found: {PLANTWILD_DIR}\n"
            "Run: python data/download.py --dataset plantwild"
        )

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


def collect_images_for_folder(folder_path: Path) -> list[Path]:
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
        if is_image_file(path)
    ]

    return sorted(image_paths)


def split_image_paths(
    image_paths: list[Path],
    rng: random.Random,
) -> tuple[list[Path], list[Path], list[Path]]:
    """
    Split image paths into train/val/test for one class.

    Splitting is class-wise, so every PlantWild source class is split
    independently.

    Small-class handling:
        - 0 images: no split rows
        - 1 image: train only
        - 2 images: 1 train, 1 test
        - 3+ images: at least 1 validation and 1 test image

    Args:
        image_paths:
            Image paths for one class.
        rng:
            random.Random instance.

    Returns:
        train_paths:
            Training image paths.
        val_paths:
            Validation image paths.
        test_paths:
            Test image paths.
    """
    image_paths = list(image_paths)
    rng.shuffle(image_paths)

    total = len(image_paths)

    if total == 0:
        return [], [], []

    if total == 1:
        return image_paths, [], []

    if total == 2:
        return image_paths[:1], [], image_paths[1:]

    val_count = max(1, int(round(total * PLANTWILD_VAL_RATIO)))
    test_count = max(1, int(round(total * PLANTWILD_TEST_RATIO)))

    if val_count + test_count >= total:
        val_count = 1
        test_count = 1

    train_count = total - val_count - test_count

    train_paths = image_paths[:train_count]
    val_paths = image_paths[train_count : train_count + val_count]
    test_paths = image_paths[train_count + val_count :]

    return train_paths, val_paths, test_paths


def make_split_rows(
    image_paths: list[Path],
    plantwild_label: str,
    mapped_label: str,
    mapping_status: str,
    class_to_idx: dict[str, int],
) -> list[dict]:
    """
    Convert image paths into split CSV rows.

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
            Expanded class-name to label-index dictionary.

    Returns:
        List of row dictionaries.
    """
    if mapped_label not in class_to_idx:
        raise KeyError(f"Mapped label not found in expanded class list: {mapped_label}")

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


def create_plantwild_splits(
    mapping_df: pd.DataFrame,
    expanded_class_names: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Create train/val/test split dataframes for PlantWild_v2.

    Args:
        mapping_df:
            Validated mapping dataframe.
        expanded_class_names:
            Expanded PlantGuard class list.

    Returns:
        train_df:
            PlantWild training split dataframe.
        val_df:
            PlantWild validation split dataframe.
        test_df:
            PlantWild test split dataframe.
        summary_df:
            Per-class split summary dataframe.
    """
    rng = random.Random(SEED)

    class_to_idx = {
        class_name: index
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
            image_paths=image_paths,
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

    train_df = pd.DataFrame(train_rows, columns=SPLIT_COLUMNS)
    val_df = pd.DataFrame(val_rows, columns=SPLIT_COLUMNS)
    test_df = pd.DataFrame(test_rows, columns=SPLIT_COLUMNS)
    summary_df = pd.DataFrame(summary_rows)

    return train_df, val_df, test_df, summary_df


def save_split_files(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    summary_df: pd.DataFrame,
) -> None:
    """
    Save PlantWild split CSV files.

    Files created:
        data/splits/plantwild_train.csv
        data/splits/plantwild_val.csv
        data/splits/plantwild_test.csv
        data/metadata/plantwild_split_summary.csv

    Args:
        train_df:
            Training split dataframe.
        val_df:
            Validation split dataframe.
        test_df:
            Test split dataframe.
        summary_df:
            Per-class split summary dataframe.
    """
    SPLITS_DIR.mkdir(parents=True, exist_ok=True)
    METADATA_DIR.mkdir(parents=True, exist_ok=True)

    train_df.to_csv(PLANTWILD_TRAIN_CSV, index=False)
    val_df.to_csv(PLANTWILD_VAL_CSV, index=False)
    test_df.to_csv(PLANTWILD_TEST_CSV, index=False)
    summary_df.to_csv(SPLIT_SUMMARY_CSV, index=False)


def print_summary(
    mapping_df: pd.DataFrame,
    plantvillage_classes: list[str],
    expanded_class_names: list[str],
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    summary_df: pd.DataFrame,
) -> None:
    """
    Print PlantWild preparation summary.

    Args:
        mapping_df:
            Mapping dataframe.
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


def main() -> None:
    """
    Prepare PlantWild_v2 split CSVs and expanded PlantGuard labels.
    """
    validate_split_ratios()

    plantvillage_classes = get_plantvillage_classes()
    mapping_df = load_mapping_csv()

    validate_mapping_df(
        mapping_df=mapping_df,
        plantvillage_classes=plantvillage_classes,
    )

    expanded_class_names = build_expanded_class_names(
        mapping_df=mapping_df,
        plantvillage_classes=plantvillage_classes,
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