"""
Download, extract, inspect, and prepare datasets for PlantGuard.

Supported datasets:

1. PlantVillage
   - Downloaded from Kaggle.
   - Used as the original 38-class clean/lab-style dataset.
   - Split into:
        data/train/
        data/val/
        data/test/

2. PlantDoc
   - Downloaded from the official PlantDoc GitHub repository.
   - Used as an external real-world dataset for zero-shot evaluation,
     fine-tuning, and replay.
   - Stored at:
        data/raw/plantdoc/

3. PlantWild_v2
   - Downloaded from Hugging Face Hub.
   - Used for expanded 132-class real-world training.
   - Stored at:
        data/raw/plantwild_v2/

4. FieldPlant
   - Downloaded from Kaggle.
   - Original format is YOLO/object-detection-style.
   - Used only as a final compatible external evaluation dataset after
     conversion by data/prepare_fieldplant.py.
   - Stored at:
        data/raw/fieldplant/

Example usage:

    python data/download.py --dataset plantvillage
    python data/download.py --dataset plantvillage --force

    python data/download.py --dataset plantdoc
    python data/download.py --dataset plantwild
    python data/download.py --dataset fieldplant
"""

from __future__ import annotations

import argparse
import random
import shutil
import subprocess
import urllib.request
import zipfile
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]

RAW_DIR = PROJECT_ROOT / "data" / "raw"
TRAIN_DIR = PROJECT_ROOT / "data" / "train"
VAL_DIR = PROJECT_ROOT / "data" / "val"
TEST_DIR = PROJECT_ROOT / "data" / "test"

PLANTVILLAGE_KAGGLE_DATASET = "mohitsingh1804/plantvillage"

PLANTDOC_DIR = RAW_DIR / "plantdoc"
PLANTDOC_ZIP_PATH = RAW_DIR / "plantdoc.zip"
PLANTDOC_TEMP_DIR = RAW_DIR / "plantdoc_temp"
PLANTDOC_URL = (
    "https://github.com/pratikkayal/PlantDoc-Dataset/archive/refs/heads/master.zip"
)

PLANTWILD_DIR = RAW_DIR / "plantwild_v2"
PLANTWILD_HF_REPO = "uqtwei2/PlantWild"

FIELDPLANT_DIR = RAW_DIR / "fieldplant"
FIELDPLANT_KAGGLE_DATASET = "manhhoangvan/fieldplant"

TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

SEED = 42

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
METADATA_EXTENSIONS = {".csv", ".json", ".txt", ".xml", ".yaml", ".yml"}


def run_command(command: list[str]) -> None:
    """
    Run an external command and raise a clear error if it fails.

    Args:
        command:
            Command as a list of tokens, e.g. ["kaggle", "datasets", "download"].
    """
    print("Running:", " ".join(command))

    result = subprocess.run(
        command,
        text=True,
    )

    if result.returncode != 0:
        raise RuntimeError(
            "Command failed with exit code "
            f"{result.returncode}: {' '.join(command)}"
        )


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


def is_metadata_file(path: Path) -> bool:
    """
    Check whether a path looks like a metadata or annotation file.

    Args:
        path:
            File path.

    Returns:
        True if suffix is one of the known metadata/annotation extensions.
    """
    return path.is_file() and path.suffix.lower() in METADATA_EXTENSIONS


def download_plantvillage_zip() -> None:
    """
    Download PlantVillage from Kaggle into data/raw/.
    """
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    command = [
        "kaggle",
        "datasets",
        "download",
        "-d",
        PLANTVILLAGE_KAGGLE_DATASET,
        "-p",
        str(RAW_DIR),
    ]

    run_command(command)


def unzip_plantvillage_zips() -> None:
    """
    Unzip PlantVillage dataset zip files inside data/raw/.

    PlantDoc has its own download/extract flow, so plantdoc.zip is skipped here.
    """
    zip_files = sorted(RAW_DIR.glob("*.zip"))

    if not zip_files:
        print("No zip files found in data/raw. Dataset may already be unzipped.")
        return

    for zip_path in zip_files:
        if zip_path.name == PLANTDOC_ZIP_PATH.name:
            print(f"Skipping PlantDoc zip during PlantVillage unzip: {zip_path.name}")
            continue

        print(f"Unzipping {zip_path.name}...")

        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(RAW_DIR)

        print("Unzipped successfully.")


def get_plantvillage_dataset_root() -> Path:
    """
    Return the PlantVillage dataset root after extraction.

    Some Kaggle archives extract directly into data/raw/, while others create a
    nested PlantVillage/ folder. This helper supports both layouts.

    Returns:
        Path to the extracted PlantVillage root.
    """
    nested_dir = RAW_DIR / "PlantVillage"

    if nested_dir.exists():
        return nested_dir

    return RAW_DIR


def collect_images_by_class(dataset_root: Path) -> dict[str, list[Path]]:
    """
    Collect image paths grouped by class folder name.

    Supports both layouts:
        dataset_root/class_name/image.jpg

    and:
        dataset_root/train/class_name/image.jpg
        dataset_root/val/class_name/image.jpg
        dataset_root/test/class_name/image.jpg

    Args:
        dataset_root:
            Root folder of the extracted PlantVillage dataset.

    Returns:
        Dictionary mapping class name to image paths.
    """
    images_by_class = {}

    possible_split_dirs = [
        dataset_root / "train",
        dataset_root / "val",
        dataset_root / "valid",
        dataset_root / "test",
    ]

    existing_split_dirs = [
        split_dir
        for split_dir in possible_split_dirs
        if split_dir.exists() and split_dir.is_dir()
    ]

    search_roots = existing_split_dirs if existing_split_dirs else [dataset_root]

    for search_root in search_roots:
        for class_dir in sorted(search_root.iterdir()):
            if not class_dir.is_dir():
                continue

            image_paths = [
                image_path
                for image_path in sorted(class_dir.iterdir())
                if is_image_file(image_path)
            ]

            if not image_paths:
                continue

            images_by_class.setdefault(class_dir.name, []).extend(image_paths)

    return images_by_class


def inspect_plantvillage_dataset(dataset_root: Path) -> None:
    """
    Print PlantVillage class count and total image count.

    Args:
        dataset_root:
            Extracted PlantVillage root.
    """
    images_by_class = collect_images_by_class(dataset_root)

    print(f"\nDataset root: {dataset_root}")
    print(f"Number of classes: {len(images_by_class)}")

    total_images = sum(len(paths) for paths in images_by_class.values())

    print(f"Total images: {total_images}")


def clear_existing_plantvillage_splits() -> None:
    """
    Remove existing PlantVillage train/val/test split folders.
    """
    for split_dir in [TRAIN_DIR, VAL_DIR, TEST_DIR]:
        if split_dir.exists():
            print(f"Removing existing split folder: {split_dir}")
            shutil.rmtree(split_dir)


def create_plantvillage_split(
    dataset_root: Path,
    force: bool = False,
) -> None:
    """
    Create deterministic PlantVillage train/val/test splits.

    Args:
        dataset_root:
            Extracted PlantVillage root.
        force:
            If True, remove existing train/val/test folders before rebuilding.
    """
    ratio_sum = TRAIN_RATIO + VAL_RATIO + TEST_RATIO

    if abs(ratio_sum - 1.0) > 1e-6:
        raise ValueError(
            "TRAIN_RATIO + VAL_RATIO + TEST_RATIO must sum to 1. "
            f"Got: {ratio_sum}"
        )

    random.seed(SEED)

    if force:
        clear_existing_plantvillage_splits()

    images_by_class = collect_images_by_class(dataset_root)

    if not images_by_class:
        raise RuntimeError(f"No class images found under: {dataset_root}")

    for split_dir in [TRAIN_DIR, VAL_DIR, TEST_DIR]:
        split_dir.mkdir(parents=True, exist_ok=True)

    print("\nCreating PlantVillage train/val/test split...")

    for class_name in sorted(images_by_class):
        images = list(images_by_class[class_name])
        random.shuffle(images)

        total = len(images)

        train_count = int(total * TRAIN_RATIO)
        val_count = int(total * VAL_RATIO)

        train_images = images[:train_count]
        val_images = images[train_count : train_count + val_count]
        test_images = images[train_count + val_count :]

        splits = {
            TRAIN_DIR: train_images,
            VAL_DIR: val_images,
            TEST_DIR: test_images,
        }

        for split_dir, split_images in splits.items():
            target_class_dir = split_dir / class_name
            target_class_dir.mkdir(parents=True, exist_ok=True)

            for image_path in split_images:
                target_path = target_class_dir / image_path.name

                if not target_path.exists():
                    shutil.copy2(image_path, target_path)

        print(
            f"{class_name}: "
            f"train={len(train_images)}, "
            f"val={len(val_images)}, "
            f"test={len(test_images)}"
        )


def prepare_plantvillage(force: bool = False) -> None:
    """
    Download, unzip, inspect, and split PlantVillage.

    Args:
        force:
            If True, remove existing generated train/val/test splits before
            rebuilding them.
    """
    download_plantvillage_zip()
    unzip_plantvillage_zips()

    dataset_root = get_plantvillage_dataset_root()

    inspect_plantvillage_dataset(dataset_root)
    create_plantvillage_split(dataset_root, force=force)


def download_plantdoc(force: bool = False) -> None:
    """
    Download PlantDoc from GitHub.

    PlantDoc is used as an external real-world dataset. It is not mixed into the
    original PlantVillage split folders.

    Args:
        force:
            If True, remove existing PlantDoc data and re-download.
    """
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    if PLANTDOC_DIR.exists() and not force:
        print(f"PlantDoc already exists at: {PLANTDOC_DIR}")
        print("Use --force to remove and re-download it.")
        return

    if force and PLANTDOC_DIR.exists():
        print(f"Removing existing PlantDoc folder: {PLANTDOC_DIR}")
        shutil.rmtree(PLANTDOC_DIR)

    if PLANTDOC_TEMP_DIR.exists():
        print(f"Removing temporary PlantDoc folder: {PLANTDOC_TEMP_DIR}")
        shutil.rmtree(PLANTDOC_TEMP_DIR)

    if PLANTDOC_ZIP_PATH.exists():
        print(f"Removing existing PlantDoc zip: {PLANTDOC_ZIP_PATH}")
        PLANTDOC_ZIP_PATH.unlink()

    print("\nDownloading PlantDoc dataset...")
    print(f"Source:      {PLANTDOC_URL}")
    print(f"Destination: {PLANTDOC_ZIP_PATH}")

    urllib.request.urlretrieve(PLANTDOC_URL, PLANTDOC_ZIP_PATH)

    print("\nExtracting PlantDoc dataset...")

    PLANTDOC_TEMP_DIR.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(PLANTDOC_ZIP_PATH, "r") as zip_ref:
        zip_ref.extractall(PLANTDOC_TEMP_DIR)

    extracted_roots = [
        path
        for path in PLANTDOC_TEMP_DIR.iterdir()
        if path.is_dir()
    ]

    if len(extracted_roots) != 1:
        raise RuntimeError(
            f"Expected one extracted PlantDoc root folder, "
            f"found {len(extracted_roots)}."
        )

    extracted_root = extracted_roots[0]

    shutil.move(str(extracted_root), str(PLANTDOC_DIR))
    shutil.rmtree(PLANTDOC_TEMP_DIR)

    print("\nPlantDoc download complete.")
    print(f"Saved to: {PLANTDOC_DIR}")

    print("\nTop-level PlantDoc contents:")
    for item in sorted(PLANTDOC_DIR.iterdir()):
        print(f"- {item.name}")


def inspect_plantdoc() -> None:
    """
    Inspect PlantDoc folder structure and class-folder image counts.
    """
    if not PLANTDOC_DIR.exists():
        print(f"PlantDoc folder does not exist: {PLANTDOC_DIR}")
        print("Run: python data/download.py --dataset plantdoc")
        return

    print(f"\nPlantDoc root: {PLANTDOC_DIR}")

    print("\nTop-level contents:")
    for item in sorted(PLANTDOC_DIR.iterdir()):
        print(f"- {item.name}")

    possible_class_roots = [
        PLANTDOC_DIR / "train",
        PLANTDOC_DIR / "test",
        PLANTDOC_DIR / "Train",
        PLANTDOC_DIR / "Test",
    ]

    for root in possible_class_roots:
        if not root.exists():
            continue

        print(f"\nClass folders inside: {root}")

        class_dirs = [
            path
            for path in sorted(root.iterdir())
            if path.is_dir()
        ]

        for class_dir in class_dirs:
            image_count = len(
                [
                    path
                    for path in class_dir.iterdir()
                    if is_image_file(path)
                ]
            )

            print(f"- {class_dir.name}: {image_count}")


def download_plantwild(force: bool = False) -> None:
    """
    Download PlantWild_v2 using Hugging Face Hub.

    PlantWild_v2 is used for expanded 132-class training.

    Args:
        force:
            If True, remove existing PlantWild_v2 data and re-download.
    """
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    if PLANTWILD_DIR.exists() and not force:
        print(f"PlantWild_v2 already exists at: {PLANTWILD_DIR}")
        print("Use --force to remove and re-download it.")
        return

    if force and PLANTWILD_DIR.exists():
        print(f"Removing existing PlantWild_v2 folder: {PLANTWILD_DIR}")
        shutil.rmtree(PLANTWILD_DIR)

    print("\nDownloading PlantWild_v2 dataset...")
    print(f"Source Hugging Face repo: {PLANTWILD_HF_REPO}")
    print(f"Destination:              {PLANTWILD_DIR}")

    try:
        from huggingface_hub import snapshot_download
    except ImportError as error:
        raise ImportError(
            "huggingface_hub is required to download PlantWild_v2.\n"
            "Install it with:\n"
            "pip install huggingface_hub"
        ) from error

    snapshot_download(
        repo_id=PLANTWILD_HF_REPO,
        repo_type="dataset",
        local_dir=PLANTWILD_DIR,
    )

    print("\nPlantWild_v2 download complete.")
    print(f"Saved to: {PLANTWILD_DIR}")


def inspect_plantwild() -> None:
    """
    Inspect PlantWild_v2 folder structure.

    Prints:
        - top-level files/folders
        - total image count
        - possible CSV/JSON/TXT metadata files
        - immediate subfolder image counts
    """
    if not PLANTWILD_DIR.exists():
        print(f"PlantWild_v2 folder does not exist: {PLANTWILD_DIR}")
        print("Run: python data/download.py --dataset plantwild")
        return

    print(f"\nPlantWild_v2 root: {PLANTWILD_DIR}")

    print("\nTop-level contents:")
    for item in sorted(PLANTWILD_DIR.iterdir()):
        print(f"- {item.name}")

    print("\nSearching for image files...")
    image_paths = [
        path
        for path in PLANTWILD_DIR.rglob("*")
        if is_image_file(path)
    ]

    print(f"Total image files found: {len(image_paths)}")

    print("\nPossible CSV/JSON/TXT metadata files:")
    metadata_files = [
        path
        for path in PLANTWILD_DIR.rglob("*")
        if path.is_file() and path.suffix.lower() in {".csv", ".json", ".txt"}
    ]

    for path in metadata_files[:50]:
        print(f"- {path.relative_to(PLANTWILD_DIR)}")

    if len(metadata_files) > 50:
        print(f"... and {len(metadata_files) - 50} more metadata files")

    print("\nImmediate subfolders:")
    subfolders = [
        path
        for path in PLANTWILD_DIR.iterdir()
        if path.is_dir()
    ]

    for folder in sorted(subfolders):
        folder_image_count = len(
            [
                path
                for path in folder.rglob("*")
                if is_image_file(path)
            ]
        )

        print(f"- {folder.name}: {folder_image_count} images")


def download_fieldplant(force: bool = False) -> None:
    """
    Download FieldPlant from Kaggle.

    FieldPlant is used only as a final external evaluation dataset and should
    not be mixed into training.

    Args:
        force:
            If True, remove existing FieldPlant data and re-download.
    """
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    if FIELDPLANT_DIR.exists() and not force:
        print(f"FieldPlant already exists at: {FIELDPLANT_DIR}")
        print("Use --force to remove and re-download it.")
        return

    if force and FIELDPLANT_DIR.exists():
        print(f"Removing existing FieldPlant folder: {FIELDPLANT_DIR}")
        shutil.rmtree(FIELDPLANT_DIR)

    FIELDPLANT_DIR.mkdir(parents=True, exist_ok=True)

    print("\nDownloading FieldPlant dataset...")
    print(f"Source Kaggle dataset: {FIELDPLANT_KAGGLE_DATASET}")
    print(f"Destination:           {FIELDPLANT_DIR}")

    command = [
        "kaggle",
        "datasets",
        "download",
        "-d",
        FIELDPLANT_KAGGLE_DATASET,
        "-p",
        str(FIELDPLANT_DIR),
        "--unzip",
    ]

    run_command(command)

    print("\nFieldPlant download complete.")
    print(f"Saved to: {FIELDPLANT_DIR}")


def inspect_fieldplant() -> None:
    """
    Inspect FieldPlant folder structure.

    Prints:
        - top-level files/folders
        - total image count
        - possible annotation/metadata files
        - immediate subfolder image and annotation counts
    """
    if not FIELDPLANT_DIR.exists():
        print(f"FieldPlant folder does not exist: {FIELDPLANT_DIR}")
        print("Run: python data/download.py --dataset fieldplant")
        return

    print(f"\nFieldPlant root: {FIELDPLANT_DIR}")

    print("\nTop-level contents:")
    for item in sorted(FIELDPLANT_DIR.iterdir()):
        print(f"- {item.name}")

    print("\nSearching for image files...")
    image_paths = [
        path
        for path in FIELDPLANT_DIR.rglob("*")
        if is_image_file(path)
    ]

    print(f"Total image files found: {len(image_paths)}")

    print("\nPossible annotation/metadata files:")
    metadata_files = [
        path
        for path in FIELDPLANT_DIR.rglob("*")
        if is_metadata_file(path)
    ]

    for path in metadata_files[:80]:
        print(f"- {path.relative_to(FIELDPLANT_DIR)}")

    if len(metadata_files) > 80:
        print(f"... and {len(metadata_files) - 80} more metadata files")

    print("\nImmediate subfolders:")
    subfolders = [
        path
        for path in FIELDPLANT_DIR.iterdir()
        if path.is_dir()
    ]

    for folder in sorted(subfolders):
        folder_image_count = len(
            [
                path
                for path in folder.rglob("*")
                if is_image_file(path)
            ]
        )

        metadata_count = len(
            [
                path
                for path in folder.rglob("*")
                if is_metadata_file(path)
            ]
        )

        print(
            f"- {folder.name}: "
            f"{folder_image_count} images, "
            f"{metadata_count} metadata/annotation files"
        )


def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments.

    Returns:
        Parsed argparse namespace.
    """
    parser = argparse.ArgumentParser(
        description="Download and prepare PlantGuard datasets."
    )

    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        choices=["plantvillage", "plantdoc", "plantwild", "fieldplant"],
        help="Which dataset to download or prepare.",
    )

    parser.add_argument(
        "--force",
        action="store_true",
        help="Remove existing generated data for the selected dataset before rebuilding.",
    )

    return parser.parse_args()


def main() -> None:
    """
    Run the selected dataset download/preparation workflow.
    """
    args = parse_args()

    if args.dataset == "plantvillage":
        prepare_plantvillage(force=args.force)

    elif args.dataset == "plantdoc":
        download_plantdoc(force=args.force)
        inspect_plantdoc()

    elif args.dataset == "plantwild":
        download_plantwild(force=args.force)
        inspect_plantwild()

    elif args.dataset == "fieldplant":
        download_fieldplant(force=args.force)
        inspect_fieldplant()


if __name__ == "__main__":
    main()