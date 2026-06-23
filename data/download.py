from pathlib import Path
import subprocess
import zipfile
import random
import shutil
import argparse
import urllib.request


DATASET = "mohitsingh1804/plantvillage"

PROJECT_ROOT = Path(__file__).resolve().parents[1]

RAW_DIR = PROJECT_ROOT / "data" / "raw"
TRAIN_DIR = PROJECT_ROOT / "data" / "train"
VAL_DIR = PROJECT_ROOT / "data" / "val"
TEST_DIR = PROJECT_ROOT / "data" / "test"

PLANTDOC_DIR = RAW_DIR / "plantdoc"
PLANTDOC_ZIP_PATH = RAW_DIR / "plantdoc.zip"
PLANTDOC_TEMP_DIR = RAW_DIR / "plantdoc_temp"
PLANTDOC_URL = "https://github.com/pratikkayal/PlantDoc-Dataset/archive/refs/heads/master.zip"
PLANTWILD_DIR = RAW_DIR / "plantwild_v2"
PLANTWILD_HF_REPO = "uqtwei2/PlantWild"
FIELDPLANT_DIR = RAW_DIR / "fieldplant"
FIELDPLANT_KAGGLE_DATASET = "manhhoangvan/fieldplant"


TRAIN_RATIO = 0.7
VAL_RATIO = 0.15
TEST_RATIO = 0.15

SEED = 42


def run_command(command):
    """
    Run a shell command and print its output.

    Args:
        command: List of command parts.
    """
    print("Running:", " ".join(command))

    result = subprocess.run(
        command,
        text=True,
    )

    if result.returncode != 0:
        raise RuntimeError("Command failed")


def download_datasets():
    """
    Download the PlantVillage dataset using the Kaggle API.
    """
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    command = [
        "kaggle",
        "datasets",
        "download",
        "-d",
        DATASET,
        "-p",
        str(RAW_DIR),
    ]

    run_command(command)


def unzip_datasets():
    """
    Unzip PlantVillage dataset zip files inside data/raw.

    PlantDoc zip is skipped here because PlantDoc has its own download/extract
    flow.
    """
    zip_files = list(RAW_DIR.glob("*.zip"))

    if not zip_files:
        print("No zip file found. Maybe already unzipped.")
        return

    for zip_path in zip_files:
        if zip_path.name == PLANTDOC_ZIP_PATH.name:
            print(f"Skipping PlantDoc zip in PlantVillage unzip step: {zip_path.name}")
            continue

        print(f"Unzipping {zip_path.name}...")

        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(RAW_DIR)

        print("Unzipped successfully.")


def get_dataset_root():
    """
    Return the PlantVillage dataset root after extraction.
    """
    nested_dir = RAW_DIR / "PlantVillage"

    if nested_dir.exists():
        return nested_dir

    return RAW_DIR


def collect_images_by_class(dataset_root):
    """
    Collect image paths grouped by class folder name.

    This supports datasets that either have class folders directly under the
    dataset root or inside train/val/test folders.

    Args:
        dataset_root: Root folder of dataset.

    Returns:
        Dictionary:
            class_name -> list of image paths
    """
    image_extensions = {".jpg", ".jpeg", ".png", ".bmp"}

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

    if existing_split_dirs:
        search_roots = existing_split_dirs
    else:
        search_roots = [dataset_root]

    for search_root in search_roots:
        for class_dir in search_root.iterdir():
            if not class_dir.is_dir():
                continue

            class_name = class_dir.name

            image_paths = [
                image_path
                for image_path in class_dir.iterdir()
                if image_path.is_file()
                and image_path.suffix.lower() in image_extensions
            ]

            if not image_paths:
                continue

            if class_name not in images_by_class:
                images_by_class[class_name] = []

            images_by_class[class_name].extend(image_paths)

    return images_by_class


def inspect_dataset(dataset_root):
    """
    Print dataset class count and total image count.
    """
    images_by_class = collect_images_by_class(dataset_root)

    print(f"\nDataset root: {dataset_root}")
    print(f"Number of classes: {len(images_by_class)}")

    total_images = 0

    for class_name in sorted(images_by_class):
        image_count = len(images_by_class[class_name])
        total_images += image_count

    print(f"\nTotal images: {total_images}")


def clear_existing_splits():
    """
    Remove existing PlantVillage train/val/test split folders.
    """
    for split_dir in [TRAIN_DIR, VAL_DIR, TEST_DIR]:
        if split_dir.exists():
            print(f"Removing existing split folder: {split_dir}")
            shutil.rmtree(split_dir)


def create_split(dataset_root, force=False):
    """
    Create PlantVillage train/val/test splits.

    Args:
        dataset_root: Root folder of PlantVillage dataset.
        force: If True, remove existing train/val/test folders first.
    """
    random.seed(SEED)

    if force:
        clear_existing_splits()

    images_by_class = collect_images_by_class(dataset_root)

    split_dirs = [TRAIN_DIR, VAL_DIR, TEST_DIR]

    for split_dir in split_dirs:
        split_dir.mkdir(parents=True, exist_ok=True)

    print("\nCreating train/val/test split...")

    for class_name in sorted(images_by_class):
        images = images_by_class[class_name]
        random.shuffle(images)

        total = len(images)

        train_count = int(total * TRAIN_RATIO)
        val_count = int(total * VAL_RATIO)

        train_images = images[:train_count]
        val_images = images[train_count:train_count + val_count]
        test_images = images[train_count + val_count:]

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


def download_plantdoc(force=False):
    """
    Download the PlantDoc dataset from the official GitHub repository.

    PlantDoc is used only as an external real-world evaluation dataset.
    It should not be mixed into PlantVillage train/val/test splits.

    The extracted dataset is stored at:
        data/raw/plantdoc/
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
        print(f"Removing existing PlantDoc zip before re-downloading: {PLANTDOC_ZIP_PATH}")
        PLANTDOC_ZIP_PATH.unlink()

    print("\nDownloading PlantDoc dataset...")
    print(f"Source: {PLANTDOC_URL}")
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
            f"Expected one extracted root folder, found {len(extracted_roots)}."
        )
    
    extracted_root = extracted_roots[0]

    shutil.move(str(extracted_root), str(PLANTDOC_DIR))

    shutil.rmtree(PLANTDOC_TEMP_DIR)

    print("\nPlantDoc download complete.")
    print(f"Saved to: {PLANTDOC_DIR}")

    print("\nTop-level PlantDoc contents:")
    for item in sorted(PLANTDOC_DIR.iterdir()):
        print(f"- {item.name}")


def inspect_plantdoc():
    """
    Inspect PlantDoc folder structure and print class folders if present.
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
            for path in root.iterdir()
            if path.is_dir()
        ]

        for class_dir in class_dirs:
            image_count = len(
                [
                    path 
                    for path in class_dir.iterdir()
                    if path.is_file()
                    and path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}
                ]
            )
            print(f"- {class_dir.name}: {image_count}")


def prepare_plantvillage(force=False):
    """
    Download, unzip, inspect, and split PlantVillage.
    """
    download_datasets()
    unzip_datasets()

    dataset_root = get_dataset_root()

    inspect_dataset(dataset_root)
    create_split(dataset_root, force=force)


def download_plantwild(force=False):
    """
    Download PlantWild_v2 using Hugging Face Hub.

    PlantWild_v2 is used for Stage C expanded-label training.

    The dataset is saved to:
        data/raw/plantwild_v2/

    Notes:
        This downloads the Hugging Face dataset repository. After download,
        we still inspect the folder structure to locate the actual PlantWild_v2
        image folders and labels.
    """
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    if PLANTWILD_DIR.exists() and not force:
        print(f"PlantWild_v2 already exists at {PLANTWILD_DIR}")
        print("Use --force to remove and re-download it.")
        return
    
    if force and PLANTWILD_DIR.exists():
        print(f"Removing existing PlantWild_v2 folder: {PLANTWILD_DIR}")
        shutil.rmtree(PLANTWILD_DIR)

    print("\nDownloading PlantWild_v2 dataset...")
    print(f"Source Hugging Face repo: {PLANTWILD_HF_REPO}")
    print(f"Destination: {PLANTWILD_DIR}")

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

def inspect_plantwild():
    """
    Inspect PlantWild_v2 folder structure.

    This prints:
        -top-level files/folders
        -possible class folders
        -rough image counts

    We use this before writing the dataloader because PlantWild_v2
    may be folder-based, CSV-based, or packed inside nested folders.
    """
    if not PLANTWILD_DIR.exists():
        print(f"PlantWild_v2 folder does not exist: {PLANTWILD_DIR}")
        print("Run: python data/download.py --dataset plantwild")
        return

    image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

    print(f"\nPlantWild_v2 root: {PLANTWILD_DIR}")

    print("\nTop-level contents:")
    for item in sorted(PLANTWILD_DIR.iterdir()):
        print(f"- {item.name}")

    print("\nSearching for image files...")
    image_paths = [
        path
        for path in PLANTWILD_DIR.rglob("*")
        if path.is_file() and path.suffix.lower() in image_extensions
    ]

    print(f"Total image files found: {len(image_paths)}")

    print("\nPossible CSV/metadata files:")
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
                if path.is_file() and path.suffix.lower() in image_extensions
            ]
        )
        print(f"- {folder.name}: {folder_image_count} images")


def download_fieldplant(force=False):
    """
    Download the FieldPlant external dataset using the Kaggle API.

    FieldPlant is used only as a final external evaluation dataset.
    It should not be mixed into PlantVillage, PlantDoc, or PlantWild training.

    The downloaded/extracted dataset is stored at:
        data/raw/fieldplant/
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

    print("\nDownloading FieldPlant dataset.")
    print(f"Source Kaggle dataset: {FIELDPLANT_KAGGLE_DATASET}")
    print(f"Destination: {FIELDPLANT_DIR}")

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


def inspect_fieldplant():
    """
    Inspect FieldPlant folder structure.

    This prints:
        - top-level files/folders
        - total image count
        - possible metadata/annotation files
        - immediate subfolder image counts

    We inspect first because FieldPlant may be classification-style,
    detection-style, or annotation-file based.
    """
    if not FIELDPLANT_DIR.exists():
        print(f"FieldPlant folder does not exist: {FIELDPLANT_DIR}")
        print("Run: python data/download.py --dataset fieldplant")
        return
    
    image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    
    print(f"\nFieldPlant root: {FIELDPLANT_DIR}")

    print("\nTop-level contents:")
    for item in sorted(FIELDPLANT_DIR.iterdir()):
        print(f"- {item.name}")

    print("\nSearching for image files...")
    image_paths = [
        path
        for path in FIELDPLANT_DIR.rglob("*")
        if path.is_file() and path.suffix.lower() in image_extensions
    ]

    print(f"Total image files found: {len(image_paths)}")

    print("\nPossible annotation/metadata files:")
    metadata_files = [
        path
        for path in FIELDPLANT_DIR.rglob("*")
        if path.is_file()
        and path.suffix.lower() in {".csv", ".json", ".txt", ".xml", ".yaml", ".yml"}
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
                if path.is_file() and path.suffix.lower() in image_extensions
            ]
        )
        metadata_count = len(
            [
                path
                for path in folder.rglob("*")
                if path.is_file()
                and path.suffix.lower() in {".csv", ".json", ".txt", ".xml", ".yaml", ".yml"}
            ]
        )

        print(
            f"- {folder.name}: "
            f"{folder_image_count} images, "
            f"{metadata_count} metadata/annotation files"
        )
    

def main():
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

    args = parser.parse_args()

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