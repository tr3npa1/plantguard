from pathlib import Path
import subprocess
import zipfile
import random
import shutil

DATASET = "mohitsingh1804/plantvillage"
PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
TRAIN_DIR = PROJECT_ROOT/ "data" / "train"
VAL_DIR = PROJECT_ROOT/ "data" / "val"
TEST_DIR = PROJECT_ROOT/ "data" / "test"

TRAIN_RATIO = 0.7
VAL_RATIO = 0.15
TEST_RATIO = 0.15

SEED = 42

def run_command(command):
    print("Running:", " ".join(command))
    result = subprocess.run(
        command,
        capture_output = True,
        text = True,
    )
    if result.stdout:
        print(result.stdout)
    if result.stderr:
        print(result.stderr)
    if result.returncode != 0:
        raise RuntimeError("Command failed")
    
def download_datasets():
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    command = [
        "kaggle",
        "datasets",
        "download",
        "-d",
        DATASET,
        "-p",
        str(RAW_DIR)
    ]
    run_command(command)

def unzip_datasets():
    zip_files = list(RAW_DIR.glob("*.zip"))
    if not zip_files:
        print("No zip file found. Maybe already unzipped.")
        return
    for zip_path in zip_files:
        print(f'Unzipping {zip_path.name}...')
        with zipfile.ZipFile(zip_path,"r") as zip_ref:
            zip_ref.extractall(RAW_DIR)
        print("Unzipped successfully. ")

def get_dataset_root():
    nested_dir = RAW_DIR / "PlantVillage"

    if nested_dir.exists():
        return nested_dir

    return RAW_DIR

def collect_images_by_class(dataset_root):
    image_extension = {".jpg", ".jpeg", ".png", ".bmp"}
    images_by_class = {}
    possible_split_dirs = [
        dataset_root / "train",
        dataset_root / "val",
        dataset_root / "valid",
        dataset_root / "test",
    ]
    existing_split_dirs = [
        split_dir for split_dir in possible_split_dirs
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
            image_paths=[
                image_path 
                for image_path in class_dir.iterdir()
                if image_path.is_file()
                and image_path.suffix.lower() in image_extension
            ]
            if not image_paths:
                continue
            if class_name not in images_by_class:
                images_by_class[class_name]=[]
            images_by_class[class_name].extend(image_paths)
    return images_by_class

def inspect_dataset(dataset_root):
    images_by_class = collect_images_by_class(dataset_root)
    print(f"\nDataset root: {dataset_root}")
    print(f"Number of classes: {len(images_by_class)}")
    total_images = 0
    for class_name in sorted(images_by_class):
        image_count = len(images_by_class[class_name])
        total_images+=image_count
    print(f'\nTotal images: {total_images}')

def create_split(dataset_root):
    random.seed(SEED)
    images_by_class = collect_images_by_class(dataset_root)
    split_dirs = [TRAIN_DIR, VAL_DIR, TEST_DIR]
    for split_dir in split_dirs:
        split_dir.mkdir(parents=True, exist_ok=True)
    print("\nCreating train/val/test split...")
    for class_name in sorted(images_by_class):
        images = images_by_class[class_name]
        random.shuffle(images)
        total = len(images)
        train_count = int(total*TRAIN_RATIO)
        val_count = int(total * VAL_RATIO)
        train_images = images[:train_count]
        val_images = images[train_count:train_count + val_count]
        test_images = images[train_count + val_count:]
        splits = {
            TRAIN_DIR: train_images,
            VAL_DIR: val_images,
            TEST_DIR: test_images
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

if __name__ == "__main__":
    download_datasets()
    unzip_datasets()
    dataset_root = get_dataset_root()
    inspect_dataset(dataset_root)
    create_split(dataset_root)
