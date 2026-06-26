"""
Dataset and DataLoader utilities for PlantGuard.

This module centralizes all dataset logic used across the PlantGuard pipeline:

1. PlantVillage
   - Original 38-class class-folder dataset.
   - Used for base training, validation, testing, and replay.

2. PlantDoc
   - External real-world dataset with folder names that differ from PlantVillage.
   - Mapped into the PlantGuard label space for external evaluation,
     fine-tuning, and replay.

3. PlantWild_v2
   - Expanded real-world dataset prepared into CSV split files.
   - Introduces the 132-class expanded PlantGuard label space.

4. FieldPlant
   - External YOLO-style dataset converted into a compatible image-level
     classification CSV for final external testing.

The most important invariant in this file is label order:

    indices 0-37   -> original PlantVillage / PlantGuard labels
    indices 38-131 -> new PlantWild_v2 labels

Keeping this invariant allows older 38-class checkpoints to be expanded safely
and allows external datasets to be evaluated consistently.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import (
    ConcatDataset,
    DataLoader,
    Dataset,
    Subset,
    WeightedRandomSampler,
)
from torchvision import transforms


PROJECT_ROOT = Path(__file__).resolve().parents[1]

TRAIN_DIR = PROJECT_ROOT / "data" / "train"
VAL_DIR = PROJECT_ROOT / "data" / "val"
TEST_DIR = PROJECT_ROOT / "data" / "test"

PLANTDOC_DIR = PROJECT_ROOT / "data" / "raw" / "plantdoc"
PLANTWILD_DIR = PROJECT_ROOT / "data" / "raw" / "plantwild_v2"

METADATA_DIR = PROJECT_ROOT / "data" / "metadata"
SPLITS_DIR = PROJECT_ROOT / "data" / "splits"

EXPANDED_CLASS_NAMES_JSON = METADATA_DIR / "plantguard_expanded_class_names.json"

PLANTWILD_TRAIN_CSV = SPLITS_DIR / "plantwild_train.csv"
PLANTWILD_VAL_CSV = SPLITS_DIR / "plantwild_val.csv"
PLANTWILD_TEST_CSV = SPLITS_DIR / "plantwild_test.csv"

FIELDPLANT_TEST_CSV = SPLITS_DIR / "fieldplant_test.csv"

IMAGE_SIZE = 256
BATCH_SIZE = 32
NUM_WORKERS = 2
EXPECTED_EXPANDED_NUM_CLASSES = 132
SEED = 42

# Dataset-specific normalization values calculated from the PlantVillage
# training split only. These are used consistently for train/val/test,
# external evaluation, GradCAM, and later ONNX/API preprocessing.
DATASET_MEAN = [0.4665524959564209, 0.48931649327278137, 0.4102632701396942]
DATASET_STD = [0.19912923872470856, 0.174818754196167, 0.21729066967964172]

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

PLANTDOC_TO_PLANTVILLAGE = {
    "Apple leaf": "Apple___healthy",
    "Apple rust leaf": "Apple___Cedar_apple_rust",
    "Apple Scab Leaf": "Apple___Apple_scab",
    "Bell_pepper leaf": "Pepper,_bell___healthy",
    "Bell_pepper leaf spot": "Pepper,_bell___Bacterial_spot",
    "Blueberry leaf": "Blueberry___healthy",
    "Cherry leaf": "Cherry_(including_sour)___healthy",
    "Corn Gray leaf spot": "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot",
    "Corn leaf blight": "Corn_(maize)___Northern_Leaf_Blight",
    "Corn rust leaf": "Corn_(maize)___Common_rust_",
    "grape leaf": "Grape___healthy",
    "grape leaf black rot": "Grape___Black_rot",
    "Peach leaf": "Peach___healthy",
    "Potato leaf early blight": "Potato___Early_blight",
    "Potato leaf late blight": "Potato___Late_blight",
    "Raspberry leaf": "Raspberry___healthy",
    "Soyabean leaf": "Soybean___healthy",
    "Squash Powdery mildew leaf": "Squash___Powdery_mildew",
    "Strawberry leaf": "Strawberry___healthy",
    "Tomato Early blight leaf": "Tomato___Early_blight",
    "Tomato leaf": "Tomato___healthy",
    "Tomato leaf bacterial spot": "Tomato___Bacterial_spot",
    "Tomato leaf late blight": "Tomato___Late_blight",
    "Tomato leaf mosaic virus": "Tomato___Tomato_mosaic_virus",
    "Tomato leaf yellow virus": "Tomato___Tomato_Yellow_Leaf_Curl_Virus",
    "Tomato mold leaf": "Tomato___Leaf_Mold",
    "Tomato Septoria leaf spot": "Tomato___Septoria_leaf_spot",
    "Tomato two spotted spider mites leaf": (
        "Tomato___Spider_mites Two-spotted_spider_mite"
    ),
}


def is_image_file(path: Path) -> bool:
    """
    Check whether a path is a supported image file.

    Args:
        path:
            File path.

    Returns:
        True if the path is a file with a supported image extension.
    """
    return path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS


def resolve_project_path(path_value: str | Path) -> Path:
    """
    Resolve a path that may be absolute or project-relative.

    CSV files store image paths relative to PROJECT_ROOT. This helper also
    safely supports absolute paths if needed.

    Args:
        path_value:
            Absolute path or project-relative path string.

    Returns:
        Resolved Path object.
    """
    path = Path(str(path_value))

    if path.is_absolute():
        return path

    return PROJECT_ROOT / path


def get_transforms() -> tuple[transforms.Compose, transforms.Compose]:
    """
    Create training and evaluation transform pipelines.

    Training transforms include light augmentation to improve generalization.
    Evaluation transforms are deterministic so validation/test metrics are
    reproducible and comparable across runs.

    Returns:
        train_transforms:
            Augmented transform pipeline for training.
        eval_transforms:
            Deterministic transform pipeline for validation/testing.
    """
    train_transforms = transforms.Compose(
        [
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=15),
            transforms.ColorJitter(
                brightness=0.2,
                contrast=0.2,
                saturation=0.2,
                hue=0.05,
            ),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=DATASET_MEAN,
                std=DATASET_STD,
            ),
        ]
    )

    eval_transforms = transforms.Compose(
        [
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=DATASET_MEAN,
                std=DATASET_STD,
            ),
        ]
    )

    return train_transforms, eval_transforms


def select_transform(transform_type: str) -> transforms.Compose:
    """
    Select train or evaluation preprocessing.

    Args:
        transform_type:
            "train" for augmented transforms or "eval" for deterministic transforms.

    Returns:
        Selected torchvision transform pipeline.
    """
    train_transforms, eval_transforms = get_transforms()

    if transform_type == "train":
        return train_transforms

    if transform_type == "eval":
        return eval_transforms

    raise ValueError(
        f"Unsupported transform_type: {transform_type}. "
        "Choose 'train' or 'eval'."
    )


class PlantDiseaseDataset(Dataset):
    """
    Class-folder dataset for PlantVillage-style image classification.

    Expected directory layout:

        data/train/
            Apple___Apple_scab/
                image1.jpg
                image2.jpg
            Apple___healthy/
                image3.jpg

    Class names are sorted alphabetically to produce a stable label mapping.
    Each sample is stored as:

        (image_path, label_index)
    """

    def __init__(
        self,
        root_dir: str | Path,
        transform: transforms.Compose | None = None,
    ) -> None:
        """
        Scan class folders and collect image paths.

        Args:
            root_dir:
                Split folder such as data/train, data/val, or data/test.
            transform:
                Optional transform pipeline applied to each image.
        """
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.image_extensions = IMAGE_EXTENSIONS

        if not self.root_dir.exists():
            raise FileNotFoundError(f"Dataset directory not found: {self.root_dir}")

        self.class_names = sorted(
            folder.name
            for folder in self.root_dir.iterdir()
            if folder.is_dir()
        )

        if not self.class_names:
            raise RuntimeError(f"No class folders found in: {self.root_dir}")

        self.class_to_idx = {
            class_name: index
            for index, class_name in enumerate(self.class_names)
        }

        self.samples = []

        for class_name in self.class_names:
            class_dir = self.root_dir / class_name
            label = self.class_to_idx[class_name]

            for image_path in sorted(class_dir.iterdir()):
                if is_image_file(image_path):
                    self.samples.append((image_path, label))

        if not self.samples:
            raise RuntimeError(f"No image files found in: {self.root_dir}")

    def __len__(self) -> int:
        """
        Return the number of image samples.
        """
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        """
        Load one image-label pair.

        Args:
            index:
                Sample index.

        Returns:
            image:
                Transformed image tensor.
            label:
                Integer class label.
        """
        image_path, label = self.samples[index]

        image = Image.open(image_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, label


class PlantDocEvaluationDataset(Dataset):
    """
    PlantDoc dataset mapped into the PlantGuard label space.

    PlantDoc folder names differ from PlantVillage folder names. This dataset
    uses PLANTDOC_TO_PLANTVILLAGE to map compatible PlantDoc classes into the
    current model label order.

    Each sample is stored as:

        (image_path, label_index, plantdoc_class_name, mapped_plantguard_label)

    The extra metadata is useful for debugging and reporting, while __getitem__
    still returns only:

        (image, label)
    """

    def __init__(
        self,
        root_dir: str | Path,
        class_names: list[str],
        transform: transforms.Compose | None = None,
        splits: tuple[str, ...] = ("train", "test"),
    ) -> None:
        """
        Build a mapped PlantDoc dataset.

        Args:
            root_dir:
                PlantDoc root directory containing train/ and test/ folders.
            class_names:
                Target PlantGuard class names in model label-index order.
            transform:
                Optional image transform pipeline.
            splits:
                PlantDoc splits to load, e.g. ("train",), ("test",), or
                ("train", "test").
        """
        self.root_dir = Path(root_dir)
        self.class_names = list(class_names)
        self.transform = transform
        self.splits = tuple(splits)

        if not self.root_dir.exists():
            raise FileNotFoundError(f"PlantDoc directory not found: {self.root_dir}")

        self.class_to_idx = {
            class_name: index
            for index, class_name in enumerate(self.class_names)
        }

        self.samples = []
        self.skipped_classes = {}
        self.mapped_class_counts = {}

        for split_name in self.splits:
            split_dir = self.root_dir / split_name

            if not split_dir.exists():
                continue

            for plantdoc_class_dir in sorted(split_dir.iterdir()):
                if not plantdoc_class_dir.is_dir():
                    continue

                plantdoc_class_name = plantdoc_class_dir.name

                image_paths = sorted(
                    image_path
                    for image_path in plantdoc_class_dir.iterdir()
                    if is_image_file(image_path)
                )

                if plantdoc_class_name not in PLANTDOC_TO_PLANTVILLAGE:
                    self.skipped_classes[plantdoc_class_name] = (
                        self.skipped_classes.get(plantdoc_class_name, 0)
                        + len(image_paths)
                    )
                    continue

                mapped_class_name = PLANTDOC_TO_PLANTVILLAGE[plantdoc_class_name]

                if mapped_class_name not in self.class_to_idx:
                    self.skipped_classes[plantdoc_class_name] = (
                        self.skipped_classes.get(plantdoc_class_name, 0)
                        + len(image_paths)
                    )
                    continue

                label = self.class_to_idx[mapped_class_name]

                for image_path in image_paths:
                    self.samples.append(
                        (
                            image_path,
                            label,
                            plantdoc_class_name,
                            mapped_class_name,
                        )
                    )

                self.mapped_class_counts[mapped_class_name] = (
                    self.mapped_class_counts.get(mapped_class_name, 0)
                    + len(image_paths)
                )

        if not self.samples:
            raise RuntimeError(
                f"No mapped PlantDoc images found in {self.root_dir}. "
                "Check that PlantDoc is downloaded and that the label mapping "
                "matches the current class_names list."
            )

    def __len__(self) -> int:
        """
        Return the number of mapped PlantDoc samples.
        """
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        """
        Load one mapped PlantDoc image-label pair.
        """
        image_path, label, _, _ = self.samples[index]

        image = Image.open(image_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, label


class CSVImageDataset(Dataset):
    """
    Generic image dataset backed by a split CSV.

    This is used for datasets where physically copying files into class folders
    would be unnecessary or messy, especially PlantWild_v2 and FieldPlant.

    Required CSV columns:
        image_path
        label_index

    Optional metadata columns may include:
        source_dataset
        plantwild_label
        plantguard_label
        class_name
        mapping_status
    """

    def __init__(
        self,
        csv_path: str | Path,
        transform: transforms.Compose | None = None,
    ) -> None:
        """
        Load image paths and labels from a CSV file.

        Args:
            csv_path:
                Path to a split CSV file.
            transform:
                Optional transform pipeline applied to each image.
        """
        self.csv_path = Path(csv_path)
        self.transform = transform

        if not self.csv_path.exists():
            raise FileNotFoundError(f"CSV file not found: {self.csv_path}")

        self.dataframe = pd.read_csv(self.csv_path)

        required_columns = {"image_path", "label_index"}
        missing_columns = required_columns - set(self.dataframe.columns)

        if missing_columns:
            raise ValueError(
                f"{self.csv_path} is missing required columns: "
                f"{sorted(missing_columns)}"
            )

        self.samples = []

        for _, row in self.dataframe.iterrows():
            image_path = resolve_project_path(row["image_path"])
            label = int(row["label_index"])

            if not image_path.exists():
                raise FileNotFoundError(f"Image not found: {image_path}")

            self.samples.append((image_path, label))

        if not self.samples:
            raise RuntimeError(f"No samples found in CSV: {self.csv_path}")

    def __len__(self) -> int:
        """
        Return the number of CSV-backed samples.
        """
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        """
        Load one image-label pair.

        Args:
            index:
                Sample index.

        Returns:
            image:
                Transformed image tensor.
            label:
                Integer label index.
        """
        image_path, label = self.samples[index]

        image = Image.open(image_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, label


def get_datasets() -> tuple[PlantDiseaseDataset, PlantDiseaseDataset, PlantDiseaseDataset]:
    """
    Create PlantVillage train, validation, and test datasets.

    Returns:
        train_dataset:
            Augmented training dataset.
        val_dataset:
            Deterministic validation dataset.
        test_dataset:
            Deterministic test dataset.
    """
    train_transforms, eval_transforms = get_transforms()

    train_dataset = PlantDiseaseDataset(
        root_dir=TRAIN_DIR,
        transform=train_transforms,
    )

    val_dataset = PlantDiseaseDataset(
        root_dir=VAL_DIR,
        transform=eval_transforms,
    )

    test_dataset = PlantDiseaseDataset(
        root_dir=TEST_DIR,
        transform=eval_transforms,
    )

    return train_dataset, val_dataset, test_dataset


def get_dataloaders(
    batch_size: int = BATCH_SIZE,
    num_workers: int = NUM_WORKERS,
) -> tuple[DataLoader, DataLoader, DataLoader, list[str]]:
    """
    Create PlantVillage train, validation, and test dataloaders.

    Args:
        batch_size:
            Number of images per batch.
        num_workers:
            Number of DataLoader worker processes.

    Returns:
        train_loader:
            Shuffled training DataLoader.
        val_loader:
            Validation DataLoader.
        test_loader:
            Test DataLoader.
        class_names:
            PlantVillage class names in label-index order.
    """
    train_dataset, val_dataset, test_dataset = get_datasets()

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
    )

    val_loader = DataLoader(
        dataset=val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    class_names = train_dataset.class_names

    return train_loader, val_loader, test_loader, class_names


def get_plantdoc_dataset(
    class_names: list[str],
    splits: tuple[str, ...] = ("train", "test"),
    transform_type: str = "eval",
) -> PlantDocEvaluationDataset:
    """
    Create a PlantDoc dataset mapped into the current PlantGuard label space.

    Args:
        class_names:
            Target class names in model label-index order.
        splits:
            PlantDoc folders to load.
            Examples: ("train",), ("test",), or ("train", "test").
        transform_type:
            "train" for augmentation or "eval" for deterministic transforms.

    Returns:
        PlantDocEvaluationDataset.
    """
    selected_transform = select_transform(transform_type)

    return PlantDocEvaluationDataset(
        root_dir=PLANTDOC_DIR,
        class_names=class_names,
        transform=selected_transform,
        splits=splits,
    )


def get_plantdoc_loader(
    class_names: list[str],
    batch_size: int = BATCH_SIZE,
    num_workers: int = NUM_WORKERS,
    splits: tuple[str, ...] = ("train", "test"),
    transform_type: str = "eval",
    shuffle: bool = False,
) -> tuple[DataLoader, PlantDocEvaluationDataset]:
    """
    Create a DataLoader for mapped PlantDoc samples.

    Args:
        class_names:
            Target class names in model label-index order.
        batch_size:
            Number of images per batch.
        num_workers:
            Number of DataLoader worker processes.
        splits:
            PlantDoc splits to load.
        transform_type:
            "train" or "eval".
        shuffle:
            Whether to shuffle the DataLoader.

    Returns:
        plantdoc_loader:
            PlantDoc DataLoader.
        plantdoc_dataset:
            Dataset object with mapping/skipped-class metadata.
    """
    plantdoc_dataset = get_plantdoc_dataset(
        class_names=class_names,
        splits=splits,
        transform_type=transform_type,
    )

    plantdoc_loader = DataLoader(
        dataset=plantdoc_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
    )

    return plantdoc_loader, plantdoc_dataset


def get_plantdoc_finetune_loaders(
    class_names: list[str],
    batch_size: int = BATCH_SIZE,
    num_workers: int = NUM_WORKERS,
    val_ratio: float = 0.2,
    seed: int = SEED,
) -> tuple[DataLoader, DataLoader, DataLoader, Subset, Subset, PlantDocEvaluationDataset]:
    """
    Create PlantDoc train-adapt, validation-adapt, and held-out test loaders.

    PlantDoc train images are split into:
        - train-adapt subset with training augmentation
        - val-adapt subset with deterministic evaluation transforms

    The held-out PlantDoc test folder is kept separate and is not used for
    adaptation.

    Args:
        class_names:
            Target class names in model label-index order.
        batch_size:
            Number of images per batch.
        num_workers:
            Number of DataLoader worker processes.
        val_ratio:
            Fraction of PlantDoc train samples used for validation-adapt.
        seed:
            Random seed for deterministic splitting.

    Returns:
        train_loader:
            PlantDoc train-adapt loader.
        val_loader:
            PlantDoc validation-adapt loader.
        test_loader:
            PlantDoc held-out test loader.
        train_subset:
            Training Subset object.
        val_subset:
            Validation Subset object.
        test_dataset:
            Held-out PlantDoc test dataset.
    """
    if not 0.0 < val_ratio < 1.0:
        raise ValueError("val_ratio must be between 0 and 1.")

    plantdoc_train_aug = get_plantdoc_dataset(
        class_names=class_names,
        splits=("train",),
        transform_type="train",
    )

    plantdoc_train_eval = get_plantdoc_dataset(
        class_names=class_names,
        splits=("train",),
        transform_type="eval",
    )

    if len(plantdoc_train_aug) != len(plantdoc_train_eval):
        raise RuntimeError(
            "PlantDoc train dataset length mismatch between train/eval transforms."
        )

    num_samples = len(plantdoc_train_aug)
    num_val = max(1, int(num_samples * val_ratio))
    num_train = num_samples - num_val

    if num_train <= 0:
        raise RuntimeError(
            f"PlantDoc train split is too small: {num_samples} samples."
        )

    generator = torch.Generator().manual_seed(seed)
    shuffled_indices = torch.randperm(num_samples, generator=generator).tolist()

    val_indices = shuffled_indices[:num_val]
    train_indices = shuffled_indices[num_val:]

    train_subset = Subset(plantdoc_train_aug, train_indices)
    val_subset = Subset(plantdoc_train_eval, val_indices)

    test_dataset = get_plantdoc_dataset(
        class_names=class_names,
        splits=("test",),
        transform_type="eval",
    )

    train_loader = DataLoader(
        dataset=train_subset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )

    val_loader = DataLoader(
        dataset=val_subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    print("PlantDoc fine-tuning split:")
    print(f"  Train-adapt samples: {num_train}")
    print(f"  Val-adapt samples:   {num_val}")
    print(f"  Test samples:        {len(test_dataset)}")
    print(f"  Train skipped:       {plantdoc_train_aug.skipped_classes}")
    print(f"  Test skipped:        {test_dataset.skipped_classes}")

    return (
        train_loader,
        val_loader,
        test_loader,
        train_subset,
        val_subset,
        test_dataset,
    )


def load_expanded_class_names() -> list[str]:
    """
    Load the 132-class expanded PlantGuard label list.

    The expanded class list is created by:

        python data/prepare_plantwild.py

    Label convention:
        indices 0-37   -> original PlantVillage labels
        indices 38-131 -> new PlantWild_v2 labels

    Returns:
        Expanded class names in model output-index order.
    """
    if not EXPANDED_CLASS_NAMES_JSON.exists():
        raise FileNotFoundError(
            f"Expanded class-name file not found: {EXPANDED_CLASS_NAMES_JSON}\n"
            "Run: python data/prepare_plantwild.py"
        )

    with EXPANDED_CLASS_NAMES_JSON.open("r", encoding="utf-8") as file:
        class_names = json.load(file)

    if not isinstance(class_names, list):
        raise TypeError("Expanded class-name JSON must contain a list.")

    if len(class_names) != EXPECTED_EXPANDED_NUM_CLASSES:
        raise RuntimeError(
            f"Expected {EXPECTED_EXPANDED_NUM_CLASSES} expanded classes, "
            f"found {len(class_names)}."
        )

    return class_names


def validate_expanded_class_order(
    expanded_class_names: list[str],
    original_class_names: list[str],
) -> None:
    """
    Verify that original labels remain first in the expanded label list.

    This is critical because PlantDoc-fine-tuned checkpoints start from a
    38-class classifier head. During expansion, classifier rows 0-37 are copied
    into the new 132-class head, so the first 38 labels must be identical.

    Args:
        expanded_class_names:
            Expanded PlantGuard class-name list.
        original_class_names:
            Original PlantVillage class-name list.

    Raises:
        RuntimeError:
            If the first original-class slots do not match exactly.
    """
    original_count = len(original_class_names)

    if list(expanded_class_names[:original_count]) != list(original_class_names):
        raise RuntimeError(
            "Expanded class order is incompatible with the original PlantGuard "
            "class order. The first original labels must exactly match "
            "PlantVillage."
        )


def get_plantwild_dataset(
    split: str,
    transform_type: str = "eval",
) -> CSVImageDataset:
    """
    Create a PlantWild_v2 CSV-backed dataset.

    Args:
        split:
            One of "train", "val", or "test".
        transform_type:
            "train" for augmentation or "eval" for deterministic transforms.

    Returns:
        CSVImageDataset for the requested PlantWild split.
    """
    split_to_csv = {
        "train": PLANTWILD_TRAIN_CSV,
        "val": PLANTWILD_VAL_CSV,
        "test": PLANTWILD_TEST_CSV,
    }

    if split not in split_to_csv:
        raise ValueError(
            f"Unsupported PlantWild split: {split}. "
            "Choose 'train', 'val', or 'test'."
        )

    selected_transform = select_transform(transform_type)

    return CSVImageDataset(
        csv_path=split_to_csv[split],
        transform=selected_transform,
    )


def get_plantwild_loader(
    split: str,
    batch_size: int = BATCH_SIZE,
    num_workers: int = NUM_WORKERS,
    transform_type: str = "eval",
    shuffle: bool = False,
) -> tuple[DataLoader, CSVImageDataset]:
    """
    Create a DataLoader for one PlantWild_v2 split.

    Args:
        split:
            "train", "val", or "test".
        batch_size:
            Number of images per batch.
        num_workers:
            Number of DataLoader workers.
        transform_type:
            "train" or "eval".
        shuffle:
            Whether to shuffle the loader.

    Returns:
        loader:
            PlantWild DataLoader.
        dataset:
            PlantWild CSV dataset.
    """
    dataset = get_plantwild_dataset(
        split=split,
        transform_type=transform_type,
    )

    loader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
    )

    return loader, dataset


def get_fieldplant_dataset(transform_type: str = "eval") -> CSVImageDataset:
    """
    Create the FieldPlant external test dataset.

    FieldPlant is originally a YOLO detection-style dataset. It is converted
    into a compatible classification CSV by:

        python data/prepare_fieldplant.py

    Args:
        transform_type:
            "train" or "eval". Evaluation is the normal choice.

    Returns:
        CSVImageDataset for FieldPlant external evaluation.
    """
    selected_transform = select_transform(transform_type)

    return CSVImageDataset(
        csv_path=FIELDPLANT_TEST_CSV,
        transform=selected_transform,
    )


def get_fieldplant_loader(
    batch_size: int = BATCH_SIZE,
    num_workers: int = NUM_WORKERS,
    transform_type: str = "eval",
    shuffle: bool = False,
) -> tuple[DataLoader, CSVImageDataset]:
    """
    Create the FieldPlant external test DataLoader.

    Args:
        batch_size:
            Number of images per batch.
        num_workers:
            Number of DataLoader worker processes.
        transform_type:
            "train" or "eval".
        shuffle:
            Whether to shuffle the loader.

    Returns:
        loader:
            FieldPlant DataLoader.
        dataset:
            FieldPlant CSV dataset.
    """
    dataset = get_fieldplant_dataset(transform_type=transform_type)

    loader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
    )

    return loader, dataset


def make_mixed_replay_sampler(
    plantwild_dataset: Dataset,
    plantdoc_dataset: Dataset,
    plantvillage_dataset: Dataset,
    plantwild_ratio: float = 0.80,
    plantdoc_ratio: float = 0.12,
    plantvillage_ratio: float = 0.08,
    seed: int = SEED,
) -> WeightedRandomSampler:
    """
    Create a WeightedRandomSampler for mixed expanded training.

    Desired sampling ratio:
        80% PlantWild_v2
        12% PlantDoc train-adapt replay
        8% PlantVillage replay

    The sampler uses replacement so that each mini-epoch can follow the desired
    mixture even when source datasets have very different sizes.

    Args:
        plantwild_dataset:
            PlantWild training dataset.
        plantdoc_dataset:
            PlantDoc train-adapt subset.
        plantvillage_dataset:
            PlantVillage training dataset.
        plantwild_ratio:
            Desired PlantWild sampling mass.
        plantdoc_ratio:
            Desired PlantDoc sampling mass.
        plantvillage_ratio:
            Desired PlantVillage sampling mass.
        seed:
            Random seed for reproducibility.

    Returns:
        WeightedRandomSampler for the concatenated mixed dataset.
    """
    ratios = [plantwild_ratio, plantdoc_ratio, plantvillage_ratio]
    ratio_sum = sum(ratios)

    if abs(ratio_sum - 1.0) > 1e-6:
        raise ValueError(
            "Replay ratios must sum to 1. "
            f"Got: {ratio_sum}"
        )

    lengths = [
        len(plantwild_dataset),
        len(plantdoc_dataset),
        len(plantvillage_dataset),
    ]

    if any(length == 0 for length in lengths):
        raise RuntimeError(
            f"Cannot create mixed sampler with empty dataset lengths: {lengths}"
        )

    weights = []

    for ratio, length in zip(ratios, lengths):
        per_sample_weight = ratio / length
        weights.extend([per_sample_weight] * length)

    # Define one mixed epoch so PlantWild contributes approximately one full pass.
    # Example: if PlantWild is 80% of training, total epoch size is PlantWild / 0.8.
    num_samples = int(round(len(plantwild_dataset) / plantwild_ratio))

    generator = torch.Generator().manual_seed(seed)

    return WeightedRandomSampler(
        weights=torch.DoubleTensor(weights),
        num_samples=num_samples,
        replacement=True,
        generator=generator,
    )


def get_expanded_training_loaders(
    batch_size: int = BATCH_SIZE,
    num_workers: int = NUM_WORKERS,
    plantwild_ratio: float = 0.80,
    plantdoc_ratio: float = 0.12,
    plantvillage_ratio: float = 0.08,
    plantdoc_val_ratio: float = 0.2,
    seed: int = SEED,
) -> tuple[DataLoader, DataLoader, DataLoader, DataLoader, list[str], dict]:
    """
    Create dataloaders for expanded 132-class PlantGuard training.

    Training loader:
        Uses a mixed replay sampler:
            80% PlantWild_v2
            12% PlantDoc train-adapt replay
            8% PlantVillage replay

    Validation loaders:
        Returned separately so training code can track:
            PlantWild validation performance
            PlantDoc validation retention
            PlantVillage validation retention

    Args:
        batch_size:
            Number of images per batch.
        num_workers:
            Number of DataLoader workers.
        plantwild_ratio:
            PlantWild training ratio.
        plantdoc_ratio:
            PlantDoc replay ratio.
        plantvillage_ratio:
            PlantVillage replay ratio.
        plantdoc_val_ratio:
            PlantDoc train/validation split ratio.
        seed:
            Random seed.

    Returns:
        mixed_train_loader:
            Combined training loader using weighted replay sampling.
        plantwild_val_loader:
            PlantWild validation loader.
        plantdoc_val_loader:
            PlantDoc validation-adapt loader.
        plantvillage_val_loader:
            PlantVillage validation loader.
        expanded_class_names:
            Expanded 132-class label list.
        metadata:
            Dataset sizes and sampling-ratio metadata.
    """
    expanded_class_names = load_expanded_class_names()

    plantvillage_train_dataset, plantvillage_val_dataset, _ = get_datasets()
    original_class_names = plantvillage_train_dataset.class_names

    validate_expanded_class_order(
        expanded_class_names=expanded_class_names,
        original_class_names=original_class_names,
    )

    plantwild_train_dataset = get_plantwild_dataset(
        split="train",
        transform_type="train",
    )

    plantwild_val_dataset = get_plantwild_dataset(
        split="val",
        transform_type="eval",
    )

    (
        _plantdoc_train_loader,
        _plantdoc_val_loader,
        _plantdoc_test_loader,
        plantdoc_train_subset,
        plantdoc_val_subset,
        _plantdoc_test_dataset,
    ) = get_plantdoc_finetune_loaders(
        class_names=expanded_class_names,
        batch_size=batch_size,
        num_workers=num_workers,
        val_ratio=plantdoc_val_ratio,
        seed=seed,
    )

    mixed_train_dataset = ConcatDataset(
        [
            plantwild_train_dataset,
            plantdoc_train_subset,
            plantvillage_train_dataset,
        ]
    )

    mixed_sampler = make_mixed_replay_sampler(
        plantwild_dataset=plantwild_train_dataset,
        plantdoc_dataset=plantdoc_train_subset,
        plantvillage_dataset=plantvillage_train_dataset,
        plantwild_ratio=plantwild_ratio,
        plantdoc_ratio=plantdoc_ratio,
        plantvillage_ratio=plantvillage_ratio,
        seed=seed,
    )

    mixed_train_loader = DataLoader(
        dataset=mixed_train_dataset,
        batch_size=batch_size,
        sampler=mixed_sampler,
        num_workers=num_workers,
        pin_memory=True,
    )

    plantwild_val_loader = DataLoader(
        dataset=plantwild_val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    plantdoc_val_loader = DataLoader(
        dataset=plantdoc_val_subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    plantvillage_val_loader = DataLoader(
        dataset=plantvillage_val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    metadata = {
        "plantwild_train_samples": len(plantwild_train_dataset),
        "plantdoc_train_replay_samples": len(plantdoc_train_subset),
        "plantvillage_train_replay_samples": len(plantvillage_train_dataset),
        "mixed_epoch_samples": len(mixed_sampler),
        "expanded_num_classes": len(expanded_class_names),
        "plantwild_ratio": plantwild_ratio,
        "plantdoc_ratio": plantdoc_ratio,
        "plantvillage_ratio": plantvillage_ratio,
    }

    print("\nExpanded PlantGuard training loaders:")
    print(f"  Expanded classes:          {len(expanded_class_names)}")
    print(f"  PlantWild train samples:   {len(plantwild_train_dataset)}")
    print(f"  PlantDoc replay samples:   {len(plantdoc_train_subset)}")
    print(f"  PlantVillage replay:       {len(plantvillage_train_dataset)}")
    print(f"  Mixed epoch samples:       {len(mixed_sampler)}")
    print(f"  Mixed train batches:       {len(mixed_train_loader)}")
    print(f"  PlantWild val samples:     {len(plantwild_val_dataset)}")
    print(f"  PlantDoc val samples:      {len(plantdoc_val_subset)}")
    print(f"  PlantVillage val samples:  {len(plantvillage_val_dataset)}")

    return (
        mixed_train_loader,
        plantwild_val_loader,
        plantdoc_val_loader,
        plantvillage_val_loader,
        expanded_class_names,
        metadata,
    )


def calculate_mean_std(
    batch_size: int = BATCH_SIZE,
    num_workers: int = NUM_WORKERS,
) -> tuple[list[float], list[float]]:
    """
    Calculate RGB mean and standard deviation from the PlantVillage train split.

    This should be run only when the training dataset or split changes. The
    resulting values can then be copied into DATASET_MEAN and DATASET_STD.

    Args:
        batch_size:
            Number of images per batch.
        num_workers:
            Number of DataLoader worker processes.

    Returns:
        mean:
            RGB channel mean.
        std:
            RGB channel standard deviation.
    """
    stats_transform = transforms.Compose(
        [
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor(),
        ]
    )

    train_dataset = PlantDiseaseDataset(
        root_dir=TRAIN_DIR,
        transform=stats_transform,
    )

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    channel_sum = torch.zeros(3)
    channel_squared_sum = torch.zeros(3)
    total_pixels = 0

    for images, _ in train_loader:
        current_batch_size, _, height, width = images.shape
        pixels_per_batch = current_batch_size * height * width

        channel_sum += images.sum(dim=[0, 2, 3])
        channel_squared_sum += (images**2).sum(dim=[0, 2, 3])
        total_pixels += pixels_per_batch

    mean = channel_sum / total_pixels
    std = torch.sqrt((channel_squared_sum / total_pixels) - (mean**2))

    return mean.tolist(), std.tolist()


def smoke_test_plantvillage() -> list[str]:
    """
    Run a quick PlantVillage DataLoader smoke test.

    Returns:
        PlantVillage class names.
    """
    train_loader, val_loader, test_loader, class_names = get_dataloaders()

    print(f"Number of classes: {len(class_names)}")
    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")
    print(f"Test batches: {len(test_loader)}")

    images, labels = next(iter(train_loader))

    print(f"Image batch shape: {images.shape}")
    print(f"Label batch shape: {labels.shape}")
    print(f"First 5 labels: {labels[:5]}")

    return class_names


def smoke_test_plantdoc(class_names: list[str]) -> None:
    """
    Run PlantDoc loader smoke tests when PlantDoc data exists.
    """
    if not PLANTDOC_DIR.exists():
        print("\nPlantDoc directory not found. Skipping PlantDoc smoke test.")
        return

    print("\n--- PlantDoc zero-shot external evaluation loader ---")

    plantdoc_loader, plantdoc_dataset = get_plantdoc_loader(
        class_names=class_names,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        splits=("train", "test"),
        transform_type="eval",
        shuffle=False,
    )

    print(f"PlantDoc full samples: {len(plantdoc_dataset)}")
    print(f"PlantDoc full batches: {len(plantdoc_loader)}")
    print(f"PlantDoc full skipped classes: {plantdoc_dataset.skipped_classes}")

    print("\n--- PlantDoc fine-tuning loaders ---")

    (
        plantdoc_train_loader,
        plantdoc_val_loader,
        plantdoc_test_loader,
        plantdoc_train_subset,
        plantdoc_val_subset,
        plantdoc_test_dataset,
    ) = get_plantdoc_finetune_loaders(
        class_names=class_names,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        val_ratio=0.2,
        seed=SEED,
    )

    print(f"PlantDoc fine-tune train subset samples: {len(plantdoc_train_subset)}")
    print(f"PlantDoc fine-tune val subset samples: {len(plantdoc_val_subset)}")
    print(f"PlantDoc held-out test samples: {len(plantdoc_test_dataset)}")
    print(f"PlantDoc fine-tune train batches: {len(plantdoc_train_loader)}")
    print(f"PlantDoc fine-tune val batches: {len(plantdoc_val_loader)}")
    print(f"PlantDoc held-out test batches: {len(plantdoc_test_loader)}")


def smoke_test_expanded_training() -> None:
    """
    Run expanded mixed-training loader smoke test when files exist.
    """
    if not (EXPANDED_CLASS_NAMES_JSON.exists() and PLANTWILD_TRAIN_CSV.exists()):
        print("\nExpanded PlantWild files not found. Skipping expanded loader smoke test.")
        return

    print("\n--- Expanded PlantGuard mixed training loader ---")

    (
        mixed_train_loader,
        _plantwild_val_loader,
        _plantdoc_val_loader,
        _plantvillage_val_loader,
        expanded_class_names,
        metadata,
    ) = get_expanded_training_loaders(
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        plantwild_ratio=0.80,
        plantdoc_ratio=0.12,
        plantvillage_ratio=0.08,
        plantdoc_val_ratio=0.2,
        seed=SEED,
    )

    images, labels = next(iter(mixed_train_loader))

    print(f"Expanded classes: {len(expanded_class_names)}")
    print(f"Mixed image batch shape: {images.shape}")
    print(f"Mixed label batch shape: {labels.shape}")
    print(f"Mixed label min: {labels.min().item()}")
    print(f"Mixed label max: {labels.max().item()}")
    print(f"Metadata: {metadata}")


def smoke_test_fieldplant() -> None:
    """
    Run FieldPlant loader smoke test when FieldPlant files exist.
    """
    if not (FIELDPLANT_TEST_CSV.exists() and EXPANDED_CLASS_NAMES_JSON.exists()):
        print(
            "\nFieldPlant test CSV or expanded labels not found. "
            "Skipping FieldPlant smoke test."
        )
        return

    print("\n--- FieldPlant external test loader ---")

    expanded_class_names = load_expanded_class_names()

    fieldplant_loader, fieldplant_dataset = get_fieldplant_loader(
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        transform_type="eval",
        shuffle=False,
    )

    print(f"FieldPlant samples: {len(fieldplant_dataset)}")
    print(f"FieldPlant batches: {len(fieldplant_loader)}")

    images, labels = next(iter(fieldplant_loader))

    print(f"FieldPlant image batch shape: {images.shape}")
    print(f"FieldPlant label batch shape: {labels.shape}")
    print(f"FieldPlant label min: {labels.min().item()}")
    print(f"FieldPlant label max: {labels.max().item()}")
    print(f"FieldPlant first 5 labels: {labels[:5]}")

    first_5_class_names = [
        expanded_class_names[int(label)]
        for label in labels[:5]
    ]

    print(f"FieldPlant first 5 class names: {first_5_class_names}")

    if hasattr(fieldplant_dataset, "dataframe") and "class_name" in fieldplant_dataset.dataframe:
        print("\nFieldPlant CSV class distribution:")
        print(
            fieldplant_dataset.dataframe["class_name"]
            .value_counts()
            .sort_index()
            .to_string()
        )


if __name__ == "__main__":
    plantvillage_class_names = smoke_test_plantvillage()
    smoke_test_plantdoc(class_names=plantvillage_class_names)
    smoke_test_expanded_training()
    smoke_test_fieldplant()