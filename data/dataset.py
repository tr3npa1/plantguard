"""
Dataset utilities for the PlantGuard project.

This module defines a custom PyTorch Dataset for PlantVillage-style
class-folder image datasets and provides helper functions to create
train, validation, and test DataLoaders.
"""

from pathlib import Path
import json

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

IMAGE_SIZE = 256
BATCH_SIZE = 32
NUM_WORKERS = 2

# Dataset-specific normalization values calculated from the training split only.
DATASET_MEAN = [0.4665524959564209, 0.48931649327278137, 0.4102632701396942]
DATASET_STD = [0.19912923872470856, 0.174818754196167, 0.21729066967964172]

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

SEED = 42

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
    "Tomato two spotted spider mites leaf": "Tomato___Spider_mites Two-spotted_spider_mite",
}


class PlantDiseaseDataset(Dataset):
    """
    Custom PyTorch Dataset for plant disease image classification.

    The dataset expects a folder structure where each class has its own
    subfolder. It scans those class folders, creates a stable class-to-index
    mapping from sorted class names, stores image paths with their numeric
    labels, and applies an optional transform when returning each sample.
    """

    def __init__(self, root_dir, transform=None):
        """
        Initialize the dataset by scanning class folders and collecting image paths.

        Args:
            root_dir: Path to a split folder such as data/train, data/val, or data/test.
            transform: Optional torchvision transform pipeline to apply to each image.
        """
        self.root_dir = Path(root_dir)
        self.transform = transform

        # Allowed image file types.
        self.image_extensions = IMAGE_EXTENSIONS

        # Sort class names so label mapping is stable and reproducible.
        self.class_names = sorted([
            folder.name
            for folder in self.root_dir.iterdir()
            if folder.is_dir()
        ])

        # Map class names to integer labels because neural networks need numeric targets.
        self.class_to_idx = {
            class_name: idx
            for idx, class_name in enumerate(self.class_names)
        }

        # Store each sample as (image_path, label).
        self.samples = []

        for class_name in self.class_names:
            class_dir = self.root_dir / class_name
            label = self.class_to_idx[class_name]

            for image_path in class_dir.iterdir():
                if (
                    image_path.is_file()
                    and image_path.suffix.lower() in self.image_extensions
                ):
                    self.samples.append((image_path, label))

    def __len__(self):
        """
        Return the total number of image samples in the dataset.
        """
        return len(self.samples)

    def __getitem__(self, index):
        """
        Load and return one image-label pair by index.

        Args:
            index: Position of the sample in self.samples.

        Returns:
            image: Transformed image tensor.
            label: Integer class label.
        """
        image_path, label = self.samples[index]

        # Convert every image to RGB so all samples have 3 channels.
        image = Image.open(image_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, label
    

class PlantDocEvaluationDataset(Dataset):
    """
    Dataset for evaluating PlantVillage-trained models on PlantDoc.

    PlantDoc has different folder names from PlantVillage, so this dataset maps
    PlantDoc class folder names to PlantVillage class labels. It combines the
    PlantDoc train and test folders into one external evaluation dataset.

    This dataset is for evaluation only. It should not be used for training.
    """

    def __init__(
            self,
            root_dir,
            class_names,
            transform=None,
            splits=("train", "test")
    ):
        self.root_dir = Path(root_dir)
        self.class_names = list(class_names)
        self.transform = transform
        self.splits = splits

        self.class_to_idx = {
            class_name: idx
            for idx, class_name in enumerate(self.class_names)
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

                image_paths = sorted([
                    image_path
                    for image_path in plantdoc_class_dir.iterdir()
                    if(
                        image_path.is_file()
                        and image_path.suffix.lower() in IMAGE_EXTENSIONS
                    )
                ])

                if plantdoc_class_name not in PLANTDOC_TO_PLANTVILLAGE:
                    self.skipped_classes[plantdoc_class_name] = (
                        self.skipped_classes.get(plantdoc_class_name,0)
                        + len(image_paths)
                    )
                    continue

                plantvillage_class_name = PLANTDOC_TO_PLANTVILLAGE[plantdoc_class_name]

                if plantvillage_class_name not in self.class_to_idx:
                    self.skipped_classes[plantdoc_class_name] = (
                        self.skipped_classes.get(plantdoc_class_name,0)
                        + len(image_paths)
                    )
                    continue

                label = self.class_to_idx[plantvillage_class_name]

                for image_path in image_paths:
                    self.samples.append(
                        (
                            image_path,
                            label,
                            plantdoc_class_name,
                            plantvillage_class_name,
                        )
                    )

                self.mapped_class_counts[plantvillage_class_name] = (
                    self.mapped_class_counts.get(plantvillage_class_name,0)
                    + len(image_paths)
                )
            
        if len(self.samples)==0:
            raise RuntimeError(
                f"No mapped PlantDoc images found in {self.root_dir}. "
                "Check that PlantDoc is downloaded and label mapping is correct."
            )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        image_path, label, _, _ = self.samples[index]

        image = Image.open(image_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, label       


class CSVImageDataset(Dataset):
    """
    Generic image dataset backed by a CSV file.

    This is used for PlantWild_v2 because we prepared train/val/test split CSVs
    instead of physically copying images into split folders.

    Expected CSV columns:
        image_path
        label_index

    Optional metadata columns:
        source_dataset
        plantwild_label
        plantguard_label
        mapping_status
    """

    def __init__(self, csv_path, transform=None):
        """
        Initialize a CSV-backed image dataset.

        Args:
            csv_path: Path to a split CSV file.
            transform: Optional torchvision transform pipeline.
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
            image_path = PROJECT_ROOT / row["image_path"]
            label = int(row["label_index"])

            if not image_path.exists():
                raise FileNotFoundError(f"Image not found: {image_path}")

            self.samples.append((image_path, label))

    
    def __len__(self):
        """
        Return the number of samples in the CSV dataset.
        """
        return len(self.samples)
    
    def __getitem__(self,index):
        """
        Load and return one image-label pair.

        Args:
            index: Sample index.

        Returns:
            image: Transformed image tensor.
            label: Integer label index.
        """
        image_path, label = self.samples[index]

        image = Image.open(image_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, label


def get_transforms():
    """
    Create transform pipelines for training and evaluation.

    Training transforms include augmentation to improve generalization.
    Validation and test transforms are deterministic and do not use random
    augmentation.
    """
    # Training uses random augmentation so the model learns disease patterns,
    # not exact image positions, lighting, or orientation.
    train_transforms = transforms.Compose([
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
    ])

    # Validation and test use deterministic transforms for fair evaluation.
    eval_transforms = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=DATASET_MEAN,
            std=DATASET_STD,
        ),
    ])

    return train_transforms, eval_transforms


def get_datasets():
    """
    Create train, validation, and test Dataset objects.

    Returns:
        train_dataset: Dataset for the training split.
        val_dataset: Dataset for the validation split.
        test_dataset: Dataset for the test split.
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


def get_dataloaders(batch_size=BATCH_SIZE, num_workers=NUM_WORKERS):
    """
    Create DataLoaders for training, validation, and testing.

    Args:
        batch_size: Number of images per batch.
        num_workers: Number of worker processes used for loading data.

    Returns:
        train_loader: DataLoader for the training split.
        val_loader: DataLoader for the validation split.
        test_loader: DataLoader for the test split.
        class_names: List of class names in label-index order.
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
        class_names,
        splits=("train", "test"),
        transform_type="eval",
    ):
    """
    Create a PlantDoc dataset mapped into the PlantVillage 38-class label space.

    Args:
        class_names: PlantVillage class names in model label-index order.
        splits: PlantDoc folders to load. Examples:
            ("train",) for PlantDoc training/adaptation data.
            ("test",) for held-out PlantDoc test data.
            ("train", "test") for zero-shot external evaluation.
        transform_type: "train" for augmentation, "eval" for deterministic evaluation transforms.

    Returns:
        PlantDocEvaluationDataset.
    """
    train_transforms, eval_transforms = get_transforms()

    if transform_type == "train":
        selected_transform = train_transforms
    elif transform_type == "eval":
        selected_transform = eval_transforms
    else:
        raise ValueError(
            f"Unsupported transform_type: {transform_type}. "
            "Choose 'train' or 'eval'."
        )

    plantdoc_dataset = PlantDocEvaluationDataset(
        root_dir=PLANTDOC_DIR,
        class_names=class_names,
        transform=selected_transform,
        splits=splits,
    )

    return plantdoc_dataset


def get_plantdoc_loader(
        class_names,
        batch_size = BATCH_SIZE,
        num_workers = NUM_WORKERS,
        splits=("train","test"),
        transform_type="eval",
        shuffle=False,
):
    """
    Create a DataLoader for PlantDoc external evaluation.

    Args:
        class_names: PlantVillage class names in model label-index order.
        batch_size: Number of images per batch.
        num_workers: Number of DataLoader worker processes.
        splits: PlantDoc splits to load. Example: ("train",), ("test",), or ("train", "test").
        transform_type: "train" for augmentation, "eval" for deterministic transforms.
        shuffle: Whether to shuffle the DataLoader.

    Returns:
        plantdoc_loader: DataLoader for PlantDoc.
        plantdoc_dataset: Dataset object with metadata about mapping/skipped classes.
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
        class_names,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        val_ratio=0.2,
        seed=SEED
):
    if not 0.0 < val_ratio < 1.0:
        raise ValueError("val_ratio must be between 0 and 1.")

    # Same PlantDoc train samples, but two transform versions:
    # one augmented for training, one deterministic for validation.
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
    num_val = int(num_samples*val_ratio)
    num_val = max(1,num_val)
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


def load_expanded_class_names():
    """
    Load expanded PlantGuard class names.

    The expanded class list is produced by data/prepare_plantwild.py.

    Label convention:
        indices 0-37   -> original PlantGuard/PlantVillage labels
        indices 38+    -> new PlantWild labels

    Returns:
        List of expanded class names.
    """
    if not EXPANDED_CLASS_NAMES_JSON.exists():
        raise FileNotFoundError(
            f"Expanded class-name file not found: {EXPANDED_CLASS_NAMES_JSON}\n"
            "Run: python data/prepare_plantwild.py"
        )

    with EXPANDED_CLASS_NAMES_JSON.open("r", encoding="utf-8") as file:
        class_names = json.load(file)

    if len(class_names) <= 38:
        raise RuntimeError(
            f"Expanded class list looks wrong. Found {len(class_names)} classes."
        )

    return class_names


def validate_expanded_class_order(expanded_class_names, original_class_names):
    """
    Verify that original 38 labels remain first in the expanded label list.

    This is critical because PlantDoc-fine-tuned checkpoints have a 38-class
    classifier head. When we expand the head, rows 0-37 must correspond to the
    same original labels.

    Args:
        expanded_class_names: Expanded class-name list.
        original_class_names: Original PlantVillage class-name list.
    """
    if list(expanded_class_names[:len(original_class_names)]) != list(original_class_names):
        raise RuntimeError(
            "Expanded class order is incompatible with the original PlantGuard "
            "class order. The first 38 labels must exactly match PlantVillage."
        )


def get_plantwild_dataset(split, transform_type="eval"):
    """
    Create a PlantWild_v2 CSV-backed dataset.

    Args:
        split: One of "train", "val", or "test".
        transform_type: "train" for augmentation, "eval" for deterministic transforms.

    Returns:
        CSVImageDataset.
    """
    train_transforms, eval_transforms = get_transforms()

    if transform_type == "train":
        selected_transform = train_transforms
    elif transform_type == "eval":
        selected_transform = eval_transforms
    else:
        raise ValueError(
            f"Unsupported transform_type: {transform_type}. "
            "Choose 'train' or 'eval'."
        )
    
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

    return CSVImageDataset(
        csv_path=split_to_csv[split],
        transform=selected_transform,
    )


def get_plantwild_loader(
        split,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        transform_type="eval",
        shuffle=False,
):
    """
    Create a DataLoader for one PlantWild_v2 split.

    Args:
        split: "train", "val", or "test".
        batch_size: Number of images per batch.
        num_workers: Number of DataLoader workers.
        transform_type: "train" or "eval".
        shuffle: Whether to shuffle the loader.

    Returns:
        loader: PlantWild DataLoader.
        dataset: PlantWild dataset.
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


def make_mixed_replay_sampler(
        plantwild_dataset,
        plantdoc_dataset,
        plantvillage_dataset,
        plantwild_ratio=0.80,
        plantdoc_ratio=0.12,
        plantvillage_ratio=0.08,
        seed=SEED,
):
    """
    Create a WeightedRandomSampler for 80/12/8 mixed training.

    Desired sampling ratio:
        80% PlantWild_v2
        12% PlantDoc train-adapt replay
        8% PlantVillage replay

    Args:
        plantwild_dataset: PlantWild training dataset.
        plantdoc_dataset: PlantDoc train-adapt subset.
        plantvillage_dataset: PlantVillage training dataset.
        plantwild_ratio: Desired PlantWild sampling mass.
        plantdoc_ratio: Desired PlantDoc sampling mass.
        plantvillage_ratio: Desired PlantVillage sampling mass.
        seed: Random seed.

    Returns:
        WeightedRandomSampler.
    """

    ratios = [plantwild_ratio, plantdoc_ratio, plantvillage_ratio]

    if not torch.isclose(torch.tensor(sum(ratios)), torch.tensor(1.0), atol=1e-6):
        raise ValueError(
            "Replay ratios must sum to 1. "
            f"Got: {plantwild_ratio + plantdoc_ratio + plantvillage_ratio}"
        )
    
    lengths = [
        len(plantwild_dataset),
        len(plantdoc_dataset),
        len(plantvillage_dataset),
    ]

    if any(length == 0 for length in lengths):
        raise RuntimeError(f"Cannot create mixed sampler with empty dataset: {lengths}")
    
    weights = []

    for ratio, length in zip(ratios, lengths):
        per_sample_weight = ratio / length
        weights.extend([per_sample_weight] * length)

    # Define one mixed epoch so PlantWild contributes approximately one full pass.
    # Example: if PlantWild is 80% of training, total epoch size is PlantWild / 0.8.
    num_samples = int(round(len(plantwild_dataset) / plantwild_ratio))

    generator = torch.Generator().manual_seed(seed)

    sampler = WeightedRandomSampler(
        weights=torch.DoubleTensor(weights),
        num_samples=num_samples,
        replacement=True,
        generator=generator,
    )

    return sampler


def get_expanded_training_loaders(
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        plantwild_ratio=0.80,
        plantdoc_ratio=0.12,
        plantvillage_ratio=0.08,
        plantdoc_val_ratio=0.2,
        seed=SEED,
):
    """
    Create dataloaders for expanded PlantGuard training.

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
        batch_size: Number of images per batch.
        num_workers: Number of DataLoader workers.
        plantwild_ratio: PlantWild training ratio.
        plantdoc_ratio: PlantDoc replay ratio.
        plantvillage_ratio: PlantVillage replay ratio.
        plantdoc_val_ratio: PlantDoc train/val split ratio.
        seed: Random seed.

    Returns:
        mixed_train_loader
        plantwild_val_loader
        plantdoc_val_loader
        plantvillage_val_loader
        expanded_class_names
        metadata dictionary
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


def calculate_mean_std(batch_size=BATCH_SIZE, num_workers=NUM_WORKERS):
    """
    Calculate RGB mean and standard deviation from the training split.

    This should be run only when the training dataset or split changes.
    The resulting values can be copied into DATASET_MEAN and DATASET_STD.
    """
     
    stats_transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
    ])

    train_dataset = PlantDiseaseDataset(
        root_dir=TRAIN_DIR,
        transform=stats_transform,
    )

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )
    channel_sum = torch.zeros(3)
    channel_squared_sum = torch.zeros(3)
    total_pixels = 0

    for images, _ in train_loader:
        current_batch_size, channels, height, width = images.shape
        pixels_per_batch = current_batch_size * height * width

        channel_sum += images.sum(dim=[0,2,3])
        channel_squared_sum += (images**2).sum(dim=[0,2,3])
        total_pixels += pixels_per_batch
    
    mean = channel_sum/total_pixels
    std = torch.sqrt((channel_squared_sum / total_pixels) - (mean ** 2))

    return mean.tolist(), std.tolist()


# Smoke test: run this file directly to verify that the dataset and loaders work.
if __name__ == "__main__":
    train_loader, val_loader, test_loader, class_names = get_dataloaders()

    print(f"Number of classes: {len(class_names)}")
    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")
    print(f"Test batches: {len(test_loader)}")

    images, labels = next(iter(train_loader))

    print(f"Image batch shape: {images.shape}")
    print(f"Label batch shape: {labels.shape}")
    print(f"First 5 labels: {labels[:5]}")

    if PLANTDOC_DIR.exists():
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
            seed=42,
        )

        print(f"PlantDoc fine-tune train subset samples: {len(plantdoc_train_subset)}")
        print(f"PlantDoc fine-tune val subset samples: {len(plantdoc_val_subset)}")
        print(f"PlantDoc held-out test samples: {len(plantdoc_test_dataset)}")

        print(f"PlantDoc fine-tune train batches: {len(plantdoc_train_loader)}")
        print(f"PlantDoc fine-tune val batches: {len(plantdoc_val_loader)}")
        print(f"PlantDoc held-out test batches: {len(plantdoc_test_loader)}")
    else:
        print("\nPlantDoc directory not found. Skipping PlantDoc smoke test.")

    if EXPANDED_CLASS_NAMES_JSON.exists() and PLANTWILD_TRAIN_CSV.exists():
        print("\n--- Expanded PlantGuard mixed training loader ---")

        (
            mixed_train_loader,
            plantwild_val_loader,
            plantdoc_val_loader,
            plantvillage_val_loader,
            expanded_class_names,
            metadata,
        ) = get_expanded_training_loaders(
            batch_size=BATCH_SIZE,
            num_workers=NUM_WORKERS,
            plantwild_ratio=0.80,
            plantdoc_ratio=0.12,
            plantvillage_ratio=0.08,
            plantdoc_val_ratio=0.2,
            seed=42,
        )

        images, labels = next(iter(mixed_train_loader))

        print(f"Expanded classes: {len(expanded_class_names)}")
        print(f"Mixed image batch shape: {images.shape}")
        print(f"Mixed label batch shape: {labels.shape}")
        print(f"Mixed label min: {labels.min().item()}")
        print(f"Mixed label max: {labels.max().item()}")
        print(f"Metadata: {metadata}")
    else:
        print("\nExpanded PlantWild files not found. Skipping expanded loader smoke test.")