"""
Dataset utilities for the PlantGuard project.

This module defines a custom PyTorch Dataset for PlantVillage-style
class-folder image datasets and provides helper functions to create
train, validation, and test DataLoaders.
"""

from pathlib import Path
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms


PROJECT_ROOT = Path(__file__).resolve().parents[1]

TRAIN_DIR = PROJECT_ROOT / "data" / "train"
VAL_DIR = PROJECT_ROOT / "data" / "val"
TEST_DIR = PROJECT_ROOT / "data" / "test"
PLANTDOC_DIR = PROJECT_ROOT / "data" / "raw" / "plantdoc"

IMAGE_SIZE = 256
BATCH_SIZE = 32
NUM_WORKERS = 2

# Dataset-specific normalization values calculated from the training split only.
DATASET_MEAN = [0.4665524959564209, 0.48931649327278137, 0.4102632701396942]
DATASET_STD = [0.19912923872470856, 0.174818754196167, 0.21729066967964172]

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}

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


def get_plantdoc_dataset(class_names):
    """
    Create the PlantDoc external evaluation dataset.

    Args:
        class_names: PlantVillage class names in the same order used by the model.

    Returns:
        PlantDocEvaluationDataset.
    """
    _, eval_transforms = get_transforms()

    plantdoc_dataset = PlantDocEvaluationDataset(
        root_dir=PLANTDOC_DIR,
        class_names=class_names,
        transform=eval_transforms,
    )

    return plantdoc_dataset


def get_plantdoc_loader(
        class_names,
        batch_size = BATCH_SIZE,
        num_workers = NUM_WORKERS,
):
    """
    Create a DataLoader for PlantDoc external evaluation.

    Args:
        class_names: PlantVillage class names in model label-index order.
        batch_size: Number of images per batch.
        num_workers: Number of DataLoader worker processes.

    Returns:
        plantdoc_loader: DataLoader for PlantDoc.
        plantdoc_dataset: Dataset object with metadata about mapping/skipped classes.
    """
    plantdoc_dataset = get_plantdoc_dataset(class_names=class_names)

    plantdoc_loader = DataLoader(
        dataset=plantdoc_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    return plantdoc_loader, plantdoc_dataset


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
        plantdoc_loader, plantdoc_dataset = get_plantdoc_loader(
            class_names=class_names,
            batch_size=BATCH_SIZE,
            num_workers=NUM_WORKERS,
        )

        print(f"PlantDoc samples: {len(plantdoc_dataset)}")
        print(f"PlantDoc batches: {len(plantdoc_loader)}")
        print(f"PlantDoc skipped classes: {plantdoc_dataset.skipped_classes}")
