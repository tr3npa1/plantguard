"""
Base PlantVillage training script for PlantGuard.

This script trains a 38-class PlantGuard image-classification model on the
PlantVillage train/validation splits.

It:
    - loads hyperparameters from training/config.yaml
    - creates PlantVillage DataLoaders
    - builds the requested torchvision architecture
    - supports Cross Entropy, Weighted Cross Entropy, and Focal Loss
    - trains and validates after every epoch
    - saves latest and best checkpoints
    - logs parameters, metrics, config, and checkpoints to MLflow

This script is the Stage A training entry point. Later stages use separate
scripts for PlantDoc fine-tuning and PlantWild_v2 expanded-label training.
"""

from __future__ import annotations

import random
import sys
from pathlib import Path

import mlflow
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import yaml
from torchvision import models

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "training" / "config.yaml"
MLFLOW_DB_PATH = PROJECT_ROOT / "mlflow.db"

sys.path.append(str(PROJECT_ROOT))

from data.dataset import get_dataloaders  # noqa: E402


SUPPORTED_MODELS = {"efficientnet_b0", "efficientnet_b3", "resnet50"}
SUPPORTED_OPTIMIZERS = {"adam", "adamw"}
SUPPORTED_LOSSES = {"cross_entropy", "weighted_cross_entropy", "focal_loss"}


def load_config(config_path: str | Path) -> dict:
    """
    Load training configuration from a YAML file.

    Args:
        config_path:
            Path to training/config.yaml.

    Returns:
        Parsed configuration dictionary.
    """
    config_path = Path(config_path)

    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with config_path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    if not isinstance(config, dict):
        raise TypeError(f"Config file must contain a YAML dictionary: {config_path}")

    return config


def set_seed(seed: int) -> None:
    """
    Set random seeds for reproducible training behavior.

    Args:
        seed:
            Integer seed value.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device(device_name: str) -> torch.device:
    """
    Select the training device.

    Args:
        device_name:
            "auto", "cuda", or "cpu".

    Returns:
        Selected torch.device.
    """
    if device_name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if device_name not in {"cuda", "cpu"}:
        raise ValueError(
            f"Unsupported device: {device_name}. Choose 'auto', 'cuda', or 'cpu'."
        )

    if device_name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but torch.cuda.is_available() is False.")

    return torch.device(device_name)


def build_model(
    model_name: str,
    num_classes: int,
    pretrained: bool = True,
) -> nn.Module:
    """
    Build an image-classification model with a PlantGuard classifier head.

    Supported architectures:
        efficientnet_b0:
            Lightweight EfficientNet baseline.

        efficientnet_b3:
            Higher-capacity EfficientNet candidate.

        resnet50:
            Standard CNN baseline for comparison.

    Args:
        model_name:
            Architecture name.
        num_classes:
            Number of output classes.
        pretrained:
            If True, initialize the backbone with ImageNet weights.

    Returns:
        PyTorch model with the final classification layer replaced.
    """
    if num_classes <= 0:
        raise ValueError(f"num_classes must be positive. Got: {num_classes}")

    if model_name == "efficientnet_b0":
        weights = models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
        model = models.efficientnet_b0(weights=weights)

        in_features = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(in_features, num_classes)

        return model

    if model_name == "efficientnet_b3":
        weights = models.EfficientNet_B3_Weights.DEFAULT if pretrained else None
        model = models.efficientnet_b3(weights=weights)

        in_features = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(in_features, num_classes)

        return model

    if model_name == "resnet50":
        weights = models.ResNet50_Weights.DEFAULT if pretrained else None
        model = models.resnet50(weights=weights)

        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)

        return model

    raise ValueError(
        f"Unsupported model name: {model_name}. "
        f"Choose from: {sorted(SUPPORTED_MODELS)}."
    )


def create_optimizer(
    model: nn.Module,
    optimizer_name: str,
    learning_rate: float,
    weight_decay: float,
) -> optim.Optimizer:
    """
    Create the optimizer used for training.

    Args:
        model:
            PyTorch model.
        optimizer_name:
            Optimizer name from config.
        learning_rate:
            Optimizer learning rate.
        weight_decay:
            L2 regularization value.

    Returns:
        PyTorch optimizer.
    """
    if learning_rate <= 0:
        raise ValueError(f"learning_rate must be positive. Got: {learning_rate}")

    if weight_decay < 0:
        raise ValueError(f"weight_decay must be non-negative. Got: {weight_decay}")

    if optimizer_name == "adamw":
        return optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
        )

    if optimizer_name == "adam":
        return optim.Adam(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
        )

    raise ValueError(
        f"Unsupported optimizer: {optimizer_name}. "
        f"Choose from: {sorted(SUPPORTED_OPTIMIZERS)}."
    )


class FocalLoss(nn.Module):
    """
    Multi-class Focal Loss.

    Focal Loss down-weights easy examples and gives relatively more weight to
    hard or misclassified examples.

    Formula:
        FL = (1 - pt)^gamma * CE

    where:
        CE:
            Standard cross-entropy loss.
        pt:
            Model probability assigned to the true class.
        gamma:
            Focusing parameter.
    """

    def __init__(
        self,
        gamma: float = 2.0,
        weight: torch.Tensor | None = None,
    ) -> None:
        """
        Initialize Focal Loss.

        Args:
            gamma:
                Focusing parameter. Higher values focus more on hard examples.
            weight:
                Optional class-weight tensor.
        """
        super().__init__()

        if gamma < 0:
            raise ValueError(f"gamma must be non-negative. Got: {gamma}")

        self.gamma = gamma
        self.weight = weight

    def forward(
        self,
        outputs: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute focal loss for one batch.

        Args:
            outputs:
                Raw model logits with shape [batch_size, num_classes].
            labels:
                Ground-truth class indices with shape [batch_size].

        Returns:
            Scalar focal loss.
        """
        cross_entropy_loss = F.cross_entropy(
            outputs,
            labels,
            weight=self.weight,
            reduction="none",
        )

        pt = torch.exp(-cross_entropy_loss)
        focal_loss = ((1.0 - pt) ** self.gamma) * cross_entropy_loss

        return focal_loss.mean()


def compute_class_weights(
    train_dataset,
    num_classes: int,
    device: torch.device,
) -> torch.Tensor:
    """
    Compute inverse-frequency class weights from the training dataset.

    Args:
        train_dataset:
            Training dataset with a samples attribute containing (path, label).
        num_classes:
            Number of classes.
        device:
            Device where the returned tensor should be placed.

    Returns:
        Class-weight tensor with shape [num_classes].
    """
    if not hasattr(train_dataset, "samples"):
        raise AttributeError(
            "train_dataset must expose a samples attribute to compute class weights."
        )

    labels = [
        int(label)
        for _, label in train_dataset.samples
    ]

    if not labels:
        raise RuntimeError("Cannot compute class weights from an empty dataset.")

    class_counts = torch.bincount(
        torch.tensor(labels),
        minlength=num_classes,
    ).float()

    class_counts = torch.clamp(class_counts, min=1.0)
    class_weights = class_counts.sum() / (num_classes * class_counts)

    return class_weights.to(device)


def create_loss_function(
    config: dict,
    train_dataset,
    num_classes: int,
    device: torch.device,
) -> nn.Module:
    """
    Create the loss function specified in config.yaml.

    Supported losses:
        cross_entropy
        weighted_cross_entropy
        focal_loss

    Args:
        config:
            Training configuration dictionary.
        train_dataset:
            Training dataset object.
        num_classes:
            Number of classes.
        device:
            CPU or CUDA device.

    Returns:
        PyTorch loss module.
    """
    loss_name = config["training"]["loss"]

    if loss_name == "cross_entropy":
        print("Using CrossEntropyLoss.")
        return nn.CrossEntropyLoss()

    if loss_name == "weighted_cross_entropy":
        class_weights = compute_class_weights(
            train_dataset=train_dataset,
            num_classes=num_classes,
            device=device,
        )

        print(f"Using Weighted CrossEntropyLoss with weights: {class_weights}")
        return nn.CrossEntropyLoss(weight=class_weights)

    if loss_name == "focal_loss":
        gamma = float(config["training"]["focal_gamma"])

        print(f"Using FocalLoss with gamma={gamma}.")
        return FocalLoss(gamma=gamma)

    raise ValueError(
        f"Unsupported loss function: {loss_name}. "
        f"Choose from: {sorted(SUPPORTED_LOSSES)}."
    )


def train_one_epoch(
    model: nn.Module,
    train_loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
    epoch: int,
) -> tuple[float, float]:
    """
    Train the model for one epoch.

    Args:
        model:
            PyTorch model.
        train_loader:
            DataLoader for training data.
        criterion:
            Loss function.
        optimizer:
            Optimizer.
        device:
            CPU or CUDA device.
        epoch:
            Current epoch number.

    Returns:
        epoch_loss:
            Average training loss.
        epoch_accuracy:
            Training accuracy.
    """
    model.train()

    running_loss = 0.0
    correct_predictions = 0
    total_samples = 0

    for batch_idx, (images, labels) in enumerate(train_loader):
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad(set_to_none=True)

        outputs = model(images)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        batch_size = images.size(0)
        running_loss += loss.item() * batch_size

        predictions = outputs.argmax(dim=1)
        correct_predictions += (predictions == labels).sum().item()
        total_samples += labels.size(0)

        if batch_idx % 100 == 0:
            print(
                f"Epoch {epoch} | "
                f"Batch {batch_idx}/{len(train_loader)} | "
                f"Loss: {loss.item():.4f}"
            )

    if total_samples == 0:
        raise RuntimeError("Training loader produced zero samples.")

    epoch_loss = running_loss / total_samples
    epoch_accuracy = correct_predictions / total_samples

    return epoch_loss, epoch_accuracy


def validate(
    model: nn.Module,
    val_loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float]:
    """
    Evaluate the model on the validation split.

    Args:
        model:
            PyTorch model.
        val_loader:
            Validation DataLoader.
        criterion:
            Loss function.
        device:
            CPU or CUDA device.

    Returns:
        epoch_loss:
            Average validation loss.
        epoch_accuracy:
            Validation accuracy.
    """
    model.eval()

    running_loss = 0.0
    correct_predictions = 0
    total_samples = 0

    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            batch_size = images.size(0)
            running_loss += loss.item() * batch_size

            predictions = outputs.argmax(dim=1)
            correct_predictions += (predictions == labels).sum().item()
            total_samples += labels.size(0)

    if total_samples == 0:
        raise RuntimeError("Validation loader produced zero samples.")

    epoch_loss = running_loss / total_samples
    epoch_accuracy = correct_predictions / total_samples

    return epoch_loss, epoch_accuracy


def flatten_config(
    config: dict,
    parent_key: str = "",
) -> dict:
    """
    Flatten nested config dictionaries for MLflow parameter logging.

    Example:
        {"training": {"epochs": 10}}

    becomes:
        {"training.epochs": 10}

    Args:
        config:
            Nested configuration dictionary.
        parent_key:
            Prefix used during recursion.

    Returns:
        Flattened dictionary.
    """
    flattened = {}

    for key, value in config.items():
        new_key = f"{parent_key}.{key}" if parent_key else str(key)

        if isinstance(value, dict):
            flattened.update(flatten_config(value, new_key))
        else:
            flattened[new_key] = value

    return flattened


def save_checkpoint(
    path: str | Path,
    model: nn.Module,
    optimizer: optim.Optimizer,
    epoch: int,
    best_val_accuracy: float,
    class_names: list[str],
    config: dict,
) -> None:
    """
    Save model, optimizer, metadata, and config to disk.

    Args:
        path:
            Destination checkpoint path.
        model:
            PyTorch model.
        optimizer:
            Optimizer.
        epoch:
            Current epoch number.
        best_val_accuracy:
            Best validation accuracy seen so far.
        class_names:
            Class names in label-index order.
        config:
            Training configuration.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "best_val_accuracy": best_val_accuracy,
        "class_names": class_names,
        "config": config,
    }

    torch.save(checkpoint, path)


def run_training(config: dict) -> Path:
    """
    Run the full PlantVillage training pipeline.

    Args:
        config:
            Parsed training configuration.

    Returns:
        Path to the best saved checkpoint.
    """
    set_seed(int(config["project"]["seed"]))

    device = get_device(config["training"]["device"])
    print(f"Using device: {device}")

    train_loader, val_loader, _, class_names = get_dataloaders(
        batch_size=config["data"]["batch_size"],
        num_workers=config["data"]["num_workers"],
    )

    num_classes = len(class_names)

    model = build_model(
        model_name=config["model"]["name"],
        num_classes=num_classes,
        pretrained=config["model"]["pretrained"],
    )

    model = model.to(device)

    criterion = create_loss_function(
        config=config,
        train_dataset=train_loader.dataset,
        num_classes=num_classes,
        device=device,
    )

    optimizer = create_optimizer(
        model=model,
        optimizer_name=config["training"]["optimizer"],
        learning_rate=config["training"]["learning_rate"],
        weight_decay=config["training"]["weight_decay"],
    )

    checkpoint_dir = PROJECT_ROOT / config["checkpoint"]["save_dir"]
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    run_name = config["mlflow"]["run_name"]

    best_model_path = checkpoint_dir / f"{run_name}_best_model.pth"
    latest_model_path = checkpoint_dir / f"{run_name}_latest_checkpoint.pth"

    best_val_accuracy = 0.0

    mlflow.set_tracking_uri(f"sqlite:///{MLFLOW_DB_PATH.as_posix()}")
    mlflow.set_experiment(config["mlflow"]["experiment_name"])

    with mlflow.start_run(run_name=run_name):
        mlflow.log_params(flatten_config(config))
        mlflow.log_param("actual_num_classes", num_classes)
        mlflow.log_param("device", str(device))

        for epoch in range(1, int(config["training"]["epochs"]) + 1):
            print(f"\nEpoch {epoch}/{config['training']['epochs']}")

            train_loss, train_accuracy = train_one_epoch(
                model=model,
                train_loader=train_loader,
                criterion=criterion,
                optimizer=optimizer,
                device=device,
                epoch=epoch,
            )

            val_loss, val_accuracy = validate(
                model=model,
                val_loader=val_loader,
                criterion=criterion,
                device=device,
            )

            print(
                f"Train Loss: {train_loss:.4f} | "
                f"Train Acc: {train_accuracy:.4f} | "
                f"Val Loss: {val_loss:.4f} | "
                f"Val Acc: {val_accuracy:.4f}"
            )

            mlflow.log_metric("train_loss", train_loss, step=epoch)
            mlflow.log_metric("train_accuracy", train_accuracy, step=epoch)
            mlflow.log_metric("val_loss", val_loss, step=epoch)
            mlflow.log_metric("val_accuracy", val_accuracy, step=epoch)

            save_checkpoint(
                path=latest_model_path,
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                best_val_accuracy=best_val_accuracy,
                class_names=class_names,
                config=config,
            )

            if val_accuracy > best_val_accuracy:
                best_val_accuracy = val_accuracy

                save_checkpoint(
                    path=best_model_path,
                    model=model,
                    optimizer=optimizer,
                    epoch=epoch,
                    best_val_accuracy=best_val_accuracy,
                    class_names=class_names,
                    config=config,
                )

                mlflow.log_artifact(
                    local_path=str(best_model_path),
                    artifact_path="checkpoints",
                )

                print(
                    "Saved new best model with validation accuracy: "
                    f"{best_val_accuracy:.4f}"
                )

        mlflow.log_metric("best_val_accuracy", best_val_accuracy)

        mlflow.log_artifact(
            local_path=str(CONFIG_PATH),
            artifact_path="config",
        )

    print("\nTraining complete.")
    print(f"Best validation accuracy: {best_val_accuracy:.4f}")
    print(f"Best model saved to: {best_model_path}")

    return best_model_path


def main() -> None:
    """
    Load config and run PlantVillage base training.
    """
    config = load_config(CONFIG_PATH)
    run_training(config)


if __name__ == "__main__":
    main()