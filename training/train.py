"""
Training script for the PlantGuard project.

This script loads training settings from config.yaml, creates the model,
trains it on the PlantVillage training split, validates after each epoch,
saves the best checkpoint, and logs parameters/metrics/artifacts to MLflow.
"""

from pathlib import Path
import random
import sys
import torch.nn.functional as F
import mlflow
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import yaml
from torchvision import models

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / 'training' / 'config.yaml'

sys.path.append(str(PROJECT_ROOT))

from data.dataset import get_dataloaders  # noqa: E402


def load_config(config_path):
    """
    Load training configuration from a YAML file.

    Args:
        config_path: Path to config.yaml.

    Returns:
        Dictionary containing project, data, model, training, checkpoint,
        and MLflow settings.
    """
    with open(config_path,"r") as file:
        config = yaml.safe_load(file)
    return config


def set_seed(seed):
    """
    Set random seeds for reproducible training behavior.

    Args:
        seed: Integer seed value.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device(device_name):
    """
    Select the training device.

    Args:
        device_name: "auto", "cuda", or "cpu".

    Returns:
        torch.device object.
    """
    if device_name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    return torch.device(device_name)


def build_model(model_name, num_classes, pretrained=True):
    """
    Build and return the image classification model.

    Args:
        model_name: Name of the model architecture.
        num_classes: Number of output classes.
        pretrained: Whether to use ImageNet-pretrained weights.

    Returns:
        PyTorch model with a replaced classifier head.
    """
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
    
    raise ValueError(f"Unsupported model name: {model_name}")


def create_optimizer(model, optimizer_name, learning_rate, weight_decay):
    """
    Create the optimizer used for training.

    Args:
        model: PyTorch model.
        optimizer_name: Optimizer name from config.
        learning_rate: Learning rate.
        weight_decay: L2 regularization value.

    Returns:
        PyTorch optimizer.
    """
    if optimizer_name == "adamw":
        return optim.AdamW(
            model.parameters(),
            lr = learning_rate,
            weight_decay = weight_decay,
        )
    
    if optimizer_name == "adam":
        return optim.Adam(
            model.parameters(),
            lr = learning_rate,
            weight_decay = weight_decay
        )
    
    raise ValueError(f"Unsupported optimizer: {optimizer_name}")


class FocalLoss(nn.Module):
    """
    Focal Loss for multi-class classification.

    Focal Loss reduces the contribution of easy examples and focuses training
    more on hard or misclassified examples.

    Formula:
        FL = (1 - pt)^gamma * CE

    Args:
        gamma: Focusing parameter. Higher values focus more on hard examples.
        weight: Optional class weight tensor.
    """
    def __init__(self, gamma=2.0, weight=None):
        super().__init__()
        self.gamma = gamma
        self.weight = weight
    
    def forward(self,outputs,labels):
        """
        Compute focal loss.

        Args:
            outputs: Raw model logits with shape [batch_size, num_classes].
            labels: Ground-truth class labels with shape [batch_size].

        Returns:
            Scalar focal loss.
        """
        cross_entropy_loss = F.cross_entropy(
            outputs,
            labels,
            weight = self.weight,
            reduction = "none",
        )

        pt = torch.exp(-cross_entropy_loss)
        focal_loss = ((1-pt)**self.gamma)*cross_entropy_loss

        return focal_loss.mean()


def compute_class_weights(train_dataset, num_classes, device):
    """
    Compute inverse-frequency class weights from the training dataset.

    Args:
        train_dataset: Training Dataset object containing samples.
        num_classes: Number of classes.
        device: CPU or CUDA device.

    Returns:
        Tensor of class weights with shape [num_classes].
    """
    labels = [
        label
        for _, label in train_dataset.samples
    ]

    class_counts = torch.bincount(
        torch.tensor(labels),
        minlength=num_classes,
    ).float()

    class_counts = torch.clamp(class_counts, min = 1.0)

    class_weights = class_counts.sum() / (num_classes * class_counts)

    return class_weights.to(device)


def create_loss_function(config, train_dataset, num_classes, device):
    """
    Create the loss function based on config.yaml.

    Supported losses:
        - cross_entropy
        - weighted_cross_entropy
        - focal_loss

    Args:
        config: Training configuration dictionary.
        train_dataset: Training Dataset object.
        num_classes: Number of classes.
        device: CPU or CUDA device.

    Returns:
        PyTorch loss function.
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

        print("Using Weighted CrossEntropyLoss with weight: {class_weights}")
        return nn.CrossEntropyLoss(weight = class_weights)
    
    if loss_name == "focal_loss":
        gamma = config["training"]["focal_gamma"]

        print("Using FocalLoss with gamma: {gamma}.")
        return FocalLoss(gamma=gamma)
    
    raise ValueError(f"Unsupported loss function: {loss_name}")



def train_one_epoch(model, train_loader, criterion, optimizer, device, epoch):
    """
    Train the model for one epoch.

    Args:
        model: PyTorch model.
        train_loader: DataLoader for training data.
        criterion: Loss function.
        optimizer: Optimizer.
        device: CPU or CUDA device.
        epoch: Current epoch number.

    Returns:
        Average training loss and training accuracy.
    """
    model.train()

    running_loss = 0.0
    correct_predictions = 0
    total_samples = 0

    for batch_idx, (images, labels) in enumerate(train_loader):
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        batch_size = images.size(0)
        running_loss+=loss.item() * batch_size

        predictions = outputs.argmax(dim=1)
        correct_predictions += (predictions == labels).sum().item()
        total_samples+=labels.size(0)

        if batch_idx % 100 == 0:
            print(
                f"Epoch {epoch} | "
                f"Batch {batch_idx}/{len(train_loader)} | "
                f"Loss: {loss.item():.4f}"
            )

    epoch_loss = running_loss/total_samples
    epoch_accuracy = correct_predictions/total_samples

    return epoch_loss, epoch_accuracy


def validate(model, val_loader, criterion, device):
    """
    Evaluate the model on the validation split.

    Args:
        model: PyTorch model.
        val_loader: DataLoader for validation data.
        criterion: Loss function.
        device: CPU or CUDA device.

    Returns:
        Average validation loss and validation accuracy.
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
            correct_predictions+=(predictions == labels).sum().item()
            total_samples += labels.size(0)
    
    epoch_loss = running_loss / total_samples
    epoch_accuracy = correct_predictions / total_samples

    return epoch_loss, epoch_accuracy


def flatten_config(config, parent_key=""):
    """
    Flatten nested config dictionaries for MLflow parameter logging.

    Args:
        config: Nested configuration dictionary.
        parent_key: Prefix used during recursion.

    Returns:
        Flattened dictionary.
    """
    flattened = {}
    
    for key, value in config.items():
        new_key = f"{parent_key}.{key}" if parent_key else key

        if isinstance(value, dict):
            flattened.update(flatten_config(value, new_key))
        else:
            flattened[new_key] = value

    return flattened


def save_checkpoint(path, model, optimizer, epoch, best_val_accuracy, class_names, config):
    """
    Save model and optimizer state to disk.

    Args:
        path: Where to save the checkpoint.
        model: PyTorch model.
        optimizer: Optimizer.
        epoch: Current epoch.
        best_val_accuracy: Best validation accuracy so far.
        class_names: List of class names.
        config: Training configuration.
    """
    checkpoint = {
        "epoch" : epoch,
        "model_state_dict" : model.state_dict(),
        "optimizer_state_dict" : optimizer.state_dict(),
        "best_val_accuracy" : best_val_accuracy,
        "class_names" : class_names,
        "config" : config,
    }

    torch.save(checkpoint, path)


def main():
    """
    Run the full training pipeline
    """
    config = load_config(CONFIG_PATH)

    set_seed(config["project"]["seed"])

    device = get_device(config["training"]["device"])
    print(f"Using device: {device}")

    train_loader, val_loader, _, class_names = get_dataloaders(
        batch_size = config["data"]["batch_size"],
        num_workers = config["data"]["num_workers"],
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
        model = model,
        optimizer_name = config["training"]["optimizer"],
        learning_rate = config["training"]["learning_rate"],
        weight_decay = config["training"]["weight_decay"],
    )

    checkpoint_dir = PROJECT_ROOT / config["checkpoint"]["save_dir"]
    checkpoint_dir.mkdir(parents = True, exist_ok = True)

    run_name = config["mlflow"]["run_name"]

    best_model_path = checkpoint_dir / f"{run_name}_best_model.pth"
    latest_model_path = checkpoint_dir / f"{run_name}_latest_checkpoint.pth"

    best_val_accuracy = 0.0

    mlflow_db_path = PROJECT_ROOT / "mlflow.db"
    mlflow.set_tracking_uri(f"sqlite:///{mlflow_db_path.as_posix()}")
    mlflow.set_experiment(config["mlflow"]["experiment_name"])

    with mlflow.start_run(run_name = config["mlflow"]["run_name"]):
        mlflow.log_params(flatten_config(config))
        mlflow.log_param("actual_num_classes", num_classes)
        mlflow.log_param("device", str(device))

        for epoch in range(1,config["training"]["epochs"]+1):
            print(f"\nEpoch {epoch}/{config['training']['epochs']}")

            train_loss, train_accuracy = train_one_epoch(
                model = model,
                train_loader = train_loader,
                criterion = criterion,
                optimizer = optimizer,
                device = device,
                epoch = epoch,
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

                print(f"Saved new best model with val accuracy: {best_val_accuracy:.4f}")
        
        mlflow.log_metric("best_val_accuracy", best_val_accuracy)
        mlflow.log_artifact(
            local_path=str(CONFIG_PATH),
            artifact_path="config",
        )
    print("\nTraining complete.")
    print(f"Best validation accuracy: {best_val_accuracy:.4f}")
    print(f"Best model saved to: {best_model_path}")

if __name__ == "__main__":
    main() 
