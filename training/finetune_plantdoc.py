"""
Fine-tune PlantVillage-trained PlantGuard checkpoints on PlantDoc.

This script performs Stage B domain adaptation for PlantGuard.

Pipeline:
    1. Load one or more PlantVillage-trained best checkpoints from models/.
    2. Rebuild the original checkpoint architecture.
    3. Load PlantVillage-trained weights.
    4. Fine-tune on the PlantDoc train-adapt split.
    5. Validate on the PlantDoc val-adapt split.
    6. Save the best PlantDoc-adapted checkpoint by validation macro F1.
    7. Log parameters and metrics to MLflow.
    8. Save a fine-tuning summary CSV.

Important:
    This script does not perform final testing.

Final testing is handled separately by training/evaluate.py on:
    - PlantVillage test
    - PlantDoc held-out test
    - PlantWild_v2 test
    - FieldPlant compatible external test

The purpose of this script is domain adaptation, not final model selection.
"""

from __future__ import annotations

import gc
import sys
from pathlib import Path

import mlflow
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import f1_score

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from data.dataset import get_plantdoc_finetune_loaders  # noqa: E402
from training.evaluate import count_parameters, load_checkpoint  # noqa: E402
from training.train import (  # noqa: E402
    CONFIG_PATH,
    FocalLoss,
    build_model,
    create_optimizer,
    get_device,
    load_config,
    set_seed,
    train_one_epoch,
)


MODELS_DIR = PROJECT_ROOT / "models"
MLFLOW_DB_PATH = PROJECT_ROOT / "mlflow.db"

SOURCE_CHECKPOINT_GLOB = "*_best_model.pth"


def resolve_project_path(path_value: str | Path) -> Path:
    """
    Resolve a path that may be absolute or project-relative.

    Args:
        path_value:
            Path string from config or a Path object.

    Returns:
        Absolute Path object.
    """
    path = Path(path_value)

    if path.is_absolute():
        return path

    return PROJECT_ROOT / path


def discover_checkpoints(checkpoint_from_config: str | None) -> list[Path]:
    """
    Find PlantVillage checkpoints to fine-tune.

    Args:
        checkpoint_from_config:
            If provided, only this checkpoint is used. If None or empty, all
            root-level models/*_best_model.pth checkpoints are used.

    Returns:
        Sorted list of checkpoint paths.
    """
    if checkpoint_from_config:
        checkpoint_path = resolve_project_path(checkpoint_from_config)

        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Configured checkpoint not found: {checkpoint_path}")

        return [checkpoint_path]

    checkpoint_paths = sorted(
        checkpoint_path
        for checkpoint_path in MODELS_DIR.glob(SOURCE_CHECKPOINT_GLOB)
        if checkpoint_path.parent == MODELS_DIR
        and "plantdoc_finetuned" not in checkpoint_path.name
        and "plantwild_expanded" not in checkpoint_path.name
    )

    return checkpoint_paths


def get_checkpoint_identity(checkpoint: dict) -> tuple[str, str, str]:
    """
    Read run/model/loss identity from a PlantVillage checkpoint.

    Args:
        checkpoint:
            Loaded PlantVillage checkpoint dictionary.

    Returns:
        run_name:
            Original MLflow run name.
        model_name:
            Architecture name, e.g. efficientnet_b3.
        loss_name:
            Training loss name, e.g. weighted_cross_entropy.
    """
    if "config" not in checkpoint:
        raise KeyError("Checkpoint is missing config metadata.")

    source_config = checkpoint["config"]

    run_name = source_config["mlflow"]["run_name"]
    model_name = source_config["model"]["name"]
    loss_name = source_config["training"]["loss"]

    return run_name, model_name, loss_name


def get_checkpoint_focal_gamma(
    checkpoint: dict,
    default: float = 2.0,
) -> float:
    """
    Read focal-loss gamma from the source checkpoint config.

    Args:
        checkpoint:
            Loaded checkpoint dictionary.
        default:
            Fallback gamma if missing from config.

    Returns:
        Focal-loss gamma.
    """
    source_config = checkpoint.get("config", {})
    training_config = source_config.get("training", {})

    return float(training_config.get("focal_gamma", default))


def get_subset_labels(subset: torch.utils.data.Subset) -> list[int]:
    """
    Extract labels from a torch.utils.data.Subset.

    PlantDoc fine-tuning uses a Subset of PlantDocEvaluationDataset. The base
    dataset stores samples as:

        (image_path, label, plantdoc_class_name, mapped_plantguard_label)

    Args:
        subset:
            PlantDoc train-adapt subset.

    Returns:
        Integer labels from the subset.
    """
    labels = []

    for original_index in subset.indices:
        sample = subset.dataset.samples[original_index]
        labels.append(int(sample[1]))

    return labels


def compute_subset_class_weights(
    subset: torch.utils.data.Subset,
    num_classes: int,
    device: torch.device,
) -> torch.Tensor:
    """
    Compute inverse-frequency class weights from PlantDoc train-adapt data.

    Classes absent from PlantDoc train-adapt receive weight 0. This is safe
    because absent classes do not appear as target labels during fine-tuning.

    Args:
        subset:
            PlantDoc train-adapt subset.
        num_classes:
            Total number of model output classes.
        device:
            CPU or CUDA device.

    Returns:
        Class-weight tensor with shape [num_classes].
    """
    labels = get_subset_labels(subset)

    if not labels:
        raise RuntimeError("Cannot compute class weights from an empty subset.")

    labels_tensor = torch.tensor(labels, dtype=torch.long)

    counts = torch.bincount(labels_tensor, minlength=num_classes).float()
    present_mask = counts > 0

    weights = torch.zeros(num_classes, dtype=torch.float32)

    present_counts = counts[present_mask]
    num_present_classes = present_mask.sum()
    total_present_samples = present_counts.sum()

    weights[present_mask] = total_present_samples / (
        num_present_classes * present_counts
    )

    return weights.to(device)


def create_finetune_loss(
    loss_name: str,
    train_subset: torch.utils.data.Subset,
    num_classes: int,
    device: torch.device,
    focal_gamma: float,
) -> nn.Module:
    """
    Create the loss function for PlantDoc fine-tuning.

    The fine-tuning loss follows the source checkpoint's original training loss:
        - cross_entropy
        - weighted_cross_entropy
        - focal_loss

    Args:
        loss_name:
            Loss name from original checkpoint config.
        train_subset:
            PlantDoc train-adapt subset.
        num_classes:
            Total number of model output classes.
        device:
            CPU or CUDA device.
        focal_gamma:
            Gamma value for focal loss.

    Returns:
        PyTorch loss module.
    """
    if loss_name == "cross_entropy":
        print("Using CrossEntropyLoss.")
        return nn.CrossEntropyLoss()

    if loss_name == "weighted_cross_entropy":
        weights = compute_subset_class_weights(
            subset=train_subset,
            num_classes=num_classes,
            device=device,
        )

        print("Using PlantDoc weighted CrossEntropyLoss.")
        print(f"Non-zero class weights: {(weights > 0).sum().item()}")
        print(f"Max class weight: {weights.max().item():.4f}")

        return nn.CrossEntropyLoss(weight=weights)

    if loss_name == "focal_loss":
        print(f"Using FocalLoss with gamma={focal_gamma}.")
        return FocalLoss(gamma=focal_gamma)

    raise ValueError(f"Unsupported fine-tuning loss: {loss_name}")


def freeze_backbone(model: nn.Module, model_name: str) -> None:
    """
    Freeze the feature extractor and train only the classifier head.

    This is Stage 1 of PlantDoc fine-tuning. It adapts the final classifier
    before changing the full feature extractor.

    Args:
        model:
            PyTorch model.
        model_name:
            Architecture name.
    """
    for parameter in model.parameters():
        parameter.requires_grad = False

    if model_name.startswith("efficientnet"):
        for parameter in model.classifier.parameters():
            parameter.requires_grad = True
        return

    if model_name == "resnet50":
        for parameter in model.fc.parameters():
            parameter.requires_grad = True
        return

    raise ValueError(f"Unsupported model for freezing: {model_name}")


def unfreeze_model(model: nn.Module) -> None:
    """
    Unfreeze every parameter for full-model low-learning-rate fine-tuning.
    """
    for parameter in model.parameters():
        parameter.requires_grad = True


@torch.no_grad()
def validate_with_f1(
    model: nn.Module,
    val_loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> dict[str, float]:
    """
    Evaluate a model on the PlantDoc val-adapt split.

    Args:
        model:
            PyTorch model.
        val_loader:
            PlantDoc validation DataLoader.
        criterion:
            Loss function.
        device:
            CPU or CUDA device.

    Returns:
        Dictionary with validation loss, accuracy, macro F1, and weighted F1.
    """
    model.eval()

    total_loss = 0.0
    correct_predictions = 0
    total_samples = 0

    all_labels = []
    all_predictions = []

    for images, labels in val_loader:
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)

        predictions = outputs.argmax(dim=1)
        batch_size = labels.size(0)

        total_loss += loss.item() * batch_size
        correct_predictions += (predictions == labels).sum().item()
        total_samples += batch_size

        all_labels.extend(labels.cpu().tolist())
        all_predictions.extend(predictions.cpu().tolist())

    if total_samples == 0:
        raise RuntimeError("PlantDoc validation loader produced zero samples.")

    return {
        "val_loss": total_loss / total_samples,
        "val_accuracy": correct_predictions / total_samples,
        "val_macro_f1": f1_score(
            all_labels,
            all_predictions,
            average="macro",
            zero_division=0,
        ),
        "val_weighted_f1": f1_score(
            all_labels,
            all_predictions,
            average="weighted",
            zero_division=0,
        ),
    }


def log_epoch_metrics(
    train_loss: float,
    train_accuracy: float,
    val_metrics: dict[str, float],
    epoch: int,
) -> None:
    """
    Print and log one epoch of fine-tuning metrics.

    Args:
        train_loss:
            Average training loss.
        train_accuracy:
            Training accuracy.
        val_metrics:
            Dictionary returned by validate_with_f1.
        epoch:
            Global fine-tuning epoch number.
    """
    print(
        f"Epoch {epoch} | "
        f"train_loss={train_loss:.4f} "
        f"train_acc={train_accuracy:.4f} | "
        f"val_loss={val_metrics['val_loss']:.4f} "
        f"val_acc={val_metrics['val_accuracy']:.4f} "
        f"val_macro_f1={val_metrics['val_macro_f1']:.4f} "
        f"val_weighted_f1={val_metrics['val_weighted_f1']:.4f}"
    )

    mlflow.log_metric("train_loss", train_loss, step=epoch)
    mlflow.log_metric("train_accuracy", train_accuracy, step=epoch)

    for metric_name, metric_value in val_metrics.items():
        mlflow.log_metric(metric_name, metric_value, step=epoch)


def save_finetuned_checkpoint(
    output_path: Path,
    model: nn.Module,
    optimizer: optim.Optimizer,
    source_checkpoint: dict,
    source_checkpoint_path: Path,
    run_name: str,
    model_name: str,
    loss_name: str,
    class_names: list[str],
    best_metrics: dict[str, float],
    epoch: int,
    stage: str,
    finetune_config: dict,
) -> None:
    """
    Save the best PlantDoc-adapted checkpoint.

    Args:
        output_path:
            Destination checkpoint path.
        model:
            Fine-tuned PyTorch model.
        optimizer:
            Optimizer state at save time.
        source_checkpoint:
            Original PlantVillage checkpoint dictionary.
        source_checkpoint_path:
            Original PlantVillage checkpoint path.
        run_name:
            Original training run name.
        model_name:
            Architecture name.
        loss_name:
            Original loss function name.
        class_names:
            Class names in label-index order.
        best_metrics:
            Current best validation metrics.
        epoch:
            Fine-tuning epoch when this checkpoint was saved.
        stage:
            "head" or "full".
        finetune_config:
            PlantDoc fine-tuning config dictionary.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    checkpoint = {
        "source_checkpoint": source_checkpoint_path.name,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "class_names": class_names,
        "model_name": model_name,
        "loss_name": loss_name,
        "run_name": f"{run_name}_plantdoc_finetuned",
        "epoch": epoch,
        "stage": stage,
        "best_val_macro_f1": best_metrics["val_macro_f1"],
        "best_val_accuracy": best_metrics["val_accuracy"],
        "source_config": source_checkpoint.get("config", {}),
        "finetune_config": dict(finetune_config),
    }

    torch.save(checkpoint, output_path)


def maybe_save_best(
    model: nn.Module,
    optimizer: optim.Optimizer,
    output_path: Path,
    source_checkpoint: dict,
    source_checkpoint_path: Path,
    run_name: str,
    model_name: str,
    loss_name: str,
    class_names: list[str],
    current_metrics: dict[str, float],
    best_metrics: dict[str, float],
    epoch: int,
    stage: str,
    finetune_config: dict,
) -> tuple[dict[str, float], bool]:
    """
    Save a checkpoint if current validation macro F1 is the best so far.

    Args:
        current_metrics:
            Current validation metrics.
        best_metrics:
            Best metrics seen so far.

    Returns:
        updated_best_metrics:
            Best metrics after comparison.
        improved:
            True if a new best checkpoint was saved.
    """
    if current_metrics["val_macro_f1"] <= best_metrics["val_macro_f1"]:
        return best_metrics, False

    save_finetuned_checkpoint(
        output_path=output_path,
        model=model,
        optimizer=optimizer,
        source_checkpoint=source_checkpoint,
        source_checkpoint_path=source_checkpoint_path,
        run_name=run_name,
        model_name=model_name,
        loss_name=loss_name,
        class_names=class_names,
        best_metrics=current_metrics,
        epoch=epoch,
        stage=stage,
        finetune_config=finetune_config,
    )

    print(
        f"Saved new best: {output_path.name} | "
        f"val_macro_f1={current_metrics['val_macro_f1']:.4f} | "
        f"val_acc={current_metrics['val_accuracy']:.4f}"
    )

    return current_metrics, True


def load_model_for_finetuning(
    checkpoint_path: Path,
    class_names: list[str],
    device: torch.device,
) -> tuple[nn.Module, dict, str, str, str]:
    """
    Load a PlantVillage-trained checkpoint for PlantDoc fine-tuning.

    Args:
        checkpoint_path:
            Path to the source PlantVillage checkpoint.
        class_names:
            Current class-name list in label-index order.
        device:
            CPU or CUDA device.

    Returns:
        model:
            Loaded PyTorch model.
        checkpoint:
            Loaded checkpoint dictionary.
        run_name:
            Original MLflow run name.
        model_name:
            Architecture name.
        loss_name:
            Original loss function name.
    """
    checkpoint = load_checkpoint(checkpoint_path)

    required_keys = {"model_state_dict", "class_names", "config"}
    missing_keys = required_keys - set(checkpoint.keys())

    if missing_keys:
        raise KeyError(
            f"Checkpoint {checkpoint_path.name} is missing keys: "
            f"{sorted(missing_keys)}"
        )

    if list(checkpoint["class_names"]) != list(class_names):
        raise ValueError(
            f"Class-name mismatch for {checkpoint_path.name}. "
            "Checkpoint class order does not match current class order."
        )

    run_name, model_name, loss_name = get_checkpoint_identity(checkpoint)

    model = build_model(
        model_name=model_name,
        num_classes=len(class_names),
        pretrained=False,
    )

    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)

    return model, checkpoint, run_name, model_name, loss_name


def create_stage_optimizer(
    model: nn.Module,
    stage_name: str,
    learning_rate: float,
    weight_decay: float,
    optimizer_name: str,
) -> optim.Optimizer:
    """
    Create an optimizer for one fine-tuning stage.

    Args:
        model:
            PyTorch model.
        stage_name:
            "head" or "full".
        learning_rate:
            Stage learning rate.
        weight_decay:
            Stage weight decay.
        optimizer_name:
            Optimizer name from config.

    Returns:
        PyTorch optimizer.
    """
    if learning_rate <= 0:
        raise ValueError(f"learning_rate must be positive. Got: {learning_rate}")

    if weight_decay < 0:
        raise ValueError(f"weight_decay must be non-negative. Got: {weight_decay}")

    if stage_name == "head":
        parameters = filter(lambda parameter: parameter.requires_grad, model.parameters())

        if optimizer_name == "adamw":
            return optim.AdamW(
                parameters,
                lr=learning_rate,
                weight_decay=weight_decay,
            )

        if optimizer_name == "adam":
            return optim.Adam(
                parameters,
                lr=learning_rate,
                weight_decay=weight_decay,
            )

        raise ValueError(f"Unsupported optimizer for head stage: {optimizer_name}")

    if stage_name == "full":
        return create_optimizer(
            model=model,
            optimizer_name=optimizer_name,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
        )

    raise ValueError(f"Unsupported fine-tuning stage: {stage_name}")


def run_training_stage(
    model: nn.Module,
    model_name: str,
    train_loader: torch.utils.data.DataLoader,
    val_loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    stage_name: str,
    epochs: int,
    learning_rate: float,
    weight_decay: float,
    optimizer_name: str,
    start_epoch: int,
    best_metrics: dict[str, float],
    save_context: dict,
    device: torch.device,
) -> tuple[int, dict[str, float], int | None, str | None]:
    """
    Run one PlantDoc fine-tuning stage.

    Stages:
        head:
            Freeze the backbone and train only the classifier head.

        full:
            Unfreeze all layers and fine-tune the full model with lower LR.

    Args:
        model:
            PyTorch model.
        model_name:
            Architecture name.
        train_loader:
            PlantDoc train-adapt DataLoader.
        val_loader:
            PlantDoc val-adapt DataLoader.
        criterion:
            Loss function.
        stage_name:
            "head" or "full".
        epochs:
            Number of epochs for this stage.
        learning_rate:
            Learning rate for this stage.
        weight_decay:
            Weight decay for this stage.
        optimizer_name:
            Optimizer name.
        start_epoch:
            Global epoch number before this stage starts.
        best_metrics:
            Best validation metrics so far.
        save_context:
            Static arguments passed to maybe_save_best.
        device:
            CPU or CUDA device.

    Returns:
        current_epoch:
            Last global epoch number after this stage.
        best_metrics:
            Updated best metrics.
        stage_best_epoch:
            Epoch where this stage produced a new best, or None.
        stage_best_stage:
            Stage name if this stage produced a new best, otherwise None.
    """
    if epochs <= 0:
        return start_epoch, best_metrics, None, None

    if stage_name == "head":
        print("\nStage 1: frozen backbone, train classifier head")
        freeze_backbone(model, model_name)

    elif stage_name == "full":
        print("\nStage 2: full-model low-LR fine-tuning")
        unfreeze_model(model)

    else:
        raise ValueError(f"Unsupported fine-tuning stage: {stage_name}")

    optimizer = create_stage_optimizer(
        model=model,
        stage_name=stage_name,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        optimizer_name=optimizer_name,
    )

    stage_best_epoch = None
    stage_best_stage = None
    current_epoch = start_epoch

    for _ in range(epochs):
        current_epoch += 1

        train_loss, train_accuracy = train_one_epoch(
            model=model,
            train_loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            epoch=current_epoch,
        )

        val_metrics = validate_with_f1(
            model=model,
            val_loader=val_loader,
            criterion=criterion,
            device=device,
        )

        log_epoch_metrics(
            train_loss=train_loss,
            train_accuracy=train_accuracy,
            val_metrics=val_metrics,
            epoch=current_epoch,
        )

        best_metrics, improved = maybe_save_best(
            model=model,
            optimizer=optimizer,
            current_metrics=val_metrics,
            best_metrics=best_metrics,
            epoch=current_epoch,
            stage=stage_name,
            **save_context,
        )

        if improved:
            stage_best_epoch = current_epoch
            stage_best_stage = stage_name

    return current_epoch, best_metrics, stage_best_epoch, stage_best_stage


def finetune_checkpoint(
    checkpoint_path: Path,
    class_names: list[str],
    train_loader: torch.utils.data.DataLoader,
    val_loader: torch.utils.data.DataLoader,
    train_subset: torch.utils.data.Subset,
    finetune_config: dict,
    device: torch.device,
) -> dict:
    """
    Fine-tune one PlantVillage checkpoint on PlantDoc.

    Args:
        checkpoint_path:
            Source PlantVillage best checkpoint.
        class_names:
            Class names in label-index order.
        train_loader:
            PlantDoc train-adapt DataLoader.
        val_loader:
            PlantDoc val-adapt DataLoader.
        train_subset:
            PlantDoc train-adapt subset used for class-weight computation.
        finetune_config:
            PlantDoc fine-tuning settings from config.yaml.
        device:
            CPU or CUDA device.

    Returns:
        Summary row for plantdoc_finetuning_summary.csv.
    """
    model, checkpoint, run_name, model_name, loss_name = load_model_for_finetuning(
        checkpoint_path=checkpoint_path,
        class_names=class_names,
        device=device,
    )

    total_parameters, trainable_parameters = count_parameters(model)

    output_dir = resolve_project_path(finetune_config["finetuned_model_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / f"{run_name}_plantdoc_finetuned_best_model.pth"

    focal_gamma = get_checkpoint_focal_gamma(checkpoint)

    criterion = create_finetune_loss(
        loss_name=loss_name,
        train_subset=train_subset,
        num_classes=len(class_names),
        device=device,
        focal_gamma=focal_gamma,
    )

    print("\n" + "=" * 80)
    print(f"Fine-tuning: {run_name}")
    print(f"Checkpoint:  {checkpoint_path.name}")
    print(f"Model:       {model_name}")
    print(f"Loss:        {loss_name}")
    print(f"Parameters:  {total_parameters:,}")
    print("=" * 80)

    best_metrics = {
        "val_loss": float("inf"),
        "val_accuracy": -1.0,
        "val_macro_f1": -1.0,
        "val_weighted_f1": -1.0,
    }

    save_context = {
        "output_path": output_path,
        "source_checkpoint": checkpoint,
        "source_checkpoint_path": checkpoint_path,
        "run_name": run_name,
        "model_name": model_name,
        "loss_name": loss_name,
        "class_names": class_names,
        "finetune_config": finetune_config,
    }

    best_epoch = None
    best_stage = None

    with mlflow.start_run(run_name=f"{run_name}_plantdoc_finetune"):
        mlflow.log_param("source_checkpoint", checkpoint_path.name)
        mlflow.log_param("model_name", model_name)
        mlflow.log_param("loss_name", loss_name)
        mlflow.log_param("total_parameters", total_parameters)
        mlflow.log_param("trainable_parameters_initial", trainable_parameters)
        mlflow.log_param("focal_gamma", focal_gamma)

        for key, value in finetune_config.items():
            mlflow.log_param(f"plantdoc_finetune.{key}", str(value))

        epoch = 0

        epoch, best_metrics, stage_best_epoch, stage_best_stage = run_training_stage(
            model=model,
            model_name=model_name,
            train_loader=train_loader,
            val_loader=val_loader,
            criterion=criterion,
            stage_name="head",
            epochs=int(finetune_config["head_epochs"]),
            learning_rate=float(finetune_config["lr_head"]),
            weight_decay=float(finetune_config["weight_decay"]),
            optimizer_name=finetune_config["optimizer"],
            start_epoch=epoch,
            best_metrics=best_metrics,
            save_context=save_context,
            device=device,
        )

        if stage_best_epoch is not None:
            best_epoch = stage_best_epoch
            best_stage = stage_best_stage

        epoch, best_metrics, stage_best_epoch, stage_best_stage = run_training_stage(
            model=model,
            model_name=model_name,
            train_loader=train_loader,
            val_loader=val_loader,
            criterion=criterion,
            stage_name="full",
            epochs=int(finetune_config["full_epochs"]),
            learning_rate=float(finetune_config["lr_full"]),
            weight_decay=float(finetune_config["weight_decay"]),
            optimizer_name=finetune_config["optimizer"],
            start_epoch=epoch,
            best_metrics=best_metrics,
            save_context=save_context,
            device=device,
        )

        if stage_best_epoch is not None:
            best_epoch = stage_best_epoch
            best_stage = stage_best_stage

        mlflow.log_metric("best_val_macro_f1", best_metrics["val_macro_f1"])
        mlflow.log_metric("best_val_accuracy", best_metrics["val_accuracy"])
        mlflow.log_param("best_epoch", best_epoch)
        mlflow.log_param("best_stage", best_stage)

        if output_path.exists():
            mlflow.log_artifact(
                str(output_path),
                artifact_path="checkpoints",
            )

    summary_row = {
        "source_checkpoint": checkpoint_path.name,
        "run_name": run_name,
        "model_name": model_name,
        "loss_name": loss_name,
        "best_val_macro_f1": best_metrics["val_macro_f1"],
        "best_val_accuracy": best_metrics["val_accuracy"],
        "best_epoch": best_epoch,
        "best_stage": best_stage,
        "finetuned_checkpoint": str(output_path.relative_to(PROJECT_ROOT)),
    }

    del model
    gc.collect()

    if device.type == "cuda":
        torch.cuda.empty_cache()

    return summary_row


def main() -> None:
    """
    Run PlantDoc fine-tuning for one checkpoint or all source checkpoints.

    Configuration is read from:

        training/config.yaml -> plantdoc_finetuning
    """
    config = load_config(CONFIG_PATH)
    finetune_config = config["plantdoc_finetuning"]

    set_seed(int(config["project"]["seed"]))

    device = get_device(config["training"]["device"])
    print(f"Using device: {device}")

    output_dir = resolve_project_path(finetune_config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    mlflow.set_tracking_uri(f"sqlite:///{MLFLOW_DB_PATH.as_posix()}")
    mlflow.set_experiment(finetune_config["experiment_name"])

    checkpoint_paths = discover_checkpoints(finetune_config.get("checkpoint"))

    if not checkpoint_paths:
        raise FileNotFoundError("No checkpoints found for PlantDoc fine-tuning.")

    print("\nCheckpoints to fine-tune:")
    for checkpoint_path in checkpoint_paths:
        print(f"- {checkpoint_path.name}")

    first_checkpoint = load_checkpoint(checkpoint_paths[0])
    class_names = list(first_checkpoint["class_names"])

    (
        train_loader,
        val_loader,
        _test_loader,
        train_subset,
        val_subset,
        test_dataset,
    ) = get_plantdoc_finetune_loaders(
        class_names=class_names,
        batch_size=int(finetune_config["batch_size"]),
        num_workers=int(finetune_config["num_workers"]),
        val_ratio=float(finetune_config["val_ratio"]),
        seed=int(config["project"]["seed"]),
    )

    print("\nPlantDoc fine-tuning data:")
    print(f"Train-adapt samples: {len(train_subset)}")
    print(f"Val-adapt samples:   {len(val_subset)}")
    print(f"Held-out test:        {len(test_dataset)}")

    rows = []

    for checkpoint_path in checkpoint_paths:
        rows.append(
            finetune_checkpoint(
                checkpoint_path=checkpoint_path,
                class_names=class_names,
                train_loader=train_loader,
                val_loader=val_loader,
                train_subset=train_subset,
                finetune_config=finetune_config,
                device=device,
            )
        )

    summary_df = pd.DataFrame(rows).sort_values(
        by="best_val_macro_f1",
        ascending=False,
    )

    summary_path = output_dir / "plantdoc_finetuning_summary.csv"
    summary_df.to_csv(summary_path, index=False)

    mlflow.log_artifact(
        str(summary_path),
        artifact_path="plantdoc_finetuning_summary",
    )

    print("\nPlantDoc fine-tuning complete.")
    print(f"Summary saved to: {summary_path}")
    print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()