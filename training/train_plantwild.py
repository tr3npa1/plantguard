"""
Train expanded PlantGuard models on PlantWild_v2 with replay.

This script performs Stage C training for PlantGuard.

It starts from the best PlantDoc-fine-tuned 38-class checkpoints and expands
them into 132-class PlantGuard checkpoints.

Training mix:
    80% PlantWild_v2
    12% PlantDoc train-adapt replay
    8% PlantVillage replay

Model expansion:
    PlantDoc-fine-tuned checkpoints have 38 output classes.
    PlantWild-expanded models have 132 output classes.

Expansion rule:
    - Build the same architecture with 132 output classes.
    - Load all compatible backbone tensors directly.
    - Copy old classifier rows 0-37 into the new classifier head.
    - Leave new classifier rows 38-131 randomly initialized.

Model selection:
    Save the best checkpoint by weighted validation score:

        0.80 * PlantWild macro F1
      + 0.12 * PlantDoc macro F1
      + 0.08 * PlantVillage macro F1

This file is the training entry point for the final expanded label-space stage.
Final testing is handled separately by training/evaluate.py.
"""

from __future__ import annotations

import gc
import sys
from pathlib import Path

import mlflow
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import f1_score

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from data.dataset import get_expanded_training_loaders  # noqa: E402
from training.evaluate import count_parameters, load_checkpoint  # noqa: E402
from training.finetune_plantdoc import (  # noqa: E402
    create_stage_optimizer,
    freeze_backbone,
    unfreeze_model,
)
from training.train import (  # noqa: E402
    CONFIG_PATH,
    FocalLoss,
    build_model,
    get_device,
    load_config,
    set_seed,
    train_one_epoch,
)


SOURCE_CHECKPOINT_DIR = PROJECT_ROOT / "models" / "plantdoc_finetuned"
MLFLOW_DB_PATH = PROJECT_ROOT / "mlflow.db"

SOURCE_CHECKPOINT_GLOB = "*_plantdoc_finetuned_best_model.pth"


def get_plantwild_training_config(config: dict) -> dict:
    """
    Load PlantWild training settings from config.yaml.

    Args:
        config:
            Full project configuration dictionary.

    Returns:
        PlantWild training configuration dictionary.

    Raises:
        KeyError:
            If the required plantwild_training section is missing.
    """
    if "plantwild_training" not in config:
        raise KeyError(
            "Missing 'plantwild_training' section in training/config.yaml."
        )

    return config["plantwild_training"]


def resolve_project_path(path_value: str | Path) -> Path:
    """
    Resolve a path that may be absolute or project-relative.

    Args:
        path_value:
            Path from config or a Path object.

    Returns:
        Absolute Path object.
    """
    path = Path(path_value)

    if path.is_absolute():
        return path

    return PROJECT_ROOT / path


def discover_source_checkpoints(checkpoint_from_config: str | None) -> list[Path]:
    """
    Find PlantDoc-fine-tuned checkpoints to expand with PlantWild training.

    Args:
        checkpoint_from_config:
            Optional single checkpoint path from config. If None or empty, all
            PlantDoc-fine-tuned best checkpoints are discovered automatically.

    Returns:
        Sorted list of checkpoint paths.
    """
    if checkpoint_from_config:
        checkpoint_path = resolve_project_path(checkpoint_from_config)

        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Configured checkpoint not found: {checkpoint_path}")

        return [checkpoint_path]

    checkpoint_paths = sorted(SOURCE_CHECKPOINT_DIR.glob(SOURCE_CHECKPOINT_GLOB))

    if not checkpoint_paths:
        raise FileNotFoundError(
            f"No PlantDoc-fine-tuned checkpoints found in {SOURCE_CHECKPOINT_DIR}. "
            f"Expected files matching: {SOURCE_CHECKPOINT_GLOB}"
        )

    return checkpoint_paths


def get_checkpoint_identity(checkpoint: dict) -> tuple[str, str, str]:
    """
    Read model identity fields from a PlantDoc-fine-tuned checkpoint.

    Args:
        checkpoint:
            Loaded checkpoint dictionary.

    Returns:
        model_name:
            Architecture name.
        loss_name:
            Loss function name.
        run_name:
            Source run name.
    """
    required_keys = {"model_name", "loss_name", "run_name"}
    missing_keys = required_keys - set(checkpoint.keys())

    if missing_keys:
        raise KeyError(
            f"Checkpoint is missing identity keys: {sorted(missing_keys)}"
        )

    model_name = checkpoint["model_name"]
    loss_name = checkpoint["loss_name"]
    run_name = checkpoint["run_name"]

    return model_name, loss_name, run_name


def validate_class_order(
    old_class_names: list[str],
    expanded_class_names: list[str],
) -> None:
    """
    Verify that old 38 labels remain first in the expanded 132-class label list.

    This check is critical because classifier rows 0-37 are copied from the
    PlantDoc-fine-tuned checkpoint into the expanded classifier head.

    Args:
        old_class_names:
            Class names saved inside the 38-class checkpoint.
        expanded_class_names:
            Expanded 132-class label list.

    Raises:
        RuntimeError:
            If the expanded label prefix does not match the old checkpoint labels.
    """
    old_class_names = list(old_class_names)
    expanded_prefix = list(expanded_class_names[: len(old_class_names)])

    if expanded_prefix != old_class_names:
        raise RuntimeError(
            "Expanded class order is invalid. The first old-class slots in the "
            "expanded label list must exactly match the checkpoint labels."
        )


def can_expand_classifier_tensor(
    old_tensor: torch.Tensor,
    new_tensor: torch.Tensor,
) -> bool:
    """
    Check whether a classifier tensor can be copied into an expanded tensor.

    Supported shapes:
        weight:
            [38, hidden_dim] -> [132, hidden_dim]

        bias:
            [38] -> [132]

    Args:
        old_tensor:
            Tensor from the 38-class checkpoint.
        new_tensor:
            Tensor from the 132-class model.

    Returns:
        True if the old rows can be safely copied into the new tensor.
    """
    if old_tensor.ndim not in {1, 2}:
        return False

    if old_tensor.shape[0] >= new_tensor.shape[0]:
        return False

    if old_tensor.shape[1:] != new_tensor.shape[1:]:
        return False

    return True


def load_checkpoint_into_expanded_model(
    model: nn.Module,
    checkpoint_state_dict: dict[str, torch.Tensor],
) -> tuple[list[str], list[str], list[str]]:
    """
    Load a 38-class checkpoint into a 132-class model.

    Compatible tensors are loaded directly. Classifier tensors are expanded by
    copying old rows into rows 0-37. New rows remain randomly initialized.

    Args:
        model:
            Newly built 132-class model.
        checkpoint_state_dict:
            model_state_dict from the 38-class checkpoint.

    Returns:
        loaded_keys:
            Tensor keys loaded directly.
        expanded_keys:
            Tensor keys expanded by row-copying.
        skipped_keys:
            Tensor keys that could not be loaded.
    """
    new_state_dict = model.state_dict()

    loaded_keys = []
    expanded_keys = []
    skipped_keys = []

    for key, old_tensor in checkpoint_state_dict.items():
        if key not in new_state_dict:
            skipped_keys.append(key)
            continue

        new_tensor = new_state_dict[key]

        if old_tensor.shape == new_tensor.shape:
            new_state_dict[key] = old_tensor
            loaded_keys.append(key)

        elif can_expand_classifier_tensor(
            old_tensor=old_tensor,
            new_tensor=new_tensor,
        ):
            expanded_tensor = new_tensor.clone()
            expanded_tensor[: old_tensor.shape[0]] = old_tensor
            new_state_dict[key] = expanded_tensor
            expanded_keys.append(key)

        else:
            skipped_keys.append(key)

    model.load_state_dict(new_state_dict)

    return loaded_keys, expanded_keys, skipped_keys


def extract_labels_from_dataset(dataset) -> list[int]:
    """
    Extract labels from Dataset, Subset, or ConcatDataset-like objects.

    Supported cases:
        - Dataset with .samples
        - torch.utils.data.Subset
        - torch.utils.data.ConcatDataset

    Args:
        dataset:
            Dataset-like object.

    Returns:
        List of integer labels.
    """
    if hasattr(dataset, "datasets"):
        labels = []

        for child_dataset in dataset.datasets:
            labels.extend(extract_labels_from_dataset(child_dataset))

        return labels

    if hasattr(dataset, "indices") and hasattr(dataset, "dataset"):
        labels = []

        for original_index in dataset.indices:
            sample = dataset.dataset.samples[original_index]
            labels.append(int(sample[1]))

        return labels

    if hasattr(dataset, "samples"):
        return [int(sample[1]) for sample in dataset.samples]

    raise TypeError(f"Unsupported dataset type for label extraction: {type(dataset)}")


def compute_replay_aware_class_weights(
    plantwild_dataset,
    plantdoc_dataset,
    plantvillage_dataset,
    num_classes: int,
    device: torch.device,
    plantwild_ratio: float,
    plantdoc_ratio: float,
    plantvillage_ratio: float,
) -> torch.Tensor:
    """
    Compute class weights using the intended replay sampling ratios.

    This prevents PlantVillage from dominating class weights simply because it
    contains many more raw images than PlantDoc replay data.

    Args:
        plantwild_dataset:
            PlantWild training dataset.
        plantdoc_dataset:
            PlantDoc replay subset.
        plantvillage_dataset:
            PlantVillage replay dataset.
        num_classes:
            Expanded class count.
        device:
            CPU or CUDA device.
        plantwild_ratio:
            PlantWild sampling ratio.
        plantdoc_ratio:
            PlantDoc replay ratio.
        plantvillage_ratio:
            PlantVillage replay ratio.

    Returns:
        Class-weight tensor with shape [num_classes].
    """
    ratio_sum = plantwild_ratio + plantdoc_ratio + plantvillage_ratio

    if abs(ratio_sum - 1.0) > 1e-6:
        raise ValueError(f"Replay ratios must sum to 1. Got: {ratio_sum}")

    class_mass = torch.zeros(num_classes, dtype=torch.float32)

    datasets_and_ratios = [
        (plantwild_dataset, plantwild_ratio),
        (plantdoc_dataset, plantdoc_ratio),
        (plantvillage_dataset, plantvillage_ratio),
    ]

    for dataset, ratio in datasets_and_ratios:
        labels = extract_labels_from_dataset(dataset)

        if not labels:
            continue

        per_sample_mass = ratio / len(labels)

        for label in labels:
            if label < 0 or label >= num_classes:
                raise ValueError(
                    f"Label index {label} is outside valid range "
                    f"[0, {num_classes - 1}]."
                )

            class_mass[label] += per_sample_mass

    present_mask = class_mass > 0

    if present_mask.sum() == 0:
        raise RuntimeError("No labels found while computing class weights.")

    weights = torch.zeros(num_classes, dtype=torch.float32)
    present_mass = class_mass[present_mask]

    weights[present_mask] = present_mass.sum() / (
        present_mask.sum() * present_mass
    )

    return weights.to(device)


def create_loss_function_for_plantwild(
    loss_name: str,
    focal_gamma: float,
    num_classes: int,
    device: torch.device,
    plantwild_dataset,
    plantdoc_dataset,
    plantvillage_dataset,
    plantwild_config: dict,
) -> nn.Module:
    """
    Create the loss function for PlantWild-expanded training.

    The loss follows the source checkpoint's original loss:
        - cross_entropy
        - weighted_cross_entropy
        - focal_loss

    Args:
        loss_name:
            Original loss type from the checkpoint.
        focal_gamma:
            Focal-loss gamma.
        num_classes:
            Expanded class count.
        device:
            CPU or CUDA device.
        plantwild_dataset:
            PlantWild train dataset.
        plantdoc_dataset:
            PlantDoc replay dataset.
        plantvillage_dataset:
            PlantVillage replay dataset.
        plantwild_config:
            PlantWild training configuration.

    Returns:
        PyTorch loss module.
    """
    if loss_name == "cross_entropy":
        print("Using CrossEntropyLoss.")
        return nn.CrossEntropyLoss()

    if loss_name == "weighted_cross_entropy":
        weights = compute_replay_aware_class_weights(
            plantwild_dataset=plantwild_dataset,
            plantdoc_dataset=plantdoc_dataset,
            plantvillage_dataset=plantvillage_dataset,
            num_classes=num_classes,
            device=device,
            plantwild_ratio=float(plantwild_config["plantwild_ratio"]),
            plantdoc_ratio=float(plantwild_config["plantdoc_ratio"]),
            plantvillage_ratio=float(plantwild_config["plantvillage_ratio"]),
        )

        print("Using replay-aware weighted CrossEntropyLoss.")
        print(f"Non-zero class weights: {(weights > 0).sum().item()}")
        print(f"Max class weight: {weights.max().item():.4f}")

        return nn.CrossEntropyLoss(weight=weights)

    if loss_name == "focal_loss":
        print(f"Using FocalLoss with gamma={focal_gamma}.")
        return FocalLoss(gamma=focal_gamma)

    raise ValueError(f"Unsupported PlantWild training loss: {loss_name}")


def get_focal_gamma(
    checkpoint: dict,
    default: float = 2.0,
) -> float:
    """
    Read focal gamma from the original source config when available.

    Args:
        checkpoint:
            Loaded PlantDoc-fine-tuned checkpoint.
        default:
            Fallback gamma value.

    Returns:
        Focal gamma as float.
    """
    source_config = checkpoint.get("source_config", {})
    training_config = source_config.get("training", {})

    return float(training_config.get("focal_gamma", default))


@torch.no_grad()
def validate_model(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> dict[str, float]:
    """
    Validate a model on one dataset.

    Args:
        model:
            PyTorch model.
        loader:
            Validation DataLoader.
        criterion:
            Loss function.
        device:
            CPU or CUDA device.

    Returns:
        Dictionary containing loss, accuracy, macro F1, and weighted F1.
    """
    model.eval()

    total_loss = 0.0
    correct_predictions = 0
    total_samples = 0

    all_labels = []
    all_predictions = []

    for images, labels in loader:
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
        raise RuntimeError("Validation loader produced zero samples.")

    return {
        "loss": total_loss / total_samples,
        "accuracy": correct_predictions / total_samples,
        "macro_f1": f1_score(
            all_labels,
            all_predictions,
            average="macro",
            zero_division=0,
        ),
        "weighted_f1": f1_score(
            all_labels,
            all_predictions,
            average="weighted",
            zero_division=0,
        ),
    }


def compute_selection_score(
    plantwild_metrics: dict[str, float],
    plantdoc_metrics: dict[str, float],
    plantvillage_metrics: dict[str, float],
    plantwild_config: dict,
) -> float:
    """
    Compute weighted validation score for model selection.

    Args:
        plantwild_metrics:
            PlantWild validation metrics.
        plantdoc_metrics:
            PlantDoc validation metrics.
        plantvillage_metrics:
            PlantVillage validation metrics.
        plantwild_config:
            PlantWild training configuration.

    Returns:
        Weighted selection score.
    """
    return (
        float(plantwild_config["plantwild_score_weight"])
        * plantwild_metrics["macro_f1"]
        + float(plantwild_config["plantdoc_score_weight"])
        * plantdoc_metrics["macro_f1"]
        + float(plantwild_config["plantvillage_score_weight"])
        * plantvillage_metrics["macro_f1"]
    )


def log_validation_metrics(
    plantwild_metrics: dict[str, float],
    plantdoc_metrics: dict[str, float],
    plantvillage_metrics: dict[str, float],
    selection_score: float,
    epoch: int,
) -> None:
    """
    Print and log validation metrics for one epoch.

    Args:
        plantwild_metrics:
            PlantWild validation metrics.
        plantdoc_metrics:
            PlantDoc validation metrics.
        plantvillage_metrics:
            PlantVillage validation metrics.
        selection_score:
            Weighted model-selection score.
        epoch:
            Global epoch number.
    """
    print(
        f"Epoch {epoch} validation | "
        f"score={selection_score:.4f} | "
        f"wild_f1={plantwild_metrics['macro_f1']:.4f} "
        f"doc_f1={plantdoc_metrics['macro_f1']:.4f} "
        f"pv_f1={plantvillage_metrics['macro_f1']:.4f} | "
        f"wild_acc={plantwild_metrics['accuracy']:.4f} "
        f"doc_acc={plantdoc_metrics['accuracy']:.4f} "
        f"pv_acc={plantvillage_metrics['accuracy']:.4f}"
    )

    metric_groups = [
        ("plantwild_val", plantwild_metrics),
        ("plantdoc_val", plantdoc_metrics),
        ("plantvillage_val", plantvillage_metrics),
    ]

    for prefix, metrics in metric_groups:
        for metric_name, metric_value in metrics.items():
            mlflow.log_metric(f"{prefix}_{metric_name}", metric_value, step=epoch)

    mlflow.log_metric("selection_score", selection_score, step=epoch)


def save_expanded_checkpoint(
    output_path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    source_checkpoint_path: Path,
    source_checkpoint: dict,
    expanded_class_names: list[str],
    model_name: str,
    loss_name: str,
    run_name: str,
    epoch: int,
    stage: str,
    best_metrics: dict[str, float],
    plantwild_config: dict,
) -> None:
    """
    Save the best PlantWild-expanded checkpoint.

    Args:
        output_path:
            Destination checkpoint path.
        model:
            Expanded PyTorch model.
        optimizer:
            Optimizer state at save time.
        source_checkpoint_path:
            Source PlantDoc-fine-tuned checkpoint path.
        source_checkpoint:
            Loaded source checkpoint dictionary.
        expanded_class_names:
            Expanded 132-class label list.
        model_name:
            Architecture name.
        loss_name:
            Loss name.
        run_name:
            Source run name.
        epoch:
            Epoch where checkpoint is saved.
        stage:
            Training stage, "head" or "full".
        best_metrics:
            Best validation metrics.
        plantwild_config:
            PlantWild training configuration.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    checkpoint = {
        "source_checkpoint": source_checkpoint_path.name,
        "source_stage": "plantdoc_finetuned",
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "class_names": expanded_class_names,
        "model_name": model_name,
        "loss_name": loss_name,
        "run_name": f"{run_name}_plantwild_expanded",
        "epoch": epoch,
        "stage": stage,
        "best_selection_score": best_metrics["selection_score"],
        "best_plantwild_macro_f1": best_metrics["plantwild_macro_f1"],
        "best_plantdoc_macro_f1": best_metrics["plantdoc_macro_f1"],
        "best_plantvillage_macro_f1": best_metrics["plantvillage_macro_f1"],
        "source_checkpoint_metadata": {
            key: value
            for key, value in source_checkpoint.items()
            if key not in {"model_state_dict", "optimizer_state_dict"}
        },
        "plantwild_training_config": dict(plantwild_config),
    }

    torch.save(checkpoint, output_path)


def maybe_save_best(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    output_path: Path,
    source_checkpoint_path: Path,
    source_checkpoint: dict,
    expanded_class_names: list[str],
    model_name: str,
    loss_name: str,
    run_name: str,
    epoch: int,
    stage: str,
    current_metrics: dict[str, float],
    best_metrics: dict[str, float],
    plantwild_config: dict,
) -> tuple[dict[str, float], bool]:
    """
    Save a checkpoint if the current selection score is best so far.

    Args:
        current_metrics:
            Current epoch metrics.
        best_metrics:
            Best metrics seen so far.

    Returns:
        updated_best_metrics:
            Best metrics after comparison.
        improved:
            True if a new best checkpoint was saved.
    """
    if current_metrics["selection_score"] <= best_metrics["selection_score"]:
        return best_metrics, False

    save_expanded_checkpoint(
        output_path=output_path,
        model=model,
        optimizer=optimizer,
        source_checkpoint_path=source_checkpoint_path,
        source_checkpoint=source_checkpoint,
        expanded_class_names=expanded_class_names,
        model_name=model_name,
        loss_name=loss_name,
        run_name=run_name,
        epoch=epoch,
        stage=stage,
        best_metrics=current_metrics,
        plantwild_config=plantwild_config,
    )

    print(
        f"Saved new best: {output_path.name} | "
        f"selection_score={current_metrics['selection_score']:.4f} | "
        f"wild_f1={current_metrics['plantwild_macro_f1']:.4f}"
    )

    return current_metrics, True


def run_training_stage(
    model: nn.Module,
    model_name: str,
    train_loader: torch.utils.data.DataLoader,
    plantwild_val_loader: torch.utils.data.DataLoader,
    plantdoc_val_loader: torch.utils.data.DataLoader,
    plantvillage_val_loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    stage_name: str,
    epochs: int,
    learning_rate: float,
    weight_decay: float,
    optimizer_name: str,
    start_epoch: int,
    best_metrics: dict[str, float],
    save_context: dict,
    plantwild_config: dict,
    device: torch.device,
) -> tuple[int, dict[str, float]]:
    """
    Run one PlantWild-expanded training stage.

    Stages:
        head:
            Freeze the backbone and train only the expanded classifier head.

        full:
            Unfreeze all layers and train the full model with replay.

    Args:
        model:
            PyTorch model.
        model_name:
            Architecture name.
        train_loader:
            Mixed 80/12/8 training DataLoader.
        plantwild_val_loader:
            PlantWild validation DataLoader.
        plantdoc_val_loader:
            PlantDoc validation DataLoader.
        plantvillage_val_loader:
            PlantVillage validation DataLoader.
        criterion:
            Loss function.
        stage_name:
            "head" or "full".
        epochs:
            Number of epochs in this stage.
        learning_rate:
            Stage learning rate.
        weight_decay:
            Stage weight decay.
        optimizer_name:
            Optimizer name.
        start_epoch:
            Global epoch number before this stage.
        best_metrics:
            Best metrics seen so far.
        save_context:
            Static arguments needed by maybe_save_best.
        plantwild_config:
            PlantWild training configuration.
        device:
            CPU or CUDA device.

    Returns:
        current_epoch:
            Last global epoch number after this stage.
        best_metrics:
            Updated best metrics.
    """
    if epochs <= 0:
        return start_epoch, best_metrics

    if stage_name == "head":
        print("\nStage 1: frozen backbone, train expanded classifier head")
        freeze_backbone(model, model_name)

    elif stage_name == "full":
        print("\nStage 2: full-model PlantWild replay training")
        unfreeze_model(model)

    else:
        raise ValueError(f"Unsupported PlantWild training stage: {stage_name}")

    optimizer = create_stage_optimizer(
        model=model,
        stage_name=stage_name,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        optimizer_name=optimizer_name,
    )

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

        mlflow.log_metric("train_loss", train_loss, step=current_epoch)
        mlflow.log_metric("train_accuracy", train_accuracy, step=current_epoch)

        print(
            f"Epoch {current_epoch} train | "
            f"loss={train_loss:.4f} acc={train_accuracy:.4f}"
        )

        plantwild_metrics = validate_model(
            model=model,
            loader=plantwild_val_loader,
            criterion=criterion,
            device=device,
        )

        plantdoc_metrics = validate_model(
            model=model,
            loader=plantdoc_val_loader,
            criterion=criterion,
            device=device,
        )

        plantvillage_metrics = validate_model(
            model=model,
            loader=plantvillage_val_loader,
            criterion=criterion,
            device=device,
        )

        selection_score = compute_selection_score(
            plantwild_metrics=plantwild_metrics,
            plantdoc_metrics=plantdoc_metrics,
            plantvillage_metrics=plantvillage_metrics,
            plantwild_config=plantwild_config,
        )

        log_validation_metrics(
            plantwild_metrics=plantwild_metrics,
            plantdoc_metrics=plantdoc_metrics,
            plantvillage_metrics=plantvillage_metrics,
            selection_score=selection_score,
            epoch=current_epoch,
        )

        current_metrics = {
            "selection_score": selection_score,
            "plantwild_macro_f1": plantwild_metrics["macro_f1"],
            "plantdoc_macro_f1": plantdoc_metrics["macro_f1"],
            "plantvillage_macro_f1": plantvillage_metrics["macro_f1"],
            "plantwild_accuracy": plantwild_metrics["accuracy"],
            "plantdoc_accuracy": plantdoc_metrics["accuracy"],
            "plantvillage_accuracy": plantvillage_metrics["accuracy"],
        }

        best_metrics, _ = maybe_save_best(
            model=model,
            optimizer=optimizer,
            current_metrics=current_metrics,
            best_metrics=best_metrics,
            epoch=current_epoch,
            stage=stage_name,
            plantwild_config=plantwild_config,
            **save_context,
        )

    return current_epoch, best_metrics


def build_expanded_model_from_checkpoint(
    checkpoint_path: Path,
    expanded_class_names: list[str],
    device: torch.device,
) -> tuple[nn.Module, dict, str, str, str]:
    """
    Build a 132-class model and load a 38-class checkpoint into it.

    Args:
        checkpoint_path:
            PlantDoc-fine-tuned checkpoint path.
        expanded_class_names:
            Expanded 132-class label list.
        device:
            CPU or CUDA device.

    Returns:
        model:
            Expanded model.
        checkpoint:
            Loaded source checkpoint.
        model_name:
            Architecture name.
        loss_name:
            Loss name.
        run_name:
            Source run name.
    """
    checkpoint = load_checkpoint(checkpoint_path)

    required_keys = {"class_names", "model_state_dict", "model_name", "loss_name", "run_name"}
    missing_keys = required_keys - set(checkpoint.keys())

    if missing_keys:
        raise KeyError(
            f"Checkpoint {checkpoint_path.name} is missing keys: "
            f"{sorted(missing_keys)}"
        )

    old_class_names = list(checkpoint["class_names"])

    validate_class_order(
        old_class_names=old_class_names,
        expanded_class_names=expanded_class_names,
    )

    model_name, loss_name, run_name = get_checkpoint_identity(checkpoint)

    model = build_model(
        model_name=model_name,
        num_classes=len(expanded_class_names),
        pretrained=False,
    )

    loaded_keys, expanded_keys, skipped_keys = load_checkpoint_into_expanded_model(
        model=model,
        checkpoint_state_dict=checkpoint["model_state_dict"],
    )

    if len(expanded_keys) != 2:
        raise RuntimeError(
            "Expected exactly 2 expanded classifier tensors "
            f"(weight and bias), got {len(expanded_keys)}: {expanded_keys}"
        )

    if skipped_keys:
        raise RuntimeError(
            "Some checkpoint tensors could not be loaded:\n"
            + "\n".join(f"- {key}" for key in skipped_keys)
        )

    model.to(device)

    print("\nCheckpoint expansion:")
    print(f"  Source:           {checkpoint_path.name}")
    print(f"  Model:            {model_name}")
    print(f"  Loss:             {loss_name}")
    print(f"  Old classes:      {len(old_class_names)}")
    print(f"  Expanded classes: {len(expanded_class_names)}")
    print(f"  Direct tensors:   {len(loaded_keys)}")
    print(f"  Expanded tensors: {expanded_keys}")

    return model, checkpoint, model_name, loss_name, run_name


def train_one_checkpoint(
    checkpoint_path: Path,
    loaders: dict,
    expanded_class_names: list[str],
    plantwild_config: dict,
    device: torch.device,
) -> dict:
    """
    Train one PlantWild-expanded model from one PlantDoc-fine-tuned checkpoint.

    Args:
        checkpoint_path:
            PlantDoc-fine-tuned source checkpoint.
        loaders:
            Dictionary of mixed training and validation loaders/datasets.
        expanded_class_names:
            Expanded 132-class label list.
        plantwild_config:
            PlantWild training configuration.
        device:
            CPU or CUDA device.

    Returns:
        Summary row for plantwild_training_summary.csv.
    """
    model, checkpoint, model_name, loss_name, run_name = build_expanded_model_from_checkpoint(
        checkpoint_path=checkpoint_path,
        expanded_class_names=expanded_class_names,
        device=device,
    )

    total_parameters, trainable_parameters = count_parameters(model)
    focal_gamma = get_focal_gamma(checkpoint)

    criterion = create_loss_function_for_plantwild(
        loss_name=loss_name,
        focal_gamma=focal_gamma,
        num_classes=len(expanded_class_names),
        device=device,
        plantwild_dataset=loaders["plantwild_train_dataset"],
        plantdoc_dataset=loaders["plantdoc_train_subset"],
        plantvillage_dataset=loaders["plantvillage_train_dataset"],
        plantwild_config=plantwild_config,
    )

    save_dir = resolve_project_path(plantwild_config["save_dir"])
    save_dir.mkdir(parents=True, exist_ok=True)

    output_path = save_dir / f"{run_name}_plantwild_expanded_best_model.pth"

    best_metrics = {
        "selection_score": -1.0,
        "plantwild_macro_f1": -1.0,
        "plantdoc_macro_f1": -1.0,
        "plantvillage_macro_f1": -1.0,
        "plantwild_accuracy": -1.0,
        "plantdoc_accuracy": -1.0,
        "plantvillage_accuracy": -1.0,
    }

    save_context = {
        "output_path": output_path,
        "source_checkpoint_path": checkpoint_path,
        "source_checkpoint": checkpoint,
        "expanded_class_names": expanded_class_names,
        "model_name": model_name,
        "loss_name": loss_name,
        "run_name": run_name,
    }

    print("\n" + "=" * 80)
    print(f"Training PlantWild-expanded model: {run_name}")
    print(f"Checkpoint: {checkpoint_path.name}")
    print(f"Model:      {model_name}")
    print(f"Loss:       {loss_name}")
    print(f"Params:     {total_parameters:,}")
    print("=" * 80)

    with mlflow.start_run(run_name=f"{run_name}_plantwild_expanded"):
        mlflow.log_param("source_checkpoint", checkpoint_path.name)
        mlflow.log_param("model_name", model_name)
        mlflow.log_param("loss_name", loss_name)
        mlflow.log_param("expanded_num_classes", len(expanded_class_names))
        mlflow.log_param("total_parameters", total_parameters)
        mlflow.log_param("trainable_parameters_initial", trainable_parameters)
        mlflow.log_param("focal_gamma", focal_gamma)

        for key, value in plantwild_config.items():
            mlflow.log_param(f"plantwild_training.{key}", str(value))

        epoch = 0

        epoch, best_metrics = run_training_stage(
            model=model,
            model_name=model_name,
            train_loader=loaders["mixed_train_loader"],
            plantwild_val_loader=loaders["plantwild_val_loader"],
            plantdoc_val_loader=loaders["plantdoc_val_loader"],
            plantvillage_val_loader=loaders["plantvillage_val_loader"],
            criterion=criterion,
            stage_name="head",
            epochs=int(plantwild_config["head_epochs"]),
            learning_rate=float(plantwild_config["lr_head"]),
            weight_decay=float(plantwild_config["weight_decay"]),
            optimizer_name=plantwild_config["optimizer"],
            start_epoch=epoch,
            best_metrics=best_metrics,
            save_context=save_context,
            plantwild_config=plantwild_config,
            device=device,
        )

        epoch, best_metrics = run_training_stage(
            model=model,
            model_name=model_name,
            train_loader=loaders["mixed_train_loader"],
            plantwild_val_loader=loaders["plantwild_val_loader"],
            plantdoc_val_loader=loaders["plantdoc_val_loader"],
            plantvillage_val_loader=loaders["plantvillage_val_loader"],
            criterion=criterion,
            stage_name="full",
            epochs=int(plantwild_config["full_epochs"]),
            learning_rate=float(plantwild_config["lr_full"]),
            weight_decay=float(plantwild_config["weight_decay"]),
            optimizer_name=plantwild_config["optimizer"],
            start_epoch=epoch,
            best_metrics=best_metrics,
            save_context=save_context,
            plantwild_config=plantwild_config,
            device=device,
        )

        for metric_name, metric_value in best_metrics.items():
            mlflow.log_metric(f"best_{metric_name}", metric_value)

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
        "best_selection_score": best_metrics["selection_score"],
        "best_plantwild_macro_f1": best_metrics["plantwild_macro_f1"],
        "best_plantdoc_macro_f1": best_metrics["plantdoc_macro_f1"],
        "best_plantvillage_macro_f1": best_metrics["plantvillage_macro_f1"],
        "best_plantwild_accuracy": best_metrics["plantwild_accuracy"],
        "best_plantdoc_accuracy": best_metrics["plantdoc_accuracy"],
        "best_plantvillage_accuracy": best_metrics["plantvillage_accuracy"],
        "expanded_checkpoint": str(output_path.relative_to(PROJECT_ROOT)),
    }

    del model
    gc.collect()

    if device.type == "cuda":
        torch.cuda.empty_cache()

    return summary_row


def build_loader_dictionary(
    mixed_train_loader,
    plantwild_val_loader,
    plantdoc_val_loader,
    plantvillage_val_loader,
) -> dict:
    """
    Build the loader dictionary used by train_one_checkpoint.

    Args:
        mixed_train_loader:
            Mixed 80/12/8 training loader.
        plantwild_val_loader:
            PlantWild validation loader.
        plantdoc_val_loader:
            PlantDoc validation loader.
        plantvillage_val_loader:
            PlantVillage validation loader.

    Returns:
        Dictionary containing loaders and source datasets.
    """
    mixed_dataset = mixed_train_loader.dataset

    if not hasattr(mixed_dataset, "datasets") or len(mixed_dataset.datasets) != 3:
        raise RuntimeError(
            "Expected mixed_train_loader.dataset to be a ConcatDataset with "
            "three datasets: PlantWild, PlantDoc replay, PlantVillage replay."
        )

    return {
        "mixed_train_loader": mixed_train_loader,
        "plantwild_val_loader": plantwild_val_loader,
        "plantdoc_val_loader": plantdoc_val_loader,
        "plantvillage_val_loader": plantvillage_val_loader,
        "plantwild_train_dataset": mixed_dataset.datasets[0],
        "plantdoc_train_subset": mixed_dataset.datasets[1],
        "plantvillage_train_dataset": mixed_dataset.datasets[2],
    }


def main() -> None:
    """
    Train all selected PlantWild-expanded models.
    """
    config = load_config(CONFIG_PATH)
    plantwild_config = get_plantwild_training_config(config)

    set_seed(int(config["project"]["seed"]))

    device = get_device(config["training"]["device"])
    print(f"Using device: {device}")

    output_results_dir = resolve_project_path(plantwild_config["output_dir"])
    output_results_dir.mkdir(parents=True, exist_ok=True)

    mlflow.set_tracking_uri(f"sqlite:///{MLFLOW_DB_PATH.as_posix()}")
    mlflow.set_experiment(plantwild_config["experiment_name"])

    (
        mixed_train_loader,
        plantwild_val_loader,
        plantdoc_val_loader,
        plantvillage_val_loader,
        expanded_class_names,
        loader_metadata,
    ) = get_expanded_training_loaders(
        batch_size=int(plantwild_config["batch_size"]),
        num_workers=int(plantwild_config["num_workers"]),
        plantwild_ratio=float(plantwild_config["plantwild_ratio"]),
        plantdoc_ratio=float(plantwild_config["plantdoc_ratio"]),
        plantvillage_ratio=float(plantwild_config["plantvillage_ratio"]),
        plantdoc_val_ratio=float(plantwild_config["plantdoc_val_ratio"]),
        seed=int(config["project"]["seed"]),
    )

    source_checkpoints = discover_source_checkpoints(
        plantwild_config.get("checkpoint")
    )

    print("\nPlantWild-expanded training setup:")
    print(f"  Source checkpoints: {len(source_checkpoints)}")
    print(f"  Expanded classes:   {len(expanded_class_names)}")
    print(f"  Loader metadata:    {loader_metadata}")

    loaders = build_loader_dictionary(
        mixed_train_loader=mixed_train_loader,
        plantwild_val_loader=plantwild_val_loader,
        plantdoc_val_loader=plantdoc_val_loader,
        plantvillage_val_loader=plantvillage_val_loader,
    )

    rows = []

    for checkpoint_path in source_checkpoints:
        rows.append(
            train_one_checkpoint(
                checkpoint_path=checkpoint_path,
                loaders=loaders,
                expanded_class_names=expanded_class_names,
                plantwild_config=plantwild_config,
                device=device,
            )
        )

    summary_df = pd.DataFrame(rows).sort_values(
        by="best_selection_score",
        ascending=False,
    )

    summary_path = output_results_dir / "plantwild_training_summary.csv"
    summary_df.to_csv(summary_path, index=False)

    with mlflow.start_run(run_name="plantwild_training_summary"):
        mlflow.log_param("num_trained_models", len(summary_df))
        mlflow.log_param("expanded_num_classes", len(expanded_class_names))

        for key, value in loader_metadata.items():
            mlflow.log_param(f"loader_metadata.{key}", value)

        mlflow.log_artifact(
            str(summary_path),
            artifact_path="plantwild_training_summary",
        )

    print("\nPlantWild-expanded training complete.")
    print(f"Summary saved to: {summary_path}")
    print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()