"""
Train expanded PlantGuard models on PlantWild_v2 with replay.

This script trains the final expanded PlantGuard models starting from the
best PlantDoc-finetuned checkpoints.

Training mix:
    80% PlantWild_v2
    12% PlantDoc train-adapt replay
    8% PlantVillage replay

Model expansion:
    PlantDoc-finetuned checkpoints have 38 output classes.
    PlantWild-expanded models have 132 output classes.

Expansion rule:
    - Build the same architecture with 132 output classes.
    - Copy all compatible backbone weights.
    - Copy old classifier rows 0-37 into the new classifier head.
    - Leave rows 38-131 randomly initialized.

Model selection:
    Save best checkpoint by weighted validation score:

        0.80 * PlantWild macro F1
      + 0.12 * PlantDoc macro F1
      + 0.08 * PlantVillage macro F1

This file is training code, not a separate smoke-test script.
"""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

import mlflow
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import f1_score

from data.dataset import get_expanded_training_loaders  # noqa: E402
from training.evaluate import count_parameters, load_checkpoint  # noqa: E402
from training.finetune_plantdoc import (
    freeze_backbone, 
    unfreeze_model, 
    create_stage_optimizer
)
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

SOURCE_CHECKPOINT_DIR = PROJECT_ROOT / "models" / "plantdoc_finetuned"
MLFLOW_DB_PATH = PROJECT_ROOT / "mlflow.db"

def get_plantwild_training_config(config):
    """
    Load PlantWild training config from training/config.yaml.

    The config file is the single source of truth for PlantWild-expanded
    training settings.

    Args:
        config:
            Full project config dictionary loaded from config.yaml.

    Returns:
        PlantWild training config dictionary.

    Raises:
        KeyError:
            If config.yaml does not contain the plantwild_training section.
    """
    if "plantwild_training" not in config:
        raise KeyError(
            "Missing 'plantwild_training' section in training/config.yaml."
        )

    return config["plantwild_training"]


def resolve_project_path(path_value):
    """
    Convert a relative project path or absolute path into a Path object.

    Args:
        path_value:
            Path string.

    Returns:
        Absolute Path.
    """
    path = Path(path_value)

    if path.is_absolute():
        return path

    return PROJECT_ROOT / path


def discover_source_checkpoints(checkpoint_from_config):
    """
    Find PlantDoc-finetuned checkpoints to train on PlantWild.

    Args:
        checkpoint_from_config:
            Optional single checkpoint path from config.

    Returns:
        List of checkpoint paths.
    """
    if checkpoint_from_config is not None:
        return [resolve_project_path(checkpoint_from_config)]

    checkpoint_paths = sorted(
        SOURCE_CHECKPOINT_DIR.glob("*_plantdoc_finetuned_best_model.pth")
    )

    if not checkpoint_paths:
        raise FileNotFoundError(
            f"No PlantDoc-finetuned checkpoints found in {SOURCE_CHECKPOINT_DIR}"
        )

    return checkpoint_paths


def get_checkpoint_identity(checkpoint):
    """
    Read checkpoint identity fields.

    Args:
        checkpoint:
            Loaded checkpoint dictionary.

    Returns:
        model_name, loss_name, run_name.
    """
    model_name = checkpoint["model_name"]
    loss_name = checkpoint["loss_name"]
    run_name = checkpoint["run_name"]

    return model_name, loss_name, run_name


def validate_class_order(old_class_names, expanded_class_names):
    """
    Make sure old 38 labels are still the first 38 expanded labels.

    Args:
        old_class_names:
            Class names saved inside the 38-class checkpoint.
        expanded_class_names:
            Expanded 132-class class-name list.
    """
    old_class_names = list(old_class_names)
    expanded_prefix = list(expanded_class_names[:len(old_class_names)])

    if expanded_prefix != old_class_names:
        raise RuntimeError(
            "Expanded class order is invalid. "
            "The first 38 expanded labels must exactly match checkpoint labels."
        )


def can_expand_classifier_tensor(old_tensor, new_tensor):
    """
    Check whether old classifier tensor can be copied into new tensor.

    Supports:
        weight: [38, hidden_dim] -> [132, hidden_dim]
        bias:   [38]             -> [132]

    Args:
        old_tensor:
            Tensor from old checkpoint.
        new_tensor:
            Tensor from expanded model.

    Returns:
        True if first rows can be copied safely.
    """
    if old_tensor.ndim not in {1, 2}:
        return False

    if old_tensor.shape[0] >= new_tensor.shape[0]:
        return False

    if old_tensor.shape[1:] != new_tensor.shape[1:]:
        return False

    return True


def load_checkpoint_into_expanded_model(model, checkpoint_state_dict):
    """
    Load a 38-class checkpoint into a 132-class model.

    Compatible tensors are loaded directly.
    Classifier tensors are expanded by copying the old rows into rows 0-37.

    Args:
        model:
            New expanded model.
        checkpoint_state_dict:
            38-class checkpoint model_state_dict.

    Returns:
        loaded_keys, expanded_keys, skipped_keys.
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

        elif can_expand_classifier_tensor(old_tensor,new_tensor):
            expanded_tensor = new_tensor.clone()
            expanded_tensor[:old_tensor.shape[0]] = old_tensor
            new_state_dict[key] = expanded_tensor
            expanded_keys.append(key)

        else:
            skipped_keys.append(key)

    model.load_state_dict(new_state_dict)

    return loaded_keys, expanded_keys, skipped_keys


def extract_labels_from_dataset(dataset):
    """
    Extract labels from Dataset, Subset, or ConcatDataset.

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
    num_classes,
    device,
    plantwild_ratio,
    plantdoc_ratio,
    plantvillage_ratio,
):
    """
    Compute class weights using the intended replay ratios.

    This avoids PlantVillage dominating class weights just because it has many
    more raw images.

    Args:
        plantwild_dataset:
            PlantWild train dataset.
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
        Class-weight tensor.
    """
    class_mass = torch.zeros(num_classes, dtype = torch.float32)

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
            class_mass[label] += per_sample_mass

    present_mask = class_mass>0

    if present_mask.sum() == 0:
        raise RuntimeError("No labels found while computing class weights.")
    
    weights = torch.zeros(num_classes, dtype=torch.float32)
    present_mass = class_mass[present_mask]

    weights[present_mask] = present_mass.sum() / (
        present_mask.sum() * present_mass
    )

    return weights.to(device)


def create_loss_function_for_plantwild(
        loss_name,
        focal_gamma,
        num_classes,
        device,
        plantwild_dataset,
        plantdoc_dataset,
        plantvillage_dataset,
        plantwild_config,
):
    """
    Create loss function for PlantWild-expanded training.

    Args:
        loss_name:
            Original loss type from checkpoint.
        focal_gamma:
            Focal loss gamma.
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
            PlantWild config dictionary.

    Returns:
        PyTorch loss function.
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
            plantwild_ratio=plantwild_config["plantwild_ratio"],
            plantdoc_ratio=plantwild_config["plantdoc_ratio"],
            plantvillage_ratio=plantwild_config["plantvillage_ratio"],
        )

        print("Using replay-aware weighted CrossEntropyLoss.")
        print(f"Non-zero class weights: {(weights > 0).sum().item()}")
        print(f"Max class weight: {weights.max().item():.4f}")

        return nn.CrossEntropyLoss(weight=weights)
    
    if loss_name == "focal_loss":
        print(f"Using FocalLoss with gamma={focal_gamma}.")
        return FocalLoss(gamma=focal_gamma)

    raise ValueError(f"Unsupported loss: {loss_name}")


def get_focal_gamma(checkpoint, default=2.0):
    """
    Read focal gamma from the original source config if available.

    Args:
        checkpoint:
            Loaded checkpoint dictionary.
        default:
            Default gamma.

    Returns:
        Focal gamma float.
    """
    source_config = checkpoint.get("source_config", {})
    training_config = source_config.get("training", {})

    return float(training_config.get("focal_gamma", default))


@torch.no_grad()
def validate_model(model, loader, criterion, device):
    """
    Validate model on one dataset.

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
        Dictionary of metrics.
    """
    model.eval()

    total_loss = 0.0
    correct = 0
    total = 0

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
        correct += (predictions == labels).sum().item()
        total += batch_size

        all_labels.extend(labels.cpu().tolist())
        all_predictions.extend(predictions.cpu().tolist())

    return {
        "loss": total_loss / total,
        "accuracy": correct / total,
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
        plantwild_metrics,
        plantdoc_metrics,
        plantvillage_metrics,
        plantwild_config,
):
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
            Config dictionary.

    Returns:
        Weighted selection score.
    """
    return (
        plantwild_config["plantwild_score_weight"] * plantwild_metrics["macro_f1"]
        + plantwild_config["plantdoc_score_weight"] * plantdoc_metrics["macro_f1"]
        + plantwild_config["plantvillage_score_weight"] * plantvillage_metrics["macro_f1"]
    )


def log_validation_metrics(
    plantwild_metrics,
    plantdoc_metrics,
    plantvillage_metrics,
    selection_score,
    epoch,
):
    """
    Print and log validation metrics.

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

    for prefix, metrics in [
        ("plantwild_val", plantwild_metrics),
        ("plantdoc_val", plantdoc_metrics),
        ("plantvillage_val", plantvillage_metrics),
    ]:
        for metric_name, metric_value in metrics.items():
            mlflow.log_metric(f"{prefix}_{metric_name}", metric_value, step=epoch)

    mlflow.log_metric("selection_score", selection_score, step=epoch)


def save_expanded_checkpoint(
    output_path,
    model,
    optimizer,
    source_checkpoint_path,
    source_checkpoint,
    expanded_class_names,
    model_name,
    loss_name,
    run_name,
    epoch,
    stage,
    best_metrics,
    plantwild_config,
):
    """
    Save best PlantWild-expanded checkpoint.

    Args:
        output_path:
            Save path.
        model:
            PyTorch model.
        optimizer:
            Optimizer.
        source_checkpoint_path:
            PlantDoc-finetuned checkpoint path.
        source_checkpoint:
            Loaded PlantDoc-finetuned checkpoint.
        expanded_class_names:
            Expanded class names.
        model_name:
            Architecture name.
        loss_name:
            Loss name.
        run_name:
            Source run name.
        epoch:
            Epoch where checkpoint is saved.
        stage:
            Training stage.
        best_metrics:
            Best metric dictionary.
        plantwild_config:
            Training config dictionary.
    """
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
            if key != "model_state_dict" and key != "optimizer_state_dict"
        },
        "plantwild_training_config": plantwild_config,
    }

    torch.save(checkpoint, output_path)


def maybe_save_best(
    model,
    optimizer,
    output_path,
    source_checkpoint_path,
    source_checkpoint,
    expanded_class_names,
    model_name,
    loss_name,
    run_name,
    epoch,
    stage,
    current_metrics,
    best_metrics,
    plantwild_config,
):
    """
    Save checkpoint if current score is best.

    Args:
        current_metrics:
            Current metrics dictionary.
        best_metrics:
            Best metrics so far.

    Returns:
        updated_best_metrics, improved.
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
    model,
    model_name,
    train_loader,
    plantwild_val_loader,
    plantdoc_val_loader,
    plantvillage_val_loader,
    criterion,
    stage_name,
    epochs,
    learning_rate,
    weight_decay,
    optimizer_name,
    start_epoch,
    best_metrics,
    save_context,
    plantwild_config,
    device,
):
    """
    Run one PlantWild-expanded training stage.

    Args:
        model:
            PyTorch model.
        model_name:
            Architecture name.
        train_loader:
            Mixed 80/12/8 training loader.
        plantwild_val_loader:
            PlantWild validation loader.
        plantdoc_val_loader:
            PlantDoc validation loader.
        plantvillage_val_loader:
            PlantVillage validation loader.
        criterion:
            Loss function.
        stage_name:
            "head" or "full".
        epochs:
            Number of epochs.
        learning_rate:
            Learning rate.
        weight_decay:
            Weight decay.
        optimizer_name:
            Optimizer name.
        start_epoch:
            Starting global epoch.
        best_metrics:
            Best metrics so far.
        save_context:
            Save arguments.
        plantwild_config:
            PlantWild config dictionary.
        device:
            CPU or CUDA device.

    Returns:
        current_epoch, best_metrics.
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
        raise ValueError(f"Unsupported stage: {stage_name}")
    
    optimizer = create_stage_optimizer(
        model=model,
        stage_name=stage_name,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        optimizer_name=optimizer_name,
    )

    current_epoch = start_epoch

    for _ in range(epochs):
        current_epoch+=1

        train_loss, train_accuracy = train_one_epoch(
            model=model,
            train_loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            epoch=current_epoch,
        )

        mlflow.log_metric("train_loss", train_loss, step = current_epoch)
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
    checkpoint_path,
    expanded_class_names,
    device,
):
    """
    Build expanded model and load PlantDoc-finetuned checkpoint into it.

    Args:
        checkpoint_path:
            PlantDoc-finetuned checkpoint path.
        expanded_class_names:
            Expanded class list.
        device:
            CPU or CUDA device.

    Returns:
        model, checkpoint, model_name, loss_name, run_name.
    """
    checkpoint = load_checkpoint(checkpoint_path)

    old_class_names = checkpoint["class_names"]

    validate_class_order(
        old_class_names=old_class_names,
        expanded_class_names=expanded_class_names
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

    model = model.to(device)

    print("\nCheckpoint expansion:")
    print(f"  Source:          {checkpoint_path.name}")
    print(f"  Model:           {model_name}")
    print(f"  Loss:            {loss_name}")
    print(f"  Old classes:     {len(old_class_names)}")
    print(f"  Expanded classes:{len(expanded_class_names)}")
    print(f"  Direct tensors:  {len(loaded_keys)}")
    print(f"  Expanded tensors:{expanded_keys}")

    return model, checkpoint, model_name, loss_name, run_name


def train_one_checkpoint(
    checkpoint_path,
    loaders,
    expanded_class_names,
    plantwild_config,
    device,
):
    """
    Train one PlantWild-expanded model.

    Args:
        checkpoint_path:
            PlantDoc-finetuned checkpoint.
        loaders:
            Dataloader dictionary.
        expanded_class_names:
            Expanded class list.
        plantwild_config:
            Training config.
        device:
            CPU or CUDA device.

    Returns:
        Summary dictionary.
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
            epochs=plantwild_config["head_epochs"],
            learning_rate=plantwild_config["lr_head"],
            weight_decay=plantwild_config["weight_decay"],
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
            epochs=plantwild_config["full_epochs"],
            learning_rate=plantwild_config["lr_full"],
            weight_decay=plantwild_config["weight_decay"],
            optimizer_name=plantwild_config["optimizer"],
            start_epoch=epoch,
            best_metrics=best_metrics,
            save_context=save_context,
            plantwild_config=plantwild_config,
            device=device,
        )

        for metric_name, metric_value in best_metrics.items():
            mlflow.log_metric(f"best_{metric_name}", metric_value)

    return {
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


def main():
    """
    Train all PlantWild-expanded models.
    """
    config = load_config(CONFIG_PATH)
    plantwild_config = get_plantwild_training_config(config)

    set_seed(config["project"]["seed"])

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
        batch_size=plantwild_config["batch_size"],
        num_workers=plantwild_config["num_workers"],
        plantwild_ratio=plantwild_config["plantwild_ratio"],
        plantdoc_ratio=plantwild_config["plantdoc_ratio"],
        plantvillage_ratio=plantwild_config["plantvillage_ratio"],
        plantdoc_val_ratio=plantwild_config["plantdoc_val_ratio"],
        seed=config["project"]["seed"],
    )

    source_checkpoints = discover_source_checkpoints(
        plantwild_config["checkpoint"]
    )

    print("\nPlantWild-expanded training setup:")
    print(f"  Source checkpoints: {len(source_checkpoints)}")
    print(f"  Expanded classes:   {len(expanded_class_names)}")
    print(f"  Loader metadata:    {loader_metadata}")

    loaders = {
        "mixed_train_loader": mixed_train_loader,
        "plantwild_val_loader": plantwild_val_loader,
        "plantdoc_val_loader": plantdoc_val_loader,
        "plantvillage_val_loader": plantvillage_val_loader,
        "plantwild_train_dataset": mixed_train_loader.dataset.datasets[0],
        "plantdoc_train_subset": mixed_train_loader.dataset.datasets[1],
        "plantvillage_train_dataset": mixed_train_loader.dataset.datasets[2],
    }

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

        if device.type == "cuda":
            torch.cuda.empty_cache()

    summary_df = pd.DataFrame(rows).sort_values(
        by="best_selection_score",
        ascending=False,
    )

    summary_path = output_results_dir / "plantwild_training_summary.csv"
    summary_df.to_csv(summary_path, index=False)

    print("\nPlantWild-expanded training complete.")
    print(f"Summary saved to: {summary_path}")
    print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()