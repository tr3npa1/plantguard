"""
Fine-tune PlantVillage-trained PlantGuard checkpoints on PlantDoc.

This script performs PlantDoc domain adaptation.

Pipeline:
1. Load one or more existing PlantVillage-trained best checkpoints from models/.
2. Build the same architecture used by the checkpoint.
3. Load checkpoint weights.
4. Fine-tune on PlantDoc train-adapt split.
5. Validate on PlantDoc val-adapt split.
6. Save the best PlantDoc-adapted checkpoint by validation macro F1.
7. Log parameters and metrics to MLflow.
8. Save a summary CSV under evaluation_results/.

Important:
    This script does NOT perform final testing.

Final testing should be done separately on:
    - PlantDoc test split
    - PlantVillage test split

That separate evaluation tells us:
    - whether PlantDoc fine-tuning improved real-world performance
    - whether PlantDoc fine-tuning caused catastrophic forgetting on PlantVillage
"""

import sys
from pathlib import Path

import mlflow
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import f1_score

# ---------------------------------------------------------------------
# Project setup
# ---------------------------------------------------------------------

# Resolve project root:
# training/finetune_plantdoc.py -> training/ -> project root
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Allow imports like `from data.dataset import ...`
sys.path.append(str(PROJECT_ROOT))

from data.dataset import get_plantdoc_finetune_loaders  # noqa: E402
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
from training.evaluate import count_parameters, load_checkpoint  # noqa: E402


MODELS_DIR = PROJECT_ROOT / "models"
MLFLOW_DB_PATH = PROJECT_ROOT / "mlflow.db"


# ---------------------------------------------------------------------
# Small path/config helpers
# ---------------------------------------------------------------------

def resolve_project_path(path_value):
    """
    Convert a config path into an absolute Path.

    Args:
        path_value:
            Path string from config. It can be relative to the project root
            or already absolute.

    Returns:
        Absolute pathlib.Path object.
    """
    path = Path(path_value)

    if path.is_absolute():
        return path

    return PROJECT_ROOT / path


def discover_checkpoints(checkpoint_from_config):
    """
    Find checkpoint files to fine-tune.

    Args:
        checkpoint_from_config:
            If None, all original models/*_best_model.pth checkpoints are used.
            If provided, only that checkpoint is used.

    Returns:
        Sorted list of checkpoint paths.
    """
    if checkpoint_from_config is not None:
        return [resolve_project_path(checkpoint_from_config)]

    checkpoint_paths = sorted(
        checkpoint_path
        for checkpoint_path in MODELS_DIR.glob("*_best_model.pth")
        if checkpoint_path.parent == MODELS_DIR
        and "plantdoc_finetuned" not in checkpoint_path.name
    )

    return checkpoint_paths


def get_checkpoint_identity(checkpoint):
    """
    Read model identity information from a saved PlantVillage checkpoint.

    Args:
        checkpoint:
            Loaded checkpoint dictionary.

    Returns:
        run_name:
            MLflow run name used during original PlantVillage training.
        model_name:
            Architecture name, e.g. efficientnet_b3.
        loss_name:
            Loss function name, e.g. weighted_cross_entropy.
    """
    source_config = checkpoint["config"]

    run_name = source_config["mlflow"]["run_name"]
    model_name = source_config["model"]["name"]
    loss_name = source_config["training"]["loss"]

    return run_name, model_name, loss_name


def get_checkpoint_focal_gamma(checkpoint, default=2.0):
    """
    Read focal-loss gamma from the source checkpoint config.

    Args:
        checkpoint:
            Loaded checkpoint dictionary.
        default:
            Fallback gamma if the checkpoint config does not contain it.

    Returns:
        Focal-loss gamma as float.
    """
    source_config = checkpoint.get("config", {})
    training_config = source_config.get("training", {})

    return float(training_config.get("focal_gamma", default))


# ---------------------------------------------------------------------
# Loss helpers
# ---------------------------------------------------------------------

def get_subset_labels(subset):
    """
    Extract labels from a torch.utils.data.Subset.

    PlantDoc fine-tuning uses a Subset of PlantDocEvaluationDataset.
    The base dataset stores samples as:
        (image_path, label, plantdoc_class_name, plantvillage_class_name)

    Args:
        subset:
            torch.utils.data.Subset object.

    Returns:
        List of integer labels.
    """
    labels = []

    for original_index in subset.indices:
        sample = subset.dataset.samples[original_index]
        label = sample[1]
        labels.append(label)

    return labels


def compute_subset_class_weights(subset, num_classes, device):
    """
    Compute inverse-frequency class weights from PlantDoc train-adapt subset.

    Classes absent from PlantDoc train-adapt get weight 0. That is fine because
    they do not appear as target labels during fine-tuning.

    Args:
        subset:
            PlantDoc train-adapt subset.
        num_classes:
            Total number of output classes in the model.
        device:
            CPU or CUDA device.

    Returns:
        Class-weight tensor with shape [num_classes].
    """
    labels = torch.tensor(get_subset_labels(subset), dtype=torch.long)

    counts = torch.bincount(labels, minlength=num_classes).float()
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
    loss_name,
    train_subset,
    num_classes,
    device,
    focal_gamma,
):
    """
    Create the fine-tuning loss function.

    The loss type follows the original checkpoint's training loss:
        - cross_entropy
        - weighted_cross_entropy
        - focal_loss

    Args:
        loss_name:
            Loss name from original checkpoint config.
        train_subset:
            PlantDoc train-adapt subset.
        num_classes:
            Total number of output classes.
        device:
            CPU or CUDA device.
        focal_gamma:
            Gamma value for focal loss.

    Returns:
        PyTorch loss function.
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

    raise ValueError(f"Unsupported loss: {loss_name}")


# ---------------------------------------------------------------------
# Freezing / unfreezing helpers
# ---------------------------------------------------------------------

def freeze_backbone(model, model_name):
    """
    Freeze feature extractor and train only the classifier head.

    This is used in Stage 1 of fine-tuning. It adapts the final classifier
    to PlantDoc without immediately changing the whole feature extractor.

    Args:
        model:
            PyTorch model.
        model_name:
            Architecture name from checkpoint config.
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


def unfreeze_model(model):
    """
    Unfreeze the full model for low-learning-rate fine-tuning.

    This is used in Stage 2 after classifier-head warmup.
    """
    for parameter in model.parameters():
        parameter.requires_grad = True


# ---------------------------------------------------------------------
# Validation / logging helpers
# ---------------------------------------------------------------------

@torch.no_grad()
def validate_with_f1(model, val_loader, criterion, device):
    """
    Evaluate model on PlantDoc val-adapt split.

    Args:
        model:
            PyTorch model.
        val_loader:
            DataLoader for PlantDoc validation subset.
        criterion:
            Loss function.
        device:
            CPU or CUDA device.

    Returns:
        Dictionary with validation loss, accuracy, macro F1, and weighted F1.
    """
    model.eval()

    total_loss = 0.0
    correct = 0
    total = 0

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
        correct += (predictions == labels).sum().item()
        total += batch_size

        all_labels.extend(labels.cpu().tolist())
        all_predictions.extend(predictions.cpu().tolist())

    return {
        "val_loss": total_loss / total,
        "val_accuracy": correct / total,
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


def log_epoch_metrics(train_loss, train_accuracy, val_metrics, epoch):
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


# ---------------------------------------------------------------------
# Checkpoint saving
# ---------------------------------------------------------------------

def save_finetuned_checkpoint(
    output_path,
    model,
    optimizer,
    source_checkpoint,
    source_checkpoint_path,
    run_name,
    model_name,
    loss_name,
    class_names,
    best_metrics,
    epoch,
    stage,
    finetune_config,
):
    """
    Save the best PlantDoc-adapted checkpoint.

    Args:
        output_path:
            Where to save the fine-tuned checkpoint.
        model:
            Fine-tuned PyTorch model.
        optimizer:
            Optimizer state at save time.
        source_checkpoint:
            Original PlantVillage checkpoint dictionary.
        source_checkpoint_path:
            Path to original PlantVillage checkpoint.
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
            Fine-tuning epoch where this checkpoint was saved.
        stage:
            "head" or "full".
        finetune_config:
            PlantDoc fine-tuning config dictionary.
    """
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
    model,
    optimizer,
    output_path,
    source_checkpoint,
    source_checkpoint_path,
    run_name,
    model_name,
    loss_name,
    class_names,
    current_metrics,
    best_metrics,
    epoch,
    stage,
    finetune_config,
):
    """
    Save checkpoint if current validation macro F1 is the best so far.

    Args:
        current_metrics:
            Metrics from the current validation epoch.
        best_metrics:
            Best metrics seen so far.

    Returns:
        updated_best_metrics:
            Best metrics after checking current epoch.
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


# ---------------------------------------------------------------------
# Model loading and training stages
# ---------------------------------------------------------------------

def load_model_for_finetuning(checkpoint_path, class_names, device):
    """
    Load a PlantVillage-trained checkpoint for PlantDoc fine-tuning.

    Args:
        checkpoint_path:
            Path to original PlantVillage checkpoint.
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
    model = model.to(device)

    return model, checkpoint, run_name, model_name, loss_name


def create_stage_optimizer(
    model,
    stage_name,
    learning_rate,
    weight_decay,
    optimizer_name,
):
    """
    Create optimizer for one fine-tuning stage.

    Args:
        model:
            PyTorch model.
        stage_name:
            "head" or "full".
        learning_rate:
            Learning rate for this stage.
        weight_decay:
            Weight decay for this stage.
        optimizer_name:
            Optimizer name from config.

    Returns:
        PyTorch optimizer.
    """
    if stage_name == "head":
        # For head-only training, optimize only trainable parameters.
        return optim.AdamW(
            filter(lambda parameter: parameter.requires_grad, model.parameters()),
            lr=learning_rate,
            weight_decay=weight_decay,
        )

    if stage_name == "full":
        # For full fine-tuning, reuse the project-level optimizer helper.
        return create_optimizer(
            model=model,
            optimizer_name=optimizer_name,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
        )

    raise ValueError(f"Unsupported stage: {stage_name}")


def run_training_stage(
    model,
    model_name,
    train_loader,
    val_loader,
    criterion,
    stage_name,
    epochs,
    learning_rate,
    weight_decay,
    optimizer_name,
    start_epoch,
    best_metrics,
    save_context,
    device,
):
    """
    Run one fine-tuning stage.

    Stage options:
        - "head": freeze backbone, train classifier head only
        - "full": unfreeze all layers, train whole model with low LR

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
            Global epoch count before this stage.
        best_metrics:
            Best validation metrics so far.
        save_context:
            Static arguments needed by maybe_save_best.
        device:
            CPU or CUDA device.

    Returns:
        current_epoch:
            Last global epoch number after this stage.
        best_metrics:
            Updated best metrics.
        stage_best_epoch:
            Epoch of best checkpoint inside this stage, or None.
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
        raise ValueError(f"Unsupported stage: {stage_name}")

    optimizer = create_stage_optimizer(
        model=model,
        stage_name=stage_name,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        optimizer_name=optimizer_name,
    )

    best_epoch_in_stage = None
    best_stage_in_stage = None
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
            best_epoch_in_stage = current_epoch
            best_stage_in_stage = stage_name

    return current_epoch, best_metrics, best_epoch_in_stage, best_stage_in_stage


# ---------------------------------------------------------------------
# One-checkpoint fine-tuning
# ---------------------------------------------------------------------

def finetune_checkpoint(
    checkpoint_path,
    class_names,
    train_loader,
    val_loader,
    train_subset,
    finetune_config,
    device,
):
    """
    Fine-tune one checkpoint on PlantDoc.

    Args:
        checkpoint_path:
            Path to original PlantVillage best checkpoint.
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
        Summary dictionary for this checkpoint.
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
        best_epoch = None
        best_stage = None

        epoch, best_metrics, stage_best_epoch, stage_best_stage = run_training_stage(
            model=model,
            model_name=model_name,
            train_loader=train_loader,
            val_loader=val_loader,
            criterion=criterion,
            stage_name="head",
            epochs=finetune_config["head_epochs"],
            learning_rate=finetune_config["lr_head"],
            weight_decay=finetune_config["weight_decay"],
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
            epochs=finetune_config["full_epochs"],
            learning_rate=finetune_config["lr_full"],
            weight_decay=finetune_config["weight_decay"],
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

    return {
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


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main():
    """
    Run PlantDoc fine-tuning for one checkpoint or all checkpoints.

    Configuration comes from:
        training/config.yaml -> plantdoc_finetuning section
    """
    config = load_config(CONFIG_PATH)
    finetune_config = config["plantdoc_finetuning"]

    set_seed(config["project"]["seed"])

    device = get_device(config["training"]["device"])
    print(f"Using device: {device}")

    output_dir = resolve_project_path(finetune_config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    mlflow.set_tracking_uri(f"sqlite:///{MLFLOW_DB_PATH.as_posix()}")
    mlflow.set_experiment(finetune_config["experiment_name"])

    checkpoint_paths = discover_checkpoints(finetune_config["checkpoint"])

    if not checkpoint_paths:
        raise FileNotFoundError("No checkpoints found for PlantDoc fine-tuning.")

    print("\nCheckpoints to fine-tune:")
    for checkpoint_path in checkpoint_paths:
        print(f"- {checkpoint_path.name}")

    first_checkpoint = load_checkpoint(checkpoint_paths[0])
    class_names = first_checkpoint["class_names"]

    (
        train_loader,
        val_loader,
        _test_loader,
        train_subset,
        val_subset,
        test_dataset,
    ) = get_plantdoc_finetune_loaders(
        class_names=class_names,
        batch_size=finetune_config["batch_size"],
        num_workers=finetune_config["num_workers"],
        val_ratio=finetune_config["val_ratio"],
        seed=config["project"]["seed"],
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

        if device.type == "cuda":
            torch.cuda.empty_cache()

    summary_df = pd.DataFrame(rows).sort_values(
        by="best_val_macro_f1",
        ascending=False,
    )

    summary_path = output_dir / "plantdoc_finetuning_summary.csv"
    summary_df.to_csv(summary_path, index=False)

    print("\nPlantDoc fine-tuning complete.")
    print(f"Summary saved to: {summary_path}")
    print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()