"""
Evaluation script for the PlantGuard project.

This script evaluates all saved best model checkpoints on:
1. PlantVillage test split
2. PlantDoc external evaluation dataset

For each checkpoint, it:
- loads the saved checkpoint
- reads the saved training config
- rebuilds the correct architecture
- loads trained weights
- evaluates on PlantVillage test
- evaluates on PlantDoc
- computes overall and per-class metrics
- saves CSV reports and confusion matrices
- logs metrics and artifacts to MLflow
"""

from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]

MODELS_DIR = PROJECT_ROOT / "models"
OUTPUT_DIR = PROJECT_ROOT / "evaluation_results"
MLFLOW_DB_PATH = PROJECT_ROOT / "mlflow.db"

EVAL_BATCH_SIZE = 32
NUM_WORKERS = 4
MLFLOW_EXPERIMENT_NAME = "PlantGuard_Evaluation"

sys.path.append(str(PROJECT_ROOT))

from data.dataset import get_dataloaders, get_plantdoc_loader # noqa: E402
from training.train import build_model

def get_device():
    """
    Return CUDA if available, otherwise CPU
    """
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_checkpoint(checkpoint_path):
    """
    Load a PyTorch checkpoint safely.

    Args:
        checkpoint_path: Path to checkpoint file.

    Returns:
        checkpoint dictionary.
    """
    try:
        return torch.load(
            checkpoint_path,
            map_location='cpu',
            weights_only=False,
        )
    except TypeError:
        return torch.load(
            checkpoint_path,
            map_location='cpu',

        )
    

def find_checkpoints():
    """
    Find all best model checkpoints inside the models directory.

    Returns:
        Sorted list of checkpoint paths.
    """
    checkpoint_paths = sorted(MODELS_DIR.glob("*_best_model.pth"))

    if not checkpoint_paths:
        raise FileNotFoundError(
            f"No best checkpoints found in {MODELS_DIR}. "
            "Expected files like efficientnet_b0_cross_entropy_best_model.pth."
        )

    return checkpoint_paths


def count_parameters(model):
    """
    Count total and trainable parameters in a model.

    Args:
        model: PyTorch model.

    Returns:
        total_parameters, trainable_parameters.
    """
    total_parameters = sum(parameter.numel() for parameter in model.parameters())

    trainable_parameters = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )

    return total_parameters, trainable_parameters


def load_checkpoint_model(checkpoint_path, num_classes, class_names, device):
    """
    Load one checkpoint, rebuild its architecture, and load saved weights.

    Args:
        checkpoint_path: Path to checkpoint file.
        num_classes: Number of output classes.
        class_names: Current PlantVillage class names in label-index order.
        device: CPU or CUDA device.

    Returns:
        model: Loaded PyTorch model.
        config: Saved training config from checkpoint.
        checkpoint: Full checkpoint dictionary.
    """
    checkpoint = load_checkpoint(checkpoint_path)

    config = checkpoint['config']
    checkpoint_class_names = checkpoint["class_names"]

    if list(checkpoint_class_names) != list(class_names):
        raise ValueError(
            f"Class name mismatch for checkpoint {checkpoint_path.name}. "
            "Checkpoint class order does not match current dataset class order."
        )

    model_name = config["model"]["name"]

    model = build_model(
        model_name = model_name,
        num_classes=num_classes,
        pretrained=False,
    )

    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()

    return model, config, checkpoint


def run_inference(model, dataloader, device):
    """
    Run inference on a full dataloader.

    Args:
        model: PyTorch model.
        dataloader: DataLoader returning image-label batches.
        device: CPU or CUDA device.

    Returns:
        y_true: Ground-truth labels as numpy array.
        y_pred: Predicted labels as numpy array.
    """
    all_labels = []
    all_predictions = []

    model.eval()

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)

            output = model(images)
            predictions = output.argmax(dim=1)

            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predictions.cpu().numpy())

    y_true = np.array(all_labels)
    y_pred = np.array(all_predictions)

    return y_true, y_pred


def get_present_label_indices(y_true):
    """
    Return sorted label indices that are present in y_true.

    Args:
        y_true: Ground-truth labels.

    Returns:
        Sorted list of present label indices.
    """
    return sorted(np.unique(y_true).astype(int).tolist())


def compute_metrics(y_true, y_pred, class_names, average_label_indices):
    """
    Compute overall and per-class classification metrics.

    Args:
        y_true: Ground-truth labels.
        y_pred: Predicted labels.
        class_names: Full PlantVillage class names in label-index order.
        average_label_indices: Labels to include in macro/weighted averages.

    Returns:
        summary_metrics: Dictionary of overall metrics.
        per_class_df: Per-class precision/recall/F1/support dataframe.
        report_all_df: Classification report over all classes.
        report_present_df: Classification report over present target classes.
        cm: Full confusion matrix over all classes.
        prediction_distribution_df: True/predicted count per class.
    """
    all_label_indices = list(range(len(class_names)))

    accuracy = accuracy_score(y_true,y_pred)
    balanced_accuracy = balanced_accuracy_score(y_true,y_pred)
    
    macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=average_label_indices,
        average="macro",
        zero_division=0,
    )

    weighted_precision, weighted_recall, weighted_f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=average_label_indices,
        average="weighted",
        zero_division=0,
    )

    per_class_precision, per_class_recall, per_class_f1, per_class_support =(
        precision_recall_fscore_support(
            y_true,
            y_pred,
            labels=all_label_indices,
            average=None,
            zero_division=0,
        )
    )
    
    true_counts = np.bincount(
        y_true,
        minlength=len(class_names),
    )

    predicted_counts = np.bincount(
        y_pred,
        minlength=len(class_names),
    )

    per_class_df = pd.DataFrame(
        {
            "class_index": all_label_indices,
            "class_name": class_names,
            "support": per_class_support,
            "precision": per_class_precision,
            "recall": per_class_recall,
            "f1_score": per_class_f1,
            "is_present_in_ground_truth": per_class_support > 0,
            "predicted_count": predicted_counts,
        }
    )

    report_all_dict = classification_report(
        y_true,
        y_pred,
        labels=all_label_indices,
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )

    present_class_names = [
        class_names[label_index]
        for label_index in average_label_indices
    ]

    report_present_dict = classification_report(
        y_true,
        y_pred,
        labels = average_label_indices,
        target_names=present_class_names,
        output_dict=True,
        zero_division=0,
    )

    report_all_df = pd.DataFrame(report_all_dict).transpose()
    report_present_df = pd.DataFrame(report_present_dict).transpose()

    cm = confusion_matrix(
        y_true,
        y_pred,
        labels = all_label_indices,
    )

    prediction_distribution_df = pd.DataFrame(
        {
            "class_index": all_label_indices,
            "class_name": class_names,
            "true_count": true_counts,
            "predicted_count": predicted_counts,
            "is_present_in_ground_truth": true_counts > 0,
        }
    )

    summary_metrics = {
        "accuracy": float(accuracy),
        "balanced_accuracy": float(balanced_accuracy),
        "macro_precision": float(macro_precision),
        "macro_recall": float(macro_recall),
        "macro_f1": float(macro_f1),
        "weighted_precision": float(weighted_precision),
        "weighted_recall": float(weighted_recall),
        "weighted_f1": float(weighted_f1),
        "num_samples": int(len(y_true)),
        "num_present_classes": int(len(average_label_indices)),
    }

    return (
        summary_metrics,
        per_class_df,
        report_all_df,
        report_present_df,
        cm,
        prediction_distribution_df,
    )


def save_confusion_matrix(cm, class_names, output_path, normalize=False):
    """
    Save a confusion matrix plot.

    Args:
        cm: Confusion matrix.
        class_names: Class names in label-index order.
        output_path: Path where PNG should be saved.
        normalize: Whether to normalize each row by true-class count.
    """
    if normalize:
        row_sums = cm.sum(axis=1, keepdims=True)
        matrix = np.divide(
            cm,
            row_sums,
            out = np.zeros_like(cm,dtype=float),
            where=row_sums!=0,
        )
        title = "Normalized confusion matrix"

    else:
        matrix = cm
        title = "Confusion matrix"

    fig, ax = plt.subplots(figsize=(24,20))

    image = ax.imshow(matrix, interpolation="nearest")
    ax.figure.colorbar(image, ax=ax)

    ax.set_title(title)
    ax.set_xlabel("Predicted class")
    ax.set_ylabel("True class")

    tick_marks = np.arange(len(class_names))

    ax.set_xticks(tick_marks)
    ax.set_yticks(tick_marks)

    ax.set_xticklabels(class_names, rotation=90, fontsize=6)
    ax.set_yticklabels(class_names, fontsize=6)

    fig.tight_layout()

    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def save_top_confusions(cm, class_names, output_path, top_k=25):
    """
    Save the most common off-diagonal confusions.

    Args:
        cm: Confusion matrix.
        class_names: Class names in label-index order.
        output_path: CSV output path.
        top_k: Number of confusion pairs to save.
    """
    rows = []

    for true_index in range(cm.shape[0]):
        true_support = int(cm[true_index].sum())

        for predicted_index in range(cm.shape[1]):
            if true_index==predicted_index:
                continue

            count = int(cm[true_index,predicted_index])

            if count<=0:
                continue

            rows.append(
                 {
                    "true_class_index": true_index,
                    "true_class": class_names[true_index],
                    "predicted_class_index": predicted_index,
                    "predicted_class": class_names[predicted_index],
                    "count": count,
                    "true_class_support": true_support,
                    "percent_of_true_class": (
                        count / true_support if true_support > 0 else 0.0
                    ),
                }
            )
    
    columns = [
        "true_class_index",
        "true_class",
        "predicted_class_index",
        "predicted_class",
        "count",
        "true_class_support",
        "percent_of_true_class",
    ]

    top_confusions_df = pd.DataFrame(rows, columns=columns)

    if not top_confusions_df.empty:
        top_confusions_df = top_confusions_df.sort_values(
            by = ["count", "percent_of_true_class"],
            ascending=False,
        ).head(top_k)

    top_confusions_df.to_csv(output_path, index = False)


def save_dataset_evaluation_artifacts(
    dataset_name,
    run_output_dir,
    class_names,
    summary_metrics,
    per_class_df,
    report_all_df,
    report_present_df,
    cm,
    prediction_distribution_df,
):
    """
    Save all CSV and PNG artifacts for one dataset evaluation.

    Args:
        dataset_name: Name such as plantvillage_test or plantdoc.
        run_output_dir: Output directory for this model run.
        class_names: Class names in label-index order.
        summary_metrics: Overall metrics dictionary.
        per_class_df: Per-class metrics dataframe.
        report_all_df: Classification report over all 38 classes.
        report_present_df: Classification report over present true classes.
        cm: Full confusion matrix.
        prediction_distribution_df: True/predicted count dataframe.

    Returns:
        artifact_paths: Dictionary of artifact names to paths.
    """
    dataset_output_dir = run_output_dir / dataset_name
    dataset_output_dir.mkdir(parents=True, exist_ok=True)

    per_class_path = dataset_output_dir / "per_class_metrics.csv"
    report_all_path = dataset_output_dir / "classification_report_all_classes.csv"
    report_present_path = dataset_output_dir / "classification_report_present_classes.csv"
    cm_csv_path = dataset_output_dir / "confusion_matrix.csv"
    cm_png_path = dataset_output_dir / "confusion_matrix.png"
    cm_norm_png_path = dataset_output_dir / "confusion_matrix_normalized.png"
    top_confusions_path = dataset_output_dir / "top_confusions.csv"
    prediction_distribution_path = dataset_output_dir / "prediction_distribution.csv"
    summary_path = dataset_output_dir / "summary_metrics.csv"

    per_class_df.to_csv(per_class_path, index=False)
    report_all_df.to_csv(report_all_path)
    report_present_df.to_csv(report_present_path)

    cm_df = pd.DataFrame(
        cm,
        index = class_names,
        columns = class_names,
    )
    cm_df.to_csv(cm_csv_path)

    prediction_distribution_df.to_csv(prediction_distribution_path, index = False)

    summary_df = pd.DataFrame([summary_metrics])
    summary_df.to_csv(summary_path, index=False)

    save_confusion_matrix(
        cm=cm,
        class_names = class_names,
        output_path=cm_png_path,
        normalize=False,
    )

    save_confusion_matrix(
        cm=cm,
        class_names=class_names,
        output_path=cm_norm_png_path,
        normalize=True,
    )

    save_top_confusions(
        cm=cm,
        class_names=class_names,
        output_path=top_confusions_path,
        top_k=25,
    )

    artifact_paths = {
        "per_class_metrics": per_class_path,
        "classification_report_all_classes": report_all_path,
        "classification_report_present_classes": report_present_path,
        "confusion_matrix_csv": cm_csv_path,
        "confusion_matrix_png": cm_png_path,
        "confusion_matrix_normalized_png": cm_norm_png_path,
        "top_confusions": top_confusions_path,
        "prediction_distribution": prediction_distribution_path,
        "summary_metrics": summary_path,
    }

    return artifact_paths


def evaluate_dataset(
    model,
    dataloader,
    dataset_name,
    class_names,
    output_dir,
    average_label_indices,
    device,
):
    """
    Evaluate one model on one dataset.

    Args:
        model: PyTorch model.
        dataloader: Dataset DataLoader.
        dataset_name: Name used for prefixes and output folders.
        class_names: Full PlantVillage class names in label-index order.
        output_dir: Output directory for this model run.
        average_label_indices: Labels to include in macro/weighted averages.
        device: CPU or CUDA device.

    Returns:
        summary_metrics: Overall metrics dictionary.
        artifact_paths: Saved artifact paths.
    """
    print(f"\nEvaluating dataset: {dataset_name}")

    y_true, y_pred = run_inference(
        model=model,
        dataloader=dataloader,
        device=device,
    )


    (
        summary_metrics,
        per_class_df,
        report_all_df,
        report_present_df,
        cm,
        prediction_distribution_df,
    ) = compute_metrics(
        y_true=y_true,
        y_pred=y_pred,
        class_names=class_names,
        average_label_indices=average_label_indices,
    )

    artifact_paths = save_dataset_evaluation_artifacts(
        dataset_name=dataset_name,
        run_output_dir=output_dir,
        class_names=class_names,
        summary_metrics=summary_metrics,
        per_class_df=per_class_df,
        report_all_df=report_all_df,
        report_present_df=report_present_df,
        cm=cm,
        prediction_distribution_df=prediction_distribution_df,
    )

    print(f"{dataset_name} accuracy: {summary_metrics['accuracy']:.6f}")
    print(f"{dataset_name} macro F1: {summary_metrics['macro_f1']:.6f}")
    print(f"{dataset_name} weighted F1: {summary_metrics['weighted_f1']:.6f}")

    return summary_metrics, artifact_paths


def prefix_metrics(metrics, prefix):
    """
    Prefix metric keys for MLflow and summary CSV.

    Args:
        metrics: Metrics dictionary.
        prefix: Prefix such as plantvillage_test or plantdoc.

    Returns:
        New dictionary with prefixed metric names.
    """
    return {
        f"{prefix}_{key}": value
        for key, value in metrics.items()
    }


def log_artifacts_to_mlflow(artifact_paths, artifact_root):
    """
    Log a dictionary of artifact files to MLflow.

    Args:
        artifact_paths: Dictionary of artifact names to paths.
        artifact_root: MLflow artifact folder.
    """
    for artifact_path in artifact_paths.values():
        mlflow.log_artifact(
            str(artifact_path),
            artifact_path=artifact_root,
        )


def evaluate_checkpoint(
    checkpoint_path,
    plantvillage_test_loader,
    plantdoc_loader,
    plantdoc_dataset,
    class_names,
    device,
):
    """
    Evaluate one checkpoint on PlantVillage test and PlantDoc.

    Args:
        checkpoint_path: Path to best model checkpoint.
        plantvillage_test_loader: PlantVillage test DataLoader.
        plantdoc_loader: PlantDoc external DataLoader.
        plantdoc_dataset: PlantDoc dataset object with metadata.
        class_names: PlantVillage class names in label-index order.
        device: CPU or CUDA device.

    Returns:
        summary_row: Dictionary for final comparison table.
    """
    print("\n" + "=" * 100)
    print(f"Evaluating checkpoint: {checkpoint_path.name}")
    print("=" * 100)

    model, config, checkpoint = load_checkpoint_model(
        checkpoint_path=checkpoint_path,
        num_classes=len(class_names),
        class_names = class_names,
        device = device,
    )

    run_name = config["mlflow"]["run_name"]
    model_name = config["model"]["name"]
    loss_name = config["training"]["loss"]

    total_parameters, trainable_parameters = count_parameters(model)

    checkpoint_size_mb = checkpoint_path.stat().st_size / (1024 * 1024)

    run_output_dir = OUTPUT_DIR / run_name
    run_output_dir.mkdir(parents=True, exist_ok=True)

    plantvillage_average_labels = list(range(len(class_names)))

    plantdoc_true_labels = [
        label 
        for _, label, _, _ in plantdoc_dataset.samples
    ]
    plantdoc_average_labels = sorted(set(plantdoc_true_labels))

    plantvillage_metrics, plantvillage_artifacts = evaluate_dataset(
        model=model,
        dataloader=plantvillage_test_loader,
        dataset_name="plantvillage_test",
        class_names=class_names,
        output_dir=run_output_dir,
        average_label_indices=plantvillage_average_labels,
        device=device,
    )

    plantdoc_metrics, plantdoc_artifacts = evaluate_dataset(
        model=model,
        dataloader=plantdoc_loader,
        dataset_name="plantdoc",
        class_names=class_names,
        output_dir=run_output_dir,
        average_label_indices=plantdoc_average_labels,
        device=device,
    )

    prefixed_plantvillage_metrics = prefix_metrics(
        plantvillage_metrics,
        "plantvillage_test",
    )

    prefixed_plantdoc_metrics = prefix_metrics(
        plantdoc_metrics,
        "plantdoc",
    )

    summary_row = {
        "run_name": run_name,
        "model_name": model_name,
        "loss_name": loss_name,
        "checkpoint": checkpoint_path.name,
        "checkpoint_size_mb": checkpoint_size_mb,
        "total_parameters": total_parameters,
        "trainable_parameters": trainable_parameters,
        "training_best_val_accuracy": float(checkpoint["best_val_accuracy"]),
        **prefixed_plantvillage_metrics,
        **prefixed_plantdoc_metrics,
    }

    with mlflow.start_run(run_name=f"{run_name}_evaluation"):
        mlflow.log_param("training_run_name", run_name)
        mlflow.log_param("source_checkpoint", checkpoint_path.name)
        mlflow.log_param("model_name", model_name)
        mlflow.log_param("loss_name", loss_name)
        mlflow.log_param("eval_batch_size", EVAL_BATCH_SIZE)
        mlflow.log_param("num_classes", len(class_names))
        mlflow.log_param("checkpoint_size_mb", checkpoint_size_mb)
        mlflow.log_param("total_parameters", total_parameters)
        mlflow.log_param("trainable_parameters", trainable_parameters)
        mlflow.log_param(
            "training_best_val_accuracy",
            float(checkpoint["best_val_accuracy"]),
        )

        mlflow.log_param(
            "plantdoc_samples",
            len(plantdoc_dataset),
        )
        mlflow.log_param(
            "plantdoc_skipped_classes",
            str(plantdoc_dataset.skipped_classes),
        )
        mlflow.log_param(
            "plantdoc_mapped_classes",
            len(plantdoc_dataset.mapped_class_counts),
        )

        mlflow.log_metrics(prefixed_plantvillage_metrics)
        mlflow.log_metrics(prefixed_plantdoc_metrics)

        log_artifacts_to_mlflow(
            artifact_paths=plantvillage_artifacts,
            artifact_root="evaluation/plantvillage_test",
        )

        log_artifacts_to_mlflow(
            artifact_paths=plantdoc_artifacts,
            artifact_root="evaluation/plantdoc",
        )

    return summary_row


def save_final_summary(summary_rows):
    """
    Save final model comparison CSV.

    Args:
        summary_rows: List of model evaluation summary dictionaries.

    Returns:
        summary_df: Final comparison dataframe.
    """
    summary_df = pd.DataFrame(summary_rows)

    sort_columns = [
        "plantdoc_macro_f1",
        "plantdoc_accuracy",
        "plantvillage_test_macro_f1",
        "plantvillage_test_accuracy",
    ]

    existing_sort_columns = [
        column
        for column in sort_columns
        if column in summary_df.columns
    ]
    if existing_sort_columns:
        summary_df = summary_df.sort_values(
            by=existing_sort_columns,
            ascending=False,
        )
    
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    summary_path = OUTPUT_DIR / "model_comparison_summary.csv"
    summary_df.to_csv(summary_path, index=False)

    print("\n" + "=" * 100)
    print("Final model comparison")
    print("=" * 100)

    display_columns = [
        "run_name",
        "model_name",
        "loss_name",
        "training_best_val_accuracy",
        "plantvillage_test_accuracy",
        "plantvillage_test_macro_f1",
        "plantvillage_test_weighted_f1",
        "plantdoc_accuracy",
        "plantdoc_macro_f1",
        "plantdoc_weighted_f1",
        "checkpoint_size_mb",
        "total_parameters",
    ]

    existing_display_columns = [
        column
        for column in display_columns
        if column in summary_df.columns
    ]

    print(summary_df[existing_display_columns].to_string(index=False))
    print(f"\nSaved final summary to: {summary_path}")

    return summary_df


def main():
    """
    Run evaluation for all saved best checkpoints.
    """

    device = get_device()
    print(f"Using device: {device}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("\nLoading PlantVillage test loader...")
    _, _, plantvillage_test_loader, class_names = get_dataloaders(
        batch_size=EVAL_BATCH_SIZE,
        num_workers=NUM_WORKERS,
    )

    print(f"PlantVillage classes: {len(class_names)}")
    print(f"PlantVillage test samples: {len(plantvillage_test_loader.dataset)}")

    print("\nLoading PlantDoc external evaluation loader...")
    plantdoc_loader, plantdoc_dataset = get_plantdoc_loader(
        class_names=class_names,
        batch_size=EVAL_BATCH_SIZE,
        num_workers=NUM_WORKERS,
    )

    print(f"PlantDoc samples: {len(plantdoc_dataset)}")
    print(f"PlantDoc batches: {len(plantdoc_loader)}")
    print(f"PlantDoc skipped classes: {plantdoc_dataset.skipped_classes}")

    checkpoint_paths = find_checkpoints()

    print("\nFound checkpoints:")
    for checkpoint_path in checkpoint_paths:
        print(f"- {checkpoint_path.name}")

    mlflow.set_tracking_uri(f"sqlite:///{MLFLOW_DB_PATH.as_posix()}")
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)

    summary_rows = []

    for checkpoint_path in checkpoint_paths:
        summary_row = evaluate_checkpoint(
            checkpoint_path=checkpoint_path,
            plantvillage_test_loader=plantvillage_test_loader,
            plantdoc_loader=plantdoc_loader,
            plantdoc_dataset=plantdoc_dataset,
            class_names=class_names,
            device=device,
        )

        summary_rows.append(summary_row)

    summary_df = save_final_summary(summary_rows)

    summary_path = OUTPUT_DIR / "model_comparison_summary.csv"

    with mlflow.start_run(run_name="all_models_evaluation_summary"):
        mlflow.log_param("num_evaluated_models", len(summary_df))
        mlflow.log_param("eval_batch_size", EVAL_BATCH_SIZE)
        mlflow.log_param("num_classes", len(class_names))
        mlflow.log_param("plantdoc_samples", len(plantdoc_dataset))
        mlflow.log_param("plantdoc_skipped_classes", str(plantdoc_dataset.skipped_classes))

        mlflow.log_artifact(
            str(summary_path),
            artifact_path="evaluation_summary",
        )


if __name__ == "__main__":
    main()