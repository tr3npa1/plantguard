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

EXPANDED_MODELS_DIR = PROJECT_ROOT / "models" / "plantwild_expanded"
OUTPUT_DIR = PROJECT_ROOT / "evaluation_results" / "final_expanded_evaluation"
MLFLOW_DB_PATH = PROJECT_ROOT / "mlflow.db"

EVAL_BATCH_SIZE = 32
NUM_WORKERS = 4
EXPECTED_NUM_CLASSES = 132

MLFLOW_EXPERIMENT_NAME = "PlantGuard_Final_Expanded_Evaluation"

FINAL_SCORE_WEIGHTS = {
    "plantwild_test_macro_f1": 0.50,
    "plantdoc_test_macro_f1": 0.20,
    "fieldplant_test_macro_f1": 0.20,
    "plantvillage_test_macro_f1": 0.10,
}

sys.path.append(str(PROJECT_ROOT))

from data.dataset import ( # noqa: E402
    get_dataloaders, 
    get_plantdoc_loader,
    get_fieldplant_loader,
    get_plantwild_loader,
    load_expanded_class_names,
    validate_expanded_class_order,
)
from training.train import build_model # noqa: E402

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
    checkpoint_paths = sorted(
        EXPANDED_MODELS_DIR.glob("*_plantwild_expanded_best_model.pth")
        )

    if not checkpoint_paths:
        raise FileNotFoundError(
            f"No PlantWild-expanded checkpoints found in {EXPANDED_MODELS_DIR}. "
            "Expected files like *_plantwild_expanded_best_model.pth."
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


def load_checkpoint_model(checkpoint_path, expanded_class_names, device):
    """
    Load one PlantWild-expanded checkpoint.

    Expanded checkpoints store model metadata directly in the checkpoint, not
    inside checkpoint['config'] like the older PlantVillage checkpoints.
    """
    checkpoint = load_checkpoint(checkpoint_path)

    required_keys = {
        "model_state_dict",
        "class_names",
        "model_name",
        "loss_name",
        "run_name",
    }

    missing_keys = required_keys - set(checkpoint.keys())

    if missing_keys:
        raise KeyError(
            f"Checkpoint {checkpoint_path.name} missing required keys: "
            f"{sorted(missing_keys)}"
        )

    checkpoint_class_names = checkpoint["class_names"]

    if list(checkpoint_class_names) != list(expanded_class_names):
        raise ValueError(
            f"Class-name mismatch for checkpoint {checkpoint_path.name}. "
            "Checkpoint class order does not match expanded class order."
        )

    model = build_model(
        model_name=checkpoint["model_name"],
        num_classes=len(expanded_class_names),
        pretrained=False,
    )

    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()

    return model, checkpoint


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


def compute_metrics(y_true, y_pred, class_names, average_label_indices=None):
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
    if average_label_indices is None:
        average_label_indices = get_present_label_indices(y_true)
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
            "true_count" : true_counts,
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

    fig_size = max(24, len(class_names)*0.22)
    fig, ax = plt.subplots(figsize=(fig_size,fig_size))

    image = ax.imshow(matrix, interpolation="nearest")
    ax.figure.colorbar(image, ax=ax)

    ax.set_title(title)
    ax.set_xlabel("Predicted class")
    ax.set_ylabel("True class")

    tick_marks = np.arange(len(class_names))

    ax.set_xticks(tick_marks)
    ax.set_yticks(tick_marks)

    ax.set_xticklabels(class_names, rotation=90, fontsize=4)
    ax.set_yticklabels(class_names, fontsize=4)

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


def get_dataset_sample_paths(dataset):
    """
    Extract image paths from dataset.samples when available.

    This supports:
        PlantDiseaseDataset samples: (path, label)
        CSVImageDataset samples: (path, label)
        PlantDocEvaluationDataset samples: (path, label, source_label, mapped_label)
    """
    if not hasattr(dataset, "samples"):
        return ["" for _ in range(len(dataset))]

    paths = []

    for sample in dataset.samples:
        image_path = sample[0]

        try:
            paths.append(str(Path(image_path).relative_to(PROJECT_ROOT).as_posix()))
        except ValueError:
            paths.append(str(image_path))

    return paths


def save_predictions_csv(dataset,y_true,y_pred,class_names,output_path):
    """
    Save per-image predictions for error analysis and GradCAM sampling.
    """
    image_paths = get_dataset_sample_paths(dataset)

    if len(image_paths) != len(y_true):
        image_paths = ["" for _ in range(len(y_true))]

    rows = []

    for image_path, true_index, predicted_index in zip(image_paths,y_true,y_pred):
        rows.append(
            {
                "image_path": image_path,
                "true_class_index": int(true_index),
                "true_class": class_names[int(true_index)],
                "predicted_class_index": int(predicted_index),
                "predicted_class": class_names[int(predicted_index)],
                "is_correct": bool(int(true_index) == int(predicted_index)),
            }
        )

    pd.DataFrame(rows).to_csv(output_path, index=False)


def save_dataset_evaluation_artifacts(
    dataset_name,
    run_output_dir,
    dataset,
    class_names,
    summary_metrics,
    per_class_df,
    report_all_df,
    report_present_df,
    cm,
    prediction_distribution_df,
    y_true,
    y_pred
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
    predictions_path = dataset_output_dir / "predictions.csv"

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

    save_predictions_csv(
        dataset=dataset,
        y_true=y_true,
        y_pred=y_pred,
        class_names=class_names,
        output_path=predictions_path,
    )

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
        "predictions" : predictions_path,
    }

    return artifact_paths


def evaluate_dataset(
    model,
    dataloader,
    dataset,
    dataset_name,
    class_names,
    output_dir,
    device,
    average_label_indices=None,
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
    
    if average_label_indices is None:
        average_label_indices = get_present_label_indices(y_true)

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
        dataset=dataset,
        class_names=class_names,
        summary_metrics=summary_metrics,
        per_class_df=per_class_df,
        report_all_df=report_all_df,
        report_present_df=report_present_df,
        cm=cm,
        prediction_distribution_df=prediction_distribution_df,
        y_true=y_true,
        y_pred=y_pred
    )

    print(f"{dataset_name} samples:     {summary_metrics['num_samples']}")
    print(f"{dataset_name} classes:     {summary_metrics['num_present_classes']}")
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


def get_checkpoint_metric(checkpoint, key):
    """
    Safely read numeric checkpoint metric.
    """
    value = checkpoint.get(key)

    if value is None:
        return None

    return float(value)


def compute_final_test_score(summary_row):
    """
    Compute weighted final test score from macro F1 values.
    """
    score = 0.0

    for metric_name, weight in FINAL_SCORE_WEIGHTS.items():
        score += weight * float(summary_row.get(metric_name, 0.0))

    return score


def prepare_final_test_loaders(expanded_class_names):
    """
    Create all final test loaders:
        PlantVillage test
        PlantDoc held-out test
        PlantWild test
        FieldPlant external test
    """
    print("\nLoading PlantVillage test loader...")
    _, _, plantvillage_test_loader, plantvillage_class_names = get_dataloaders(
        batch_size=EVAL_BATCH_SIZE,
        num_workers=NUM_WORKERS,
    )

    validate_expanded_class_order(
        expanded_class_names=expanded_class_names,
        original_class_names=plantvillage_class_names,
    )

    print(f"PlantVillage test samples: {len(plantvillage_test_loader.dataset)}")

    print("\nLoading PlantDoc held-out test loader...")
    plantdoc_test_loader, plantdoc_test_dataset = get_plantdoc_loader(
        class_names=expanded_class_names,
        batch_size=EVAL_BATCH_SIZE,
        num_workers=NUM_WORKERS,
        splits=("test",),
        transform_type="eval",
        shuffle=False,
    )

    print(f"PlantDoc test samples: {len(plantdoc_test_dataset)}")
    print(f"PlantDoc skipped classes: {plantdoc_test_dataset.skipped_classes}")

    print("\nLoading PlantWild test loader...")
    plantwild_test_loader, plantwild_test_dataset = get_plantwild_loader(
        split="test",
        batch_size=EVAL_BATCH_SIZE,
        num_workers=NUM_WORKERS,
        transform_type="eval",
        shuffle=False,
    )

    print(f"PlantWild test samples: {len(plantwild_test_dataset)}")

    print("\nLoading FieldPlant external test loader...")
    fieldplant_test_loader, fieldplant_test_dataset = get_fieldplant_loader(
        batch_size=EVAL_BATCH_SIZE,
        num_workers=NUM_WORKERS,
        transform_type="eval",
        shuffle=False,
    )

    print(f"FieldPlant test samples: {len(fieldplant_test_dataset)}")

    loaders = {
        "plantvillage_test": plantvillage_test_loader,
        "plantdoc_test": plantdoc_test_loader,
        "plantwild_test": plantwild_test_loader,
        "fieldplant_test": fieldplant_test_loader,
    }

    datasets = {
        "plantvillage_test": plantvillage_test_loader.dataset,
        "plantdoc_test": plantdoc_test_dataset,
        "plantwild_test": plantwild_test_dataset,
        "fieldplant_test": fieldplant_test_dataset,
    }

    return loaders, datasets


def evaluate_checkpoint(
    checkpoint_path,
    loaders,
    datasets,
    expanded_class_names,
    device,
):
    """
    Evaluate one PlantWild-expanded checkpoint on all final test datasets.
    """
    print("\n" + "=" * 100)
    print(f"Evaluating checkpoint: {checkpoint_path.name}")
    print("=" * 100)

    model, checkpoint = load_checkpoint_model(
        checkpoint_path=checkpoint_path,
        expanded_class_names=expanded_class_names,
        device=device,
    )

    run_name = checkpoint["run_name"]
    model_name = checkpoint["model_name"]
    loss_name = checkpoint["loss_name"]

    total_parameters, trainable_parameters = count_parameters(model)
    checkpoint_size_mb = checkpoint_path.stat().st_size / (1024 * 1024)

    run_output_dir = OUTPUT_DIR / run_name
    run_output_dir.mkdir(parents=True, exist_ok=True)

    dataset_results = {}

    for dataset_name, dataloader in loaders.items():
        dataset = datasets[dataset_name]

        metrics, artifact_paths = evaluate_dataset(
            model=model,
            dataloader=dataloader,
            dataset=dataset,
            dataset_name=dataset_name,
            class_names=expanded_class_names,
            output_dir=run_output_dir,
            device=device,
            average_label_indices=None,
        )

        dataset_results[dataset_name] = {
            "metrics": metrics,
            "artifacts": artifact_paths,
        }

    summary_row = {
        "run_name": run_name,
        "model_name": model_name,
        "loss_name": loss_name,
        "checkpoint": checkpoint_path.name,
        "checkpoint_size_mb": float(checkpoint_size_mb),
        "total_parameters": int(total_parameters),
        "trainable_parameters": int(trainable_parameters),
        "checkpoint_epoch": checkpoint.get("epoch"),
        "checkpoint_stage": checkpoint.get("stage"),
        "training_best_selection_score": get_checkpoint_metric(
            checkpoint,
            "best_selection_score",
        ),
        "training_best_plantwild_macro_f1": get_checkpoint_metric(
            checkpoint,
            "best_plantwild_macro_f1",
        ),
        "training_best_plantdoc_macro_f1": get_checkpoint_metric(
            checkpoint,
            "best_plantdoc_macro_f1",
        ),
        "training_best_plantvillage_macro_f1": get_checkpoint_metric(
            checkpoint,
            "best_plantvillage_macro_f1",
        ),
    }

    for dataset_name, result in dataset_results.items():
        summary_row.update(
            prefix_metrics(
                result["metrics"],
                dataset_name,
            )
        )

    summary_row["final_test_score"] = compute_final_test_score(summary_row)

    with mlflow.start_run(run_name=f"{run_name}_final_evaluation"):
        mlflow.log_param("source_checkpoint", checkpoint_path.name)
        mlflow.log_param("run_name", run_name)
        mlflow.log_param("model_name", model_name)
        mlflow.log_param("loss_name", loss_name)
        mlflow.log_param("eval_batch_size", EVAL_BATCH_SIZE)
        mlflow.log_param("num_classes", len(expanded_class_names))
        mlflow.log_param("checkpoint_size_mb", checkpoint_size_mb)
        mlflow.log_param("total_parameters", total_parameters)
        mlflow.log_param("trainable_parameters", trainable_parameters)

        for key, value in FINAL_SCORE_WEIGHTS.items():
            mlflow.log_param(f"final_score_weight.{key}", value)

        numeric_metrics = {
            key: value
            for key, value in summary_row.items()
            if isinstance(value, (int, float))
            and value is not None
        }

        mlflow.log_metrics(numeric_metrics)

        for dataset_name, result in dataset_results.items():
            log_artifacts_to_mlflow(
                artifact_paths=result["artifacts"],
                artifact_root=f"evaluation/{dataset_name}",
            )

    return summary_row


def save_final_summary(summary_rows):
    """
    Save final expanded model comparison CSV.
    """
    summary_df = pd.DataFrame(summary_rows)

    sort_columns = [
        "final_test_score",
        "plantwild_test_macro_f1",
        "fieldplant_test_macro_f1",
        "plantdoc_test_macro_f1",
        "plantvillage_test_macro_f1",
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

    summary_path = OUTPUT_DIR / "final_model_comparison_summary.csv"
    summary_df.to_csv(summary_path, index=False)

    print("\n" + "=" * 100)
    print("Final expanded model comparison")
    print("=" * 100)

    display_columns = [
        "run_name",
        "model_name",
        "loss_name",
        "final_test_score",
        "plantwild_test_macro_f1",
        "plantdoc_test_macro_f1",
        "fieldplant_test_macro_f1",
        "plantvillage_test_macro_f1",
        "plantwild_test_accuracy",
        "plantdoc_test_accuracy",
        "fieldplant_test_accuracy",
        "plantvillage_test_accuracy",
        "training_best_selection_score",
    ]

    existing_display_columns = [
        column
        for column in display_columns
        if column in summary_df.columns
    ]

    print(summary_df[existing_display_columns].to_string(index=False))
    print(f"\nSaved final summary to: {summary_path}")

    return summary_df, summary_path


def main():
    """
    Run final evaluation for all PlantWild-expanded checkpoints.
    """
    device = get_device()
    print(f"Using device: {device}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    expanded_class_names = load_expanded_class_names()

    if len(expanded_class_names) != EXPECTED_NUM_CLASSES:
        raise RuntimeError(
            f"Expected {EXPECTED_NUM_CLASSES} classes, "
            f"found {len(expanded_class_names)}."
        )

    print(f"Expanded classes: {len(expanded_class_names)}")

    loaders, datasets = prepare_final_test_loaders(
        expanded_class_names=expanded_class_names,
    )

    checkpoint_paths = find_checkpoints()

    print("\nFound PlantWild-expanded checkpoints:")
    for checkpoint_path in checkpoint_paths:
        print(f"- {checkpoint_path.name}")

    mlflow.set_tracking_uri(f"sqlite:///{MLFLOW_DB_PATH.as_posix()}")
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)

    summary_rows = []

    for checkpoint_path in checkpoint_paths:
        summary_row = evaluate_checkpoint(
            checkpoint_path=checkpoint_path,
            loaders=loaders,
            datasets=datasets,
            expanded_class_names=expanded_class_names,
            device=device,
        )

        summary_rows.append(summary_row)

        if device.type == "cuda":
            torch.cuda.empty_cache()

    summary_df, summary_path = save_final_summary(summary_rows)

    with mlflow.start_run(run_name="final_expanded_evaluation_summary"):
        mlflow.log_param("num_evaluated_models", len(summary_df))
        mlflow.log_param("eval_batch_size", EVAL_BATCH_SIZE)
        mlflow.log_param("num_classes", len(expanded_class_names))

        for key, value in FINAL_SCORE_WEIGHTS.items():
            mlflow.log_param(f"final_score_weight.{key}", value)

        mlflow.log_artifact(
            str(summary_path),
            artifact_path="evaluation_summary",
        )


if __name__ == "__main__":
    main()