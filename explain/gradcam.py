"""
GradCAM explainability pipeline for PlantGuard.

This script generates GradCAM visualizations for PlantWild-expanded PlantGuard
checkpoints using the per-image prediction CSVs produced by:

    python training/evaluate.py

The script is intentionally self-contained and uses raw PyTorch hooks instead
of an external GradCAM package. This makes the explainability method easier to
audit and easier to discuss in the project README/interviews.

Typical single-checkpoint usage:

    python explain/gradcam.py ^
        --checkpoint models/plantwild_expanded/efficientnet_b3_focal_loss_plantdoc_finetuned_plantwild_expanded_best_model.pth ^
        --datasets plantwild_test fieldplant_test plantdoc_test plantvillage_test ^
        --samples-per-dataset 8 ^
        --target-mode both

Generate GradCAMs for all final expanded checkpoints:

    python explain/gradcam.py ^
        --all-checkpoints ^
        --datasets plantwild_test fieldplant_test plantdoc_test plantvillage_test ^
        --samples-per-dataset 8 ^
        --target-mode both

Target modes:
    predicted:
        Explain the class predicted by the model.

    true:
        Explain the ground-truth class.

    both:
        Explain the predicted class for every sample. For incorrect predictions,
        also generate a second GradCAM for the true class.

Outputs are saved under:

    explain/results/gradcam/<run_name>/<dataset_name>/

Each output figure contains:
    1. Original resized RGB image
    2. GradCAM heatmap
    3. Heatmap overlay on the image
"""

from __future__ import annotations

import argparse
import gc
import re
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from data.dataset import (  # noqa: E402
    DATASET_MEAN,
    DATASET_STD,
    IMAGE_SIZE,
    load_expanded_class_names,
)
from training.train import build_model  # noqa: E402


DEFAULT_CHECKPOINT_DIR = PROJECT_ROOT / "models" / "plantwild_expanded"
DEFAULT_CHECKPOINT_GLOB = "*_plantwild_expanded_best_model.pth"

DEFAULT_FINAL_CHECKPOINT = (
    DEFAULT_CHECKPOINT_DIR
    / "efficientnet_b3_focal_loss_plantdoc_finetuned_plantwild_expanded_best_model.pth"
)

DEFAULT_EVAL_DIR = PROJECT_ROOT / "evaluation_results" / "final_expanded_evaluation"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "explain" / "results" / "gradcam"

DEFAULT_DATASETS = [
    "plantwild_test",
    "fieldplant_test",
    "plantdoc_test",
    "plantvillage_test",
]

PREDICTION_COLUMNS = {
    "image_path",
    "true_class_index",
    "true_class",
    "predicted_class_index",
    "predicted_class",
    "is_correct",
}


def get_device(device_arg: str) -> torch.device:
    """
    Resolve the requested compute device.

    Args:
        device_arg:
            "auto", "cpu", or "cuda".

    Returns:
        torch.device selected for model inference and GradCAM backpropagation.
    """
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    return torch.device(device_arg)


def safe_torch_load(checkpoint_path: Path) -> dict:
    """
    Load a PyTorch checkpoint across PyTorch versions.

    Newer PyTorch versions support the weights_only argument. Older versions do
    not, so this helper falls back gracefully.

    Args:
        checkpoint_path:
            Path to a .pth checkpoint.

    Returns:
        Loaded checkpoint dictionary.
    """
    try:
        return torch.load(
            checkpoint_path,
            map_location="cpu",
            weights_only=False,
        )
    except TypeError:
        return torch.load(
            checkpoint_path,
            map_location="cpu",
        )


def sanitize_filename(text: str) -> str:
    """
    Convert arbitrary text into a safe filename component.

    Args:
        text:
            Class name, label, or any text value.

    Returns:
        Filename-safe string truncated to a reasonable length.
    """
    text = str(text)
    text = re.sub(r"[^A-Za-z0-9_.-]+", "_", text)
    text = text.strip("_")

    return text[:120]


def safe_relative_path(path: Path) -> str:
    """
    Convert a path to a project-relative string when possible.

    Args:
        path:
            Absolute or relative path.

    Returns:
        Project-relative path if path is inside PROJECT_ROOT, otherwise the
        original path string.
    """
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def resolve_checkpoint_paths(args: argparse.Namespace) -> list[Path]:
    """
    Resolve which checkpoints should be explained.

    Priority:
        1. --all-checkpoints
        2. --checkpoints checkpoint1 checkpoint2 ...
        3. --checkpoint single_checkpoint

    Args:
        args:
            Parsed command-line arguments.

    Returns:
        Sorted list of checkpoint paths.
    """
    if args.all_checkpoints:
        checkpoint_paths = sorted(
            Path(args.checkpoint_dir).glob(args.checkpoint_glob)
        )

        if not checkpoint_paths:
            raise FileNotFoundError(
                f"No checkpoints found in {args.checkpoint_dir} "
                f"matching glob: {args.checkpoint_glob}"
            )

        return checkpoint_paths

    if args.checkpoints:
        checkpoint_paths = [Path(path) for path in args.checkpoints]
    else:
        checkpoint_paths = [Path(args.checkpoint)]

    missing_paths = [
        checkpoint_path
        for checkpoint_path in checkpoint_paths
        if not checkpoint_path.exists()
    ]

    if missing_paths:
        raise FileNotFoundError(
            "Some checkpoint paths do not exist:\n"
            + "\n".join(str(path) for path in missing_paths)
        )

    return checkpoint_paths


def load_checkpoint_model(
    checkpoint_path: Path,
    device: torch.device,
) -> tuple[torch.nn.Module, dict, list[str]]:
    """
    Load a PlantWild-expanded checkpoint and rebuild its architecture.

    Expanded checkpoints store the architecture name and 132-class label order
    directly in the checkpoint. The label order must match
    data/metadata/plantguard_expanded_class_names.json, otherwise GradCAM
    explanations would be assigned to the wrong class names.

    Args:
        checkpoint_path:
            Path to a PlantWild-expanded .pth checkpoint.
        device:
            Device where the model should be placed.

    Returns:
        model:
            Loaded model in eval mode.
        checkpoint:
            Raw checkpoint dictionary.
        class_names:
            Expanded class names in model output-index order.
    """
    checkpoint_path = Path(checkpoint_path)

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    checkpoint = safe_torch_load(checkpoint_path)

    required_keys = {
        "model_state_dict",
        "class_names",
        "model_name",
        "run_name",
    }

    missing_keys = required_keys - set(checkpoint.keys())

    if missing_keys:
        raise KeyError(
            f"Checkpoint {checkpoint_path.name} is missing required keys: "
            f"{sorted(missing_keys)}"
        )

    checkpoint_class_names = list(checkpoint["class_names"])
    expanded_class_names = load_expanded_class_names()

    if checkpoint_class_names != list(expanded_class_names):
        raise ValueError(
            "Checkpoint class order does not match "
            "data/metadata/plantguard_expanded_class_names.json. "
            "GradCAM cannot safely continue because class labels may be wrong."
        )

    model = build_model(
        model_name=checkpoint["model_name"],
        num_classes=len(checkpoint_class_names),
        pretrained=False,
    )

    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    return model, checkpoint, checkpoint_class_names


def select_target_layer(
    model: torch.nn.Module,
    model_name: str,
) -> torch.nn.Module:
    """
    Select the final convolutional feature layer for GradCAM.

    GradCAM needs a spatial feature map, so the target layer should be late
    enough to capture high-level disease features while still preserving spatial
    layout.

    Args:
        model:
            Loaded PlantGuard model.
        model_name:
            Architecture name stored in the checkpoint.

    Returns:
        PyTorch module used as the GradCAM target layer.
    """
    normalized_name = model_name.lower()

    if "resnet" in normalized_name:
        return model.layer4[-1]

    if "efficientnet" in normalized_name:
        return model.features[-1]

    raise ValueError(
        f"No GradCAM target layer configured for model_name={model_name}. "
        "Add this architecture to select_target_layer()."
    )


class GradCAM:
    """
    Minimal GradCAM implementation using PyTorch hooks.

    Method:
        1. Register a forward hook on a late convolutional layer.
        2. Store that layer's activations during the forward pass.
        3. Backpropagate the target class score.
        4. Store gradients flowing through the same layer.
        5. Average gradients over spatial dimensions to get channel weights.
        6. Weighted-sum activations, apply ReLU, resize, and normalize.

    The resulting heatmap highlights image regions that most strongly influence
    the selected class score.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        target_layer: torch.nn.Module,
    ) -> None:
        """
        Register GradCAM hooks.

        Args:
            model:
                Model to explain.
            target_layer:
                Convolutional layer whose activations/gradients are used.
        """
        self.model = model
        self.target_layer = target_layer
        self.activations = None
        self.gradients = None

        self.forward_handle = self.target_layer.register_forward_hook(
            self._save_activations
        )
        self.backward_handle = self.target_layer.register_full_backward_hook(
            self._save_gradients
        )

    def _save_activations(
        self,
        module: torch.nn.Module,
        inputs: tuple[torch.Tensor, ...],
        output: torch.Tensor,
    ) -> None:
        """
        Save target-layer activations from the forward pass.
        """
        del module, inputs
        self.activations = output.detach()

    def _save_gradients(
        self,
        module: torch.nn.Module,
        grad_input: tuple[torch.Tensor, ...],
        grad_output: tuple[torch.Tensor, ...],
    ) -> None:
        """
        Save target-layer gradients from the backward pass.
        """
        del module, grad_input
        self.gradients = grad_output[0].detach()

    def generate(
        self,
        input_tensor: torch.Tensor,
        target_class_index: int,
    ) -> np.ndarray:
        """
        Generate one normalized GradCAM heatmap.

        Args:
            input_tensor:
                Image tensor of shape [1, 3, IMAGE_SIZE, IMAGE_SIZE].
            target_class_index:
                Class index whose score should be explained.

        Returns:
            Normalized heatmap with shape [IMAGE_SIZE, IMAGE_SIZE].
        """
        if input_tensor.ndim != 4 or input_tensor.shape[0] != 1:
            raise ValueError(
                "GradCAM expects input_tensor with shape [1, 3, H, W]. "
                f"Got shape: {tuple(input_tensor.shape)}"
            )

        self.model.zero_grad(set_to_none=True)
        self.activations = None
        self.gradients = None

        logits = self.model(input_tensor)

        if not 0 <= target_class_index < logits.shape[1]:
            raise ValueError(
                f"target_class_index={target_class_index} is outside model "
                f"output range [0, {logits.shape[1] - 1}]."
            )

        target_score = logits[:, target_class_index].sum()
        target_score.backward()

        if self.activations is None:
            raise RuntimeError("Forward hook did not capture activations.")

        if self.gradients is None:
            raise RuntimeError("Backward hook did not capture gradients.")

        channel_weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = (channel_weights * self.activations).sum(dim=1)
        cam = F.relu(cam)

        cam = F.interpolate(
            cam.unsqueeze(1),
            size=(IMAGE_SIZE, IMAGE_SIZE),
            mode="bilinear",
            align_corners=False,
        )

        cam = cam.squeeze().detach().cpu().numpy()

        cam_min = float(cam.min())
        cam_max = float(cam.max())
        cam = (cam - cam_min) / (cam_max - cam_min + 1e-8)

        return cam

    def remove_hooks(self) -> None:
        """
        Remove registered PyTorch hooks.

        This should be called before a checkpoint finishes so repeated GradCAM
        generation does not accumulate hooks on the model.
        """
        self.forward_handle.remove()
        self.backward_handle.remove()


def build_eval_transform() -> transforms.Compose:
    """
    Build the deterministic preprocessing pipeline used during evaluation.

    The GradCAM input must use the same preprocessing as training/evaluation;
    otherwise the explanation could reflect a different input distribution.
    """
    return transforms.Compose(
        [
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=DATASET_MEAN,
                std=DATASET_STD,
            ),
        ]
    )


def load_image_for_gradcam(
    image_path: Path,
    device: torch.device,
) -> tuple[torch.Tensor, np.ndarray]:
    """
    Load an image for model inference and visualization.

    Args:
        image_path:
            Path to the source image.
        device:
            Device where the input tensor should be placed.

    Returns:
        input_tensor:
            Normalized tensor with shape [1, 3, IMAGE_SIZE, IMAGE_SIZE].
        base_rgb:
            Resized RGB image in [0, 1] with shape [IMAGE_SIZE, IMAGE_SIZE, 3].
    """
    image_path = Path(image_path)

    image = Image.open(image_path).convert("RGB")
    resized_image = image.resize((IMAGE_SIZE, IMAGE_SIZE))

    base_rgb = np.asarray(resized_image).astype(np.float32) / 255.0

    transform = build_eval_transform()
    input_tensor = transform(image).unsqueeze(0).to(device)

    return input_tensor, base_rgb


def make_overlay(
    base_rgb: np.ndarray,
    cam: np.ndarray,
    alpha: float = 0.45,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Convert a GradCAM heatmap into a visible overlay.

    Args:
        base_rgb:
            Original resized RGB image in [0, 1].
        cam:
            Normalized GradCAM heatmap in [0, 1].
        alpha:
            Heatmap blending strength.

    Returns:
        heatmap_rgb:
            Colorized heatmap.
        overlay:
            Heatmap blended over the image.
    """
    heatmap_rgb = plt.get_cmap("jet")(cam)[:, :, :3]
    overlay = (1.0 - alpha) * base_rgb + alpha * heatmap_rgb
    overlay = np.clip(overlay, 0.0, 1.0)

    return heatmap_rgb, overlay


def save_gradcam_figure(
    base_rgb: np.ndarray,
    cam: np.ndarray,
    output_path: Path,
    dataset_name: str,
    image_path: Path,
    true_class: str,
    predicted_class: str,
    target_class: str,
    target_mode: str,
    is_correct: bool,
) -> None:
    """
    Save the original image, heatmap, and overlay as one figure.

    Args:
        base_rgb:
            Original resized RGB image.
        cam:
            Normalized GradCAM heatmap.
        output_path:
            Destination PNG path.
        dataset_name:
            Dataset name such as plantwild_test or fieldplant_test.
        image_path:
            Source image path.
        true_class:
            Ground-truth class name.
        predicted_class:
            Predicted class name.
        target_class:
            Class being explained by this GradCAM.
        target_mode:
            "predicted" or "true".
        is_correct:
            Whether the model prediction was correct.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    heatmap_rgb, overlay = make_overlay(base_rgb=base_rgb, cam=cam)

    fig, axes = plt.subplots(1, 3, figsize=(13, 4))

    axes[0].imshow(base_rgb)
    axes[0].set_title("Input")
    axes[0].axis("off")

    axes[1].imshow(heatmap_rgb)
    axes[1].set_title("GradCAM heatmap")
    axes[1].axis("off")

    axes[2].imshow(overlay)
    axes[2].set_title("Overlay")
    axes[2].axis("off")

    correctness = "correct" if is_correct else "wrong"

    fig.suptitle(
        "\n".join(
            [
                f"{dataset_name} | {correctness} | target={target_mode}",
                f"true: {true_class}",
                f"pred: {predicted_class}",
                f"explaining: {target_class}",
                f"image: {Path(image_path).name}",
            ]
        ),
        fontsize=9,
    )

    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def resolve_image_path(image_path_value: str) -> Path:
    """
    Resolve an image path stored in predictions.csv.

    The evaluation script stores paths relative to PROJECT_ROOT when possible,
    for example:

        data/raw/plantwild_v2/...

    Args:
        image_path_value:
            Raw path value from predictions.csv.

    Returns:
        Absolute image path.
    """
    image_path = Path(str(image_path_value))

    if image_path.is_absolute():
        return image_path

    return PROJECT_ROOT / image_path


def load_predictions(
    eval_dir: Path,
    run_name: str,
    dataset_name: str,
) -> pd.DataFrame:
    """
    Load per-image predictions for one model/dataset pair.

    Args:
        eval_dir:
            Root directory produced by training/evaluate.py.
        run_name:
            Checkpoint run name.
        dataset_name:
            Dataset folder name inside the run directory.

    Returns:
        DataFrame containing image paths, labels, predictions, and correctness.
    """
    predictions_path = Path(eval_dir) / run_name / dataset_name / "predictions.csv"

    if not predictions_path.exists():
        raise FileNotFoundError(
            f"predictions.csv not found for {run_name}/{dataset_name}: "
            f"{predictions_path}\n"
            "Run training/evaluate.py before generating GradCAM outputs."
        )

    dataframe = pd.read_csv(predictions_path)

    missing_columns = PREDICTION_COLUMNS - set(dataframe.columns)

    if missing_columns:
        raise ValueError(
            f"{predictions_path} is missing required columns: "
            f"{sorted(missing_columns)}"
        )

    dataframe["is_correct"] = (
        dataframe["is_correct"]
        .astype(str)
        .str.lower()
        .isin(["true", "1", "yes"])
    )

    return dataframe


def sample_predictions(
    dataframe: pd.DataFrame,
    samples_per_dataset: int,
    seed: int,
    only_wrong: bool = False,
) -> pd.DataFrame:
    """
    Sample representative predictions for GradCAM generation.

    By default, the function samples approximately half correct and half wrong
    predictions when both are available. This produces better README/report
    examples than random sampling alone.

    Args:
        dataframe:
            Full predictions dataframe.
        samples_per_dataset:
            Target number of rows to sample.
        seed:
            Random seed for reproducible sampling.
        only_wrong:
            If True, sample only incorrect predictions when possible.

    Returns:
        Sampled dataframe.
    """
    if samples_per_dataset <= 0:
        raise ValueError("samples_per_dataset must be greater than 0.")

    if dataframe.empty:
        raise ValueError("Cannot sample from an empty predictions dataframe.")

    if only_wrong:
        wrong_df = dataframe[~dataframe["is_correct"]]

        if wrong_df.empty:
            return dataframe.sample(
                n=min(samples_per_dataset, len(dataframe)),
                random_state=seed,
            )

        return wrong_df.sample(
            n=min(samples_per_dataset, len(wrong_df)),
            random_state=seed,
        )

    correct_df = dataframe[dataframe["is_correct"]]
    wrong_df = dataframe[~dataframe["is_correct"]]

    target_correct = samples_per_dataset // 2
    target_wrong = samples_per_dataset - target_correct

    sampled_parts = []

    if not correct_df.empty:
        sampled_parts.append(
            correct_df.sample(
                n=min(target_correct, len(correct_df)),
                random_state=seed,
            )
        )

    if not wrong_df.empty:
        sampled_parts.append(
            wrong_df.sample(
                n=min(target_wrong, len(wrong_df)),
                random_state=seed + 1,
            )
        )

    if sampled_parts:
        sampled = pd.concat(sampled_parts, axis=0)
    else:
        sampled = pd.DataFrame(columns=dataframe.columns)

    remaining_needed = samples_per_dataset - len(sampled)

    if remaining_needed > 0:
        remaining_df = dataframe.drop(index=sampled.index, errors="ignore")

        if not remaining_df.empty:
            sampled_extra = remaining_df.sample(
                n=min(remaining_needed, len(remaining_df)),
                random_state=seed + 2,
            )
            sampled = pd.concat([sampled, sampled_extra], axis=0)

    return sampled.sample(frac=1.0, random_state=seed + 3).reset_index(drop=True)


def get_target_modes_for_row(
    row: pd.Series,
    target_mode: str,
) -> list[tuple[str, int]]:
    """
    Decide which class scores should be explained for one prediction row.

    Args:
        row:
            Prediction row from predictions.csv.
        target_mode:
            "predicted", "true", or "both".

    Returns:
        List of (mode_name, class_index) pairs.
    """
    true_index = int(row["true_class_index"])
    predicted_index = int(row["predicted_class_index"])

    if target_mode == "predicted":
        return [("predicted", predicted_index)]

    if target_mode == "true":
        return [("true", true_index)]

    if target_mode == "both":
        targets = [("predicted", predicted_index)]

        if true_index != predicted_index:
            targets.append(("true", true_index))

        return targets

    raise ValueError(f"Unsupported target_mode: {target_mode}")


def generate_gradcams_for_dataset(
    gradcam: GradCAM,
    dataframe: pd.DataFrame,
    dataset_name: str,
    output_dataset_dir: Path,
    class_names: list[str],
    device: torch.device,
    target_mode: str,
) -> list[dict]:
    """
    Generate GradCAM outputs for sampled predictions from one dataset.

    Args:
        gradcam:
            GradCAM object with hooks registered.
        dataframe:
            Sampled predictions dataframe.
        dataset_name:
            Dataset name such as plantwild_test.
        output_dataset_dir:
            Folder where dataset-specific PNGs are saved.
        class_names:
            Expanded class-name list.
        device:
            Compute device.
        target_mode:
            "predicted", "true", or "both".

    Returns:
        List of metadata records for gradcam_index.csv.
    """
    records = []

    for row_index, row in dataframe.iterrows():
        image_path = resolve_image_path(row["image_path"])

        if not image_path.exists():
            print(f"Skipping missing image: {image_path}")
            continue

        true_class = str(row["true_class"])
        predicted_class = str(row["predicted_class"])
        is_correct = bool(row["is_correct"])

        input_tensor, base_rgb = load_image_for_gradcam(
            image_path=image_path,
            device=device,
        )

        targets = get_target_modes_for_row(
            row=row,
            target_mode=target_mode,
        )

        for current_target_mode, target_index in targets:
            target_class = class_names[int(target_index)]
            correctness = "correct" if is_correct else "wrong"

            output_name = (
                f"{row_index:04d}"
                f"__{correctness}"
                f"__target_{current_target_mode}"
                f"__true_{sanitize_filename(true_class)}"
                f"__pred_{sanitize_filename(predicted_class)}"
                f".png"
            )

            output_path = output_dataset_dir / output_name

            cam = gradcam.generate(
                input_tensor=input_tensor,
                target_class_index=int(target_index),
            )

            save_gradcam_figure(
                base_rgb=base_rgb,
                cam=cam,
                output_path=output_path,
                dataset_name=dataset_name,
                image_path=image_path,
                true_class=true_class,
                predicted_class=predicted_class,
                target_class=target_class,
                target_mode=current_target_mode,
                is_correct=is_correct,
            )

            records.append(
                {
                    "dataset": dataset_name,
                    "image_path": safe_relative_path(image_path),
                    "true_class_index": int(row["true_class_index"]),
                    "true_class": true_class,
                    "predicted_class_index": int(row["predicted_class_index"]),
                    "predicted_class": predicted_class,
                    "is_correct": is_correct,
                    "target_mode": current_target_mode,
                    "target_class_index": int(target_index),
                    "target_class": target_class,
                    "gradcam_path": safe_relative_path(output_path),
                }
            )

            print(f"Saved: {output_path}")

    return records


def generate_for_checkpoint(
    checkpoint_path: Path,
    args: argparse.Namespace,
    device: torch.device,
) -> dict:
    """
    Generate GradCAM outputs for one checkpoint across requested datasets.

    Args:
        checkpoint_path:
            Checkpoint to explain.
        args:
            Parsed command-line arguments.
        device:
            Compute device.

    Returns:
        Summary dictionary for all_checkpoints_gradcam_summary.csv.
    """
    print("\n" + "#" * 100)
    print(f"Generating GradCAMs for checkpoint: {checkpoint_path.name}")
    print("#" * 100)

    model, checkpoint, class_names = load_checkpoint_model(
        checkpoint_path=checkpoint_path,
        device=device,
    )

    run_name = checkpoint["run_name"]
    model_name = checkpoint["model_name"]
    loss_name = checkpoint.get("loss_name", "unknown_loss")

    print(f"Run name: {run_name}")
    print(f"Model name: {model_name}")
    print(f"Loss name: {loss_name}")
    print(f"Classes: {len(class_names)}")

    target_layer = select_target_layer(
        model=model,
        model_name=model_name,
    )

    gradcam = GradCAM(
        model=model,
        target_layer=target_layer,
    )

    output_model_dir = args.output_dir / run_name
    output_model_dir.mkdir(parents=True, exist_ok=True)

    all_records = []

    try:
        for dataset_name in args.datasets:
            print("\n" + "=" * 80)
            print(f"Generating GradCAM for dataset: {dataset_name}")
            print("=" * 80)

            predictions_df = load_predictions(
                eval_dir=args.eval_dir,
                run_name=run_name,
                dataset_name=dataset_name,
            )

            sampled_df = sample_predictions(
                dataframe=predictions_df,
                samples_per_dataset=args.samples_per_dataset,
                seed=args.seed,
                only_wrong=args.only_wrong,
            )

            output_dataset_dir = output_model_dir / dataset_name
            output_dataset_dir.mkdir(parents=True, exist_ok=True)

            records = generate_gradcams_for_dataset(
                gradcam=gradcam,
                dataframe=sampled_df,
                dataset_name=dataset_name,
                output_dataset_dir=output_dataset_dir,
                class_names=class_names,
                device=device,
                target_mode=args.target_mode,
            )

            all_records.extend(records)

    finally:
        gradcam.remove_hooks()

    index_path = output_model_dir / "gradcam_index.csv"
    pd.DataFrame(all_records).to_csv(index_path, index=False)

    print("\n" + "=" * 80)
    print(f"GradCAM generation complete for: {run_name}")
    print(f"Saved index: {index_path}")
    print(f"Total GradCAM images: {len(all_records)}")
    print("=" * 80)

    del gradcam
    del model

    if device.type == "cuda":
        torch.cuda.empty_cache()

    gc.collect()

    return {
        "run_name": run_name,
        "model_name": model_name,
        "loss_name": loss_name,
        "checkpoint": checkpoint_path.name,
        "num_gradcam_images": len(all_records),
        "gradcam_index": safe_relative_path(index_path),
    }


def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments.
    """
    parser = argparse.ArgumentParser(
        description="Generate GradCAM overlays for PlantGuard final models."
    )

    checkpoint_group = parser.add_mutually_exclusive_group()

    checkpoint_group.add_argument(
        "--checkpoint",
        type=Path,
        default=DEFAULT_FINAL_CHECKPOINT,
        help="Path to a single PlantWild-expanded checkpoint.",
    )

    checkpoint_group.add_argument(
        "--checkpoints",
        nargs="+",
        type=Path,
        help="Explicit list of checkpoint paths to process.",
    )

    checkpoint_group.add_argument(
        "--all-checkpoints",
        action="store_true",
        help="Generate GradCAMs for every final expanded checkpoint.",
    )

    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=DEFAULT_CHECKPOINT_DIR,
        help="Directory searched when --all-checkpoints is used.",
    )

    parser.add_argument(
        "--checkpoint-glob",
        type=str,
        default=DEFAULT_CHECKPOINT_GLOB,
        help="Glob pattern used with --all-checkpoints.",
    )

    parser.add_argument(
        "--eval-dir",
        type=Path,
        default=DEFAULT_EVAL_DIR,
        help="Directory containing final evaluation outputs.",
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory where GradCAM results are saved.",
    )

    parser.add_argument(
        "--datasets",
        nargs="+",
        default=DEFAULT_DATASETS,
        help="Dataset names to generate GradCAMs for.",
    )

    parser.add_argument(
        "--samples-per-dataset",
        type=int,
        default=8,
        help="Number of prediction rows sampled per dataset.",
    )

    parser.add_argument(
        "--target-mode",
        choices=["predicted", "true", "both"],
        default="both",
        help="Which class score to explain.",
    )

    parser.add_argument(
        "--only-wrong",
        action="store_true",
        help="Sample only incorrect predictions when possible.",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for prediction sampling.",
    )

    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda"],
        default="auto",
        help="Device to use for model inference and GradCAM.",
    )

    return parser.parse_args()


def main() -> None:
    """
    Run GradCAM generation for one checkpoint, selected checkpoints, or all checkpoints.
    """
    args = parse_args()

    device = get_device(args.device)
    print(f"Using device: {device}")

    checkpoint_paths = resolve_checkpoint_paths(args)

    print("\nCheckpoints selected for GradCAM:")
    for checkpoint_path in checkpoint_paths:
        print(f"- {checkpoint_path}")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    summary_rows = []

    for checkpoint_path in checkpoint_paths:
        summary_row = generate_for_checkpoint(
            checkpoint_path=checkpoint_path,
            args=args,
            device=device,
        )
        summary_rows.append(summary_row)

    summary_path = args.output_dir / "all_checkpoints_gradcam_summary.csv"
    pd.DataFrame(summary_rows).to_csv(summary_path, index=False)

    print("\n" + "#" * 100)
    print("All requested GradCAM generation complete.")
    print(f"Processed checkpoints: {len(summary_rows)}")
    print(f"Saved summary: {summary_path}")
    print("#" * 100)


if __name__ == "__main__":
    main()