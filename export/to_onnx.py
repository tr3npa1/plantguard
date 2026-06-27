"""
Export the selected PlantGuard checkpoint to ONNX.

This utility converts a trained PyTorch checkpoint into an ONNX inference
artifact and writes a metadata JSON file used by downstream inference services.

The exported ONNX graph expects already-preprocessed tensors:

    input:  float32 tensor [batch_size, 3, 256, 256]
    output: float32 logits [batch_size, num_classes]

Image decoding, resizing, normalization, softmax, and top-k decoding are kept
outside the ONNX graph.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from data.dataset import (  # noqa: E402
    DATASET_MEAN,
    DATASET_STD,
    IMAGE_SIZE,
    load_expanded_class_names,
)
from training.evaluate import count_parameters, load_checkpoint  # noqa: E402
from training.train import build_model, get_device  # noqa: E402


LOGGER = logging.getLogger(__name__)

DEFAULT_CHECKPOINT_PATH = (
    PROJECT_ROOT
    / "models"
    / "plantwild_expanded"
    / "resnet50_cross_entropy_plantdoc_finetuned_plantwild_expanded_best_model.pth"
)

DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "models" / "onnx"
DEFAULT_ONNX_NAME = "plantguard_resnet50_cross_entropy.onnx"
DEFAULT_METADATA_NAME = "plantguard_resnet50_cross_entropy_metadata.json"

INPUT_NAME = "input"
OUTPUT_NAME = "logits"
DEFAULT_OPSET_VERSION = 18
EXPECTED_NUM_CLASSES = 132


def configure_logging() -> None:
    """
    Configure console logging for the export utility.
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s | %(message)s",
    )


def resolve_path(path_value: str | Path) -> Path:
    """
    Resolve an absolute or project-relative path.

    Args:
        path_value:
            Absolute path or path relative to the repository root.

    Returns:
        Absolute Path object.
    """
    path = Path(path_value)

    if path.is_absolute():
        return path

    return PROJECT_ROOT / path


def as_project_relative(path: Path) -> str:
    """
    Convert a path to a project-relative POSIX string when possible.

    Args:
        path:
            Path to convert.

    Returns:
        Project-relative path string, or absolute path if outside the project.
    """
    try:
        return path.resolve().relative_to(PROJECT_ROOT.resolve()).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def get_nested_value(
    dictionary: dict[str, Any],
    keys: tuple[str, ...],
    default: Any = None,
) -> Any:
    """
    Safely read a nested dictionary value.

    Args:
        dictionary:
            Source dictionary.
        keys:
            Nested key path.
        default:
            Fallback value if any key is missing.

    Returns:
        Nested value or default.
    """
    value: Any = dictionary

    for key in keys:
        if not isinstance(value, dict) or key not in value:
            return default

        value = value[key]

    return value


def get_checkpoint_field(
    checkpoint: dict[str, Any],
    direct_key: str,
    config_path: tuple[str, ...],
    source_config_path: tuple[str, ...],
    default: Any = None,
) -> Any:
    """
    Read checkpoint metadata from direct, config, or source_config fields.

    Args:
        checkpoint:
            Loaded checkpoint dictionary.
        direct_key:
            Top-level checkpoint key to try first.
        config_path:
            Nested path inside checkpoint["config"].
        source_config_path:
            Nested path inside checkpoint["source_config"].
        default:
            Fallback value.

    Returns:
        Metadata value.
    """
    if direct_key in checkpoint:
        return checkpoint[direct_key]

    value = get_nested_value(
        dictionary=checkpoint,
        keys=("config", *config_path),
    )

    if value is not None:
        return value

    value = get_nested_value(
        dictionary=checkpoint,
        keys=("source_config", *source_config_path),
    )

    if value is not None:
        return value

    return default


def get_model_metadata(
    checkpoint: dict[str, Any],
    checkpoint_path: Path,
) -> tuple[str, str, str, list[str]]:
    """
    Extract model metadata required for export.

    Args:
        checkpoint:
            Loaded checkpoint dictionary.
        checkpoint_path:
            Path to the source checkpoint.

    Returns:
        Tuple containing model name, loss name, run name, and class names.
    """
    model_name = get_checkpoint_field(
        checkpoint=checkpoint,
        direct_key="model_name",
        config_path=("model", "name"),
        source_config_path=("model", "name"),
    )

    loss_name = get_checkpoint_field(
        checkpoint=checkpoint,
        direct_key="loss_name",
        config_path=("training", "loss"),
        source_config_path=("training", "loss"),
        default="unknown",
    )

    run_name = get_checkpoint_field(
        checkpoint=checkpoint,
        direct_key="run_name",
        config_path=("mlflow", "run_name"),
        source_config_path=("mlflow", "run_name"),
        default=checkpoint_path.stem.replace("_best_model", ""),
    )

    class_names = checkpoint.get("class_names")

    if class_names is None:
        class_names = load_expanded_class_names()
    
    class_names = list(class_names)

    if model_name is None:
        raise KeyError("Could not determine model architecture from checkpoint.")

    if len(class_names) != EXPECTED_NUM_CLASSES:
        raise RuntimeError(
            f"Expected {EXPECTED_NUM_CLASSES} classes, found {len(class_names)}."
        )

    return str(model_name), str(loss_name), str(run_name), class_names


def load_model_for_export(
    checkpoint: dict[str, Any],
    model_name: str,
    class_names: list[str],
    device: torch.device,
) -> torch.nn.Module:
    """
    Build the model architecture and load checkpoint weights.

    Args:
        checkpoint:
            Loaded checkpoint dictionary.
        model_name:
            Model architecture name.
        class_names:
            Class names in output-index order.
        device:
            Device used for PyTorch export.

    Returns:
        Loaded model in evaluation mode.
    """
    model = build_model(
        model_name=model_name,
        num_classes=len(class_names),
        pretrained=False,
    )

    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    
    return model


def export_onnx_model(
    model: torch.nn.Module,
    onnx_path: Path,
    device: torch.device,
    opset_version: int,
    dynamic_batch: bool,
) -> None:
    """
    Export a PyTorch model to ONNX.

    Args:
        model:
            PyTorch model in evaluation mode.
        onnx_path:
            Output ONNX path.
        device:
            Device used for tracing.
        opset_version:
            ONNX opset version.
        dynamic_batch:
            Whether the exported graph should accept variable batch sizes.
    """
    onnx_path.parent.mkdir(parents=True, exist_ok=True)

    dummy_input = torch.randn(
        1,
        3,
        IMAGE_SIZE,
        IMAGE_SIZE,
        device=device,
        dtype=torch.float32,
    )

    dynamic_axes = None

    if dynamic_batch:
        dynamic_axes = {
            INPUT_NAME: {0: "batch_size"},
            OUTPUT_NAME: {0: "batch_size"},
        }
    
    torch.onnx.export(
        model,
        dummy_input,
        str(onnx_path),
        export_params=True,
        opset_version=opset_version,
        do_constant_folding=True,
        input_names=[INPUT_NAME],
        output_names=[OUTPUT_NAME],
        dynamic_axes=dynamic_axes,
        dynamo=False,
    )


def validate_onnx_graph(onnx_path: Path) -> None:
    """
    Validate the exported ONNX graph.

    Args:
        onnx_path:
            Path to exported ONNX file.
    """
    try:
        import onnx
    except ImportError as error:
        raise ImportError("Install ONNX with: pip install onnx") from error

    onnx_model = onnx.load(str(onnx_path))
    onnx.checker.check_model(onnx_model)


def run_onnxruntime(
        onnx_path: Path,
        input_array: np.ndarray,
) -> np.ndarray:
    """
    Run ONNX Runtime inference.

    Args:
        onnx_path:
            Path to ONNX model.
        input_array:
            Input array with shape [batch_size, 3, IMAGE_SIZE, IMAGE_SIZE].

    Returns:
        Output logits.
    """
    try:
        import onnxruntime as ort
    except ImportError as error:
        raise ImportError("Install ONNX Runtime with: pip install onnxruntime") from error
    
    session = ort.InferenceSession(
        str(onnx_path),
        providers=["CPUExecutionProvider"],
    )

    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name

    return session.run(
        [output_name],
        {input_name: input_array}
    )[0]


def get_topk_indices(logits: np.ndarray, k: int = 5) -> np.ndarray:
    """
    Get top-k class indices from logits.

    Args:
        logits:
            Model logits with shape [batch_size, num_classes].
        k:
            Number of top indices to return.

    Returns:
        Top-k indices sorted by descending logit value.
    """
    return np.argsort(logits, axis=1)[:, -k:][:, ::-1]


def verify_onnx_export(
    model: torch.nn.Module,
    onnx_path: Path,
    device: torch.device,
    batch_size: int,
    absolute_tolerance: float,
    relative_tolerance: float,
) -> dict[str, Any]:
    """
    Compare PyTorch and ONNX Runtime outputs on deterministic synthetic input.

    Args:
        model:
            Loaded PyTorch model.
        onnx_path:
            Exported ONNX model path.
        device:
            Device used for PyTorch inference.
        batch_size:
            Verification batch size.
        absolute_tolerance:
            Absolute tolerance for numerical comparison.
        relative_tolerance:
            Relative tolerance for numerical comparison.

    Returns:
        Verification metrics.

    Raises:
        RuntimeError:
            If ONNX Runtime output does not match PyTorch output closely enough.
    """
    torch.manual_seed(42)

    input_tensor = torch.randn(
        batch_size,
        3,
        IMAGE_SIZE,
        IMAGE_SIZE,
        device=device,
        dtype=torch.float32,
    )

    with torch.no_grad():
        pytorch_logits = model(input_tensor).detach().cpu().numpy()

    onnx_logits = run_onnxruntime(
        onnx_path=onnx_path,
        input_array=input_tensor.detach().cpu().numpy(),
    )

    if pytorch_logits.shape != onnx_logits.shape:
        raise RuntimeError(
            f"Shape mismatch: PyTorch={pytorch_logits.shape}, ONNX={onnx_logits.shape}"
        )
    
    absolute_difference = np.abs(pytorch_logits - onnx_logits) 

    max_absolute_difference = float(absolute_difference.max())
    mean_absolute_difference = float(absolute_difference.mean())

    outputs_match = bool(
        np.allclose(
            pytorch_logits,
            onnx_logits,
            atol=absolute_tolerance,
            rtol=relative_tolerance,
        )
    )

    pytorch_top1 = get_topk_indices(pytorch_logits, k=1).reshape(-1)
    onnx_top1 = get_topk_indices(onnx_logits, k=1).reshape(-1)
    top1_matches = bool(np.array_equal(pytorch_top1, onnx_top1))

    result = {
        "batch_size": batch_size,
        "max_absolute_difference": max_absolute_difference,
        "mean_absolute_difference": mean_absolute_difference,
        "outputs_match": outputs_match,
        "top1_matches": top1_matches,
        "pytorch_top1": pytorch_top1.tolist(),
        "onnx_top1": onnx_top1.tolist(),
        "pytorch_top5_first_sample": get_topk_indices(pytorch_logits, k=5)[0].tolist(),
        "onnx_top5_first_sample": get_topk_indices(onnx_logits, k=5)[0].tolist(),
        "absolute_tolerance": absolute_tolerance,
        "relative_tolerance": relative_tolerance,
    }

    if not outputs_match:
        raise RuntimeError("ONNX verification failed: logits are not close enough.")

    if not top1_matches:
        raise RuntimeError("ONNX verification failed: top-1 predictions differ.")

    return result


def save_metadata(
    metadata_path: Path,
    checkpoint_path: Path,
    onnx_path: Path,
    model_name: str,
    loss_name: str,
    run_name: str,
    class_names: list[str],
    opset_version: int,
    dynamic_batch: bool,
    verification: dict[str, Any] | None,
) -> None:
    """
    Save deployment metadata beside the ONNX model.

    Args:
        metadata_path:
            Output metadata JSON path.
        checkpoint_path:
            Source checkpoint path.
        onnx_path:
            Exported ONNX path.
        model_name:
            Model architecture name.
        loss_name:
            Training loss name.
        run_name:
            Source experiment or checkpoint run name.
        class_names:
            Class names in output-index order.
        opset_version:
            ONNX opset version.
        dynamic_batch:
            Whether ONNX uses a dynamic batch dimension.
        verification:
            Optional ONNX Runtime verification result.
    """
    metadata_path.parent.mkdir(parents=True, exist_ok=True)

    metadata = {
        "project": "PlantGuard",
        "task": "plant disease classification",
        "model": {
            "architecture": model_name,
            "loss": loss_name,
            "run_name": run_name,
            "num_classes": len(class_names),
            "class_names": class_names,
        },
        "artifacts": {
            "source_checkpoint": as_project_relative(checkpoint_path),
            "onnx_model": as_project_relative(onnx_path),
        },
        "onnx": {
            "opset_version": opset_version,
            "dynamic_batch": dynamic_batch,
            "input_name": INPUT_NAME,
            "output_name": OUTPUT_NAME,
        },
        "input": {
            "name": INPUT_NAME,
            "dtype": "float32",
            "layout": "NCHW",
            "shape": ["batch_size", 3, IMAGE_SIZE, IMAGE_SIZE],
            "color_mode": "RGB",
        },
        "output": {
            "name": OUTPUT_NAME,
            "dtype": "float32",
            "shape": ["batch_size", len(class_names)],
            "meaning": "raw logits before softmax",
        },
        "preprocessing": {
            "resize": [IMAGE_SIZE, IMAGE_SIZE],
            "pixel_scaling": "uint8 [0, 255] -> float32 [0, 1]",
            "normalization_mean": DATASET_MEAN,
            "normalization_std": DATASET_STD,
        },
        "postprocessing": {
            "apply_softmax": True,
            "recommended_output": "top-k class probabilities",
        },
        "verification": verification,
    }

    with metadata_path.open("w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=2)


def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments.

    Returns:
        Parsed arguments.
    """
    parser = argparse.ArgumentParser(
        description="Export a PlantGuard PyTorch checkpoint to ONNX."
    )

    parser.add_argument(
        "--checkpoint",
        type=str,
        default=str(DEFAULT_CHECKPOINT_PATH),
        help="Path to the PyTorch checkpoint.",
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(DEFAULT_OUTPUT_DIR),
        help="Directory for ONNX and metadata outputs.",
    )

    parser.add_argument(
        "--onnx-name",
        type=str,
        default=DEFAULT_ONNX_NAME,
        help="Output ONNX filename.",
    )

    parser.add_argument(
        "--metadata-name",
        type=str,
        default=DEFAULT_METADATA_NAME,
        help="Output metadata JSON filename.",
    )

    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        choices=["cpu", "cuda", "auto"],
        help="PyTorch device used during export.",
    )

    parser.add_argument(
        "--opset",
        type=int,
        default=DEFAULT_OPSET_VERSION,
        help="ONNX opset version.",
    )

    parser.add_argument(
        "--static-batch",
        action="store_true",
        help="Export with fixed batch size instead of dynamic batch size.",
    )

    parser.add_argument(
        "--skip-verification",
        action="store_true",
        help="Skip ONNX Runtime verification.",
    )

    parser.add_argument(
        "--verification-batch-size",
        type=int,
        default=1,
        help="Synthetic batch size used for export verification.",
    )

    parser.add_argument(
        "--atol",
        type=float,
        default=1e-4,
        help="Absolute tolerance for PyTorch vs ONNX comparison.",
    )

    parser.add_argument(
        "--rtol",
        type=float,
        default=1e-3,
        help="Relative tolerance for PyTorch vs ONNX comparison.",
    )

    return parser.parse_args()


def main() -> None:
    """
    Export the selected PlantGuard checkpoint to ONNX and save metadata.
    """
    configure_logging()
    args = parse_args()

    checkpoint_path = resolve_path(args.checkpoint)
    output_dir = resolve_path(args.output_dir)

    onnx_path = output_dir / args.onnx_name
    metadata_path = output_dir / args.metadata_name

    device = get_device(args.device)
    dynamic_batch = not args.static_batch

    LOGGER.info("Loading checkpoint: %s", checkpoint_path)

    checkpoint = load_checkpoint(
        checkpoint_path=checkpoint_path)

    model_name, loss_name, run_name, class_names = get_model_metadata(
        checkpoint=checkpoint,
        checkpoint_path=checkpoint_path,
    )

    model = load_model_for_export(
        checkpoint=checkpoint,
        model_name=model_name,
        class_names=class_names,
        device=device,
    )

    LOGGER.info("Model:       %s", model_name)
    LOGGER.info("Loss:        %s", loss_name)
    LOGGER.info("Run:         %s", run_name)
    LOGGER.info("Classes:     %s", len(class_names))
    
    total_parameters, trainable_parameters = count_parameters(model)

    LOGGER.info("Total parameters:     %s", f"{total_parameters:,}")
    LOGGER.info("Trainable parameters: %s", f"{trainable_parameters:,}")

    LOGGER.info("Exporting ONNX model: %s", onnx_path)

    export_onnx_model(
        model=model,
        onnx_path=onnx_path,
        device=device,
        opset_version=args.opset,
        dynamic_batch=dynamic_batch,
    )

    validate_onnx_graph(onnx_path)
    LOGGER.info("ONNX graph validation passed")

    verification = None

    if not args.skip_verification:
        verification = verify_onnx_export(
            model=model,
            onnx_path=onnx_path,
            device=device,
            batch_size=args.verification_batch_size,
            absolute_tolerance=args.atol,
            relative_tolerance=args.rtol,
        )

        LOGGER.info("ONNX Runtime verification passed")
        LOGGER.info(
            "Max absolute difference: %.8f",
            verification["max_absolute_difference"],
        )

    save_metadata(
        metadata_path=metadata_path,
        checkpoint_path=checkpoint_path,
        onnx_path=onnx_path,
        model_name=model_name,
        loss_name=loss_name,
        run_name=run_name,
        class_names=class_names,
        opset_version=args.opset,
        dynamic_batch=dynamic_batch,
        verification=verification,
    )

    LOGGER.info("Metadata saved: %s", metadata_path)
    LOGGER.info("Export completed successfully")


if __name__ == "__main__":
    main()
