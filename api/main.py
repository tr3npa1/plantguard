"""
FastAPI inference service for PlantGuard.

This application serves the exported PlantGuard ONNX model through HTTP
endpoints. It loads the ONNX model and metadata once at startup, accepts an
uploaded image, applies the same preprocessing used during training/evaluation,
runs ONNX Runtime inference, and returns top-k plant disease predictions.

The ONNX model expects a preprocessed tensor:

    input:  float32 tensor [batch_size, 3, image_size, image_size]
    output: float32 logits [batch_size, num_classes]

Raw image decoding, resizing, normalization, softmax, and class-name decoding
are handled by this API layer, not inside the ONNX graph.
"""

from __future__ import annotations

import io
import json
import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import numpy as np
import onnxruntime as ort
from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from PIL import Image, UnidentifiedImageError
from pydantic import BaseModel, Field


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_ONNX_MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "onnx"
    / "plantguard_resnet50_cross_entropy.onnx"
)

DEFAULT_METADATA_PATH = (
    PROJECT_ROOT
    / "models"
    / "onnx"
    / "plantguard_resnet50_cross_entropy_metadata.json"
)

MODEL_PATH_ENV = "PLANTGUARD_ONNX_MODEL_PATH"
METADATA_PATH_ENV = "PLANTGUARD_METADATA_PATH"

DEFAULT_TOP_K = 5
MAX_TOP_K = 20

LOGGER = logging.getLogger(__name__)


class Prediction(BaseModel):
    """
    One ranked prediction returned by the model.
    """

    rank: int = Field(..., examples=[1])
    class_index: int = Field(..., examples=[42])
    class_name: str = Field(..., examples=["Tomato___Late_blight"])
    confidence: float = Field(..., examples=[0.9342])


class PredictionResponse(BaseModel):
    """
    Response returned by the /predict endpoint.
    """

    filename: str
    model: str
    top_k: int
    predictions: list[Prediction]


class HealthResponse(BaseModel):
    """
    Response returned by the /health endpoint.
    """

    status: str
    model_loaded: bool
    model_path: str
    metadata_path: str


class ModelInfoResponse(BaseModel):
    """
    Response returned by the /model endpoint.
    """

    project: str
    task: str
    model: dict[str, Any]
    input: dict[str, Any]
    output: dict[str, Any]
    preprocessing: dict[str, Any]
    postprocessing: dict[str, Any]


def configure_logging() -> None:
    """
    Configure application logging.
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


def get_model_path() -> Path:
    """
    Resolve the ONNX model path from environment or default project location.

    Returns:
        ONNX model path.
    """
    return resolve_path(
        os.getenv(
            MODEL_PATH_ENV,
            str(DEFAULT_ONNX_MODEL_PATH),
        )
    )


def get_metadata_path() -> Path:
    """
    Resolve the metadata path from environment or default project location.

    Returns:
        Metadata JSON path.
    """
    return resolve_path(
        os.getenv(
            METADATA_PATH_ENV,
            str(DEFAULT_METADATA_PATH),
        )
    )


def load_json(path: Path) -> dict[str, Any]:
    """
    Load a JSON object from disk.

    Args:
        path:
            JSON file path.

    Returns:
        Parsed JSON dictionary.

    Raises:
        FileNotFoundError:
            If the file does not exist.
        ValueError:
            If the JSON root is not an object.
    """
    if not path.exists():
        raise FileNotFoundError(f"JSON file not found: {path}")

    with path.open("r", encoding="utf-8") as file:
        data = json.load(file)

    if not isinstance(data, dict):
        raise ValueError(f"Expected JSON object in {path}")

    return data


def stable_softmax(logits: np.ndarray) -> np.ndarray:
    """
    Convert logits to probabilities using a numerically stable softmax.

    Args:
        logits:
            Logit array with shape [batch_size, num_classes].

    Returns:
        Probability array with the same shape.
    """
    shifted_logits = logits - np.max(logits, axis=1, keepdims=True)
    exp_values = np.exp(shifted_logits)

    return exp_values / np.sum(exp_values, axis=1, keepdims=True)


def get_bilinear_resize_filter() -> int:
    """
    Return Pillow's bilinear resize filter in a version-compatible way.

    Returns:
        Pillow bilinear resampling value.
    """
    if hasattr(Image, "Resampling"):
        return Image.Resampling.BILINEAR

    return Image.BILINEAR


class PlantGuardInferenceEngine:
    """
    ONNX Runtime inference wrapper for PlantGuard.

    This class owns the ONNX Runtime session, metadata, preprocessing settings,
    class-name decoding, and prediction formatting.
    """

    def __init__(self, model_path: Path, metadata_path: Path) -> None:
        """
        Initialize the inference engine.

        Args:
            model_path:
                Path to the exported ONNX model.
            metadata_path:
                Path to the metadata JSON generated during ONNX export.
        """
        self.model_path = model_path
        self.metadata_path = metadata_path

        self.metadata: dict[str, Any] = {}
        self.session: ort.InferenceSession | None = None

        self.input_name = ""
        self.output_name = ""

        self.image_size = 0
        self.class_names: list[str] = []

        self.normalization_mean: np.ndarray | None = None
        self.normalization_std: np.ndarray | None = None

    @property
    def is_loaded(self) -> bool:
        """
        Check whether the ONNX Runtime session is available.

        Returns:
            True if the model has been loaded.
        """
        return self.session is not None

    @property
    def model_run_name(self) -> str:
        """
        Return the model run name from metadata.

        Returns:
            Model run name, or "unknown" if missing.
        """
        model_metadata = self.metadata.get("model", {})

        return str(model_metadata.get("run_name", "unknown"))

    def load(self) -> None:
        """
        Load metadata and initialize the ONNX Runtime session.

        Raises:
            FileNotFoundError:
                If the ONNX model or metadata file is missing.
            ValueError:
                If metadata is incomplete or invalid.
            RuntimeError:
                If the ONNX runtime contract does not match metadata.
        """
        if not self.model_path.exists():
            raise FileNotFoundError(f"ONNX model not found: {self.model_path}")

        self.metadata = load_json(self.metadata_path)
        self._load_metadata_fields()

        self.session = ort.InferenceSession(
            str(self.model_path),
            providers=["CPUExecutionProvider"],
        )

        self._validate_runtime_contract()

        LOGGER.info("Loaded ONNX model: %s", self.model_path)
        LOGGER.info("Loaded metadata: %s", self.metadata_path)
        LOGGER.info("Model run: %s", self.model_run_name)
        LOGGER.info("Classes: %s", len(self.class_names))

    def _load_metadata_fields(self) -> None:
        """
        Extract required inference fields from metadata.

        Raises:
            ValueError:
                If required metadata fields are missing or invalid.
        """
        model_metadata = self.metadata.get("model", {})
        onnx_metadata = self.metadata.get("onnx", {})
        input_metadata = self.metadata.get("input", {})
        output_metadata = self.metadata.get("output", {})
        preprocessing_metadata = self.metadata.get("preprocessing", {})

        self.class_names = list(model_metadata.get("class_names", []))

        if not self.class_names:
            raise ValueError("Metadata is missing model.class_names")

        input_shape = input_metadata.get("shape")

        if not isinstance(input_shape, list) or len(input_shape) != 4:
            raise ValueError("Metadata input.shape must be a list of four dimensions")

        self.image_size = int(input_shape[-1])

        if self.image_size <= 0:
            raise ValueError(f"Invalid image size in metadata: {self.image_size}")

        normalization_mean = preprocessing_metadata.get("normalization_mean")
        normalization_std = preprocessing_metadata.get("normalization_std")

        if normalization_mean is None or normalization_std is None:
            raise ValueError(
                "Metadata preprocessing section must include normalization_mean "
                "and normalization_std"
            )

        self.normalization_mean = np.asarray(
            normalization_mean,
            dtype=np.float32,
        ).reshape(1, 1, 3)

        self.normalization_std = np.asarray(
            normalization_std,
            dtype=np.float32,
        ).reshape(1, 1, 3)

        self.input_name = str(
            onnx_metadata.get(
                "input_name",
                input_metadata.get("name", "input"),
            )
        )

        self.output_name = str(
            onnx_metadata.get(
                "output_name",
                output_metadata.get("name", "logits"),
            )
        )

    def _validate_runtime_contract(self) -> None:
        """
        Confirm ONNX Runtime input/output names match metadata.

        Raises:
            RuntimeError:
                If ONNX model input/output names do not match metadata.
        """
        if self.session is None:
            raise RuntimeError("ONNX Runtime session has not been initialized")

        runtime_input_name = self.session.get_inputs()[0].name
        runtime_output_name = self.session.get_outputs()[0].name

        if runtime_input_name != self.input_name:
            raise RuntimeError(
                "ONNX input name does not match metadata: "
                f"runtime={runtime_input_name}, metadata={self.input_name}"
            )

        if runtime_output_name != self.output_name:
            raise RuntimeError(
                "ONNX output name does not match metadata: "
                f"runtime={runtime_output_name}, metadata={self.output_name}"
            )

    def preprocess_image(self, image_bytes: bytes) -> np.ndarray:
        """
        Decode and preprocess uploaded image bytes for ONNX inference.

        Args:
            image_bytes:
                Raw uploaded image bytes.

        Returns:
            Float32 NumPy tensor with shape [1, 3, image_size, image_size].

        Raises:
            ValueError:
                If the uploaded bytes are not a valid image.
            RuntimeError:
                If metadata has not been loaded.
        """
        if self.normalization_mean is None or self.normalization_std is None:
            raise RuntimeError("Model metadata has not been loaded")

        try:
            image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        except UnidentifiedImageError as error:
            raise ValueError("Uploaded file is not a valid image") from error

        image = image.resize(
            (self.image_size, self.image_size),
            resample=get_bilinear_resize_filter(),
        )

        image_array = np.asarray(image, dtype=np.float32) / 255.0
        image_array = (image_array - self.normalization_mean) / self.normalization_std

        # Convert image layout from HWC to NCHW.
        image_array = np.transpose(image_array, (2, 0, 1))

        # Add batch dimension: [3, H, W] -> [1, 3, H, W].
        image_array = np.expand_dims(image_array, axis=0)

        return image_array.astype(np.float32)

    def predict(self, image_bytes: bytes, top_k: int) -> list[Prediction]:
        """
        Run ONNX inference and return top-k predictions.

        Args:
            image_bytes:
                Raw uploaded image bytes.
            top_k:
                Number of predictions to return.

        Returns:
            Ranked prediction list.

        Raises:
            RuntimeError:
                If the model has not been loaded.
            ValueError:
                If the uploaded image cannot be decoded.
        """
        if self.session is None:
            raise RuntimeError("Model is not loaded")

        input_tensor = self.preprocess_image(image_bytes)

        logits = self.session.run(
            [self.output_name],
            {self.input_name: input_tensor},
        )[0]

        probabilities = stable_softmax(logits)[0]
        top_k = min(top_k, len(self.class_names))

        top_indices = np.argsort(probabilities)[-top_k:][::-1]

        predictions: list[Prediction] = []

        for rank, class_index in enumerate(top_indices, start=1):
            class_index = int(class_index)

            predictions.append(
                Prediction(
                    rank=rank,
                    class_index=class_index,
                    class_name=self.class_names[class_index],
                    confidence=float(probabilities[class_index]),
                )
            )

        return predictions


inference_engine = PlantGuardInferenceEngine(
    model_path=get_model_path(),
    metadata_path=get_metadata_path(),
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Load model artifacts when the application starts.

    Args:
        app:
            FastAPI application instance.
    """
    configure_logging()

    try:
        inference_engine.load()
    except Exception:
        LOGGER.exception("Failed to load PlantGuard model artifacts")
        raise

    app.state.inference_engine = inference_engine

    yield

    LOGGER.info("Shutting down PlantGuard Inference API")


app = FastAPI(
    title="PlantGuard Inference API",
    description=(
        "ONNX Runtime inference service for the PlantGuard plant disease "
        "classification model."
    ),
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/")
def root() -> dict[str, Any]:
    """
    Return basic service information.

    Returns:
        Dictionary containing available endpoint paths.
    """
    return {
        "service": "PlantGuard Inference API",
        "status": "running",
        "endpoints": {
            "health": "/health",
            "model": "/model",
            "predict": "/predict",
            "docs": "/docs",
        },
    }


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    """
    Check API and model-loading status.

    Returns:
        Health response.
    """
    return HealthResponse(
        status="ok" if inference_engine.is_loaded else "model_not_loaded",
        model_loaded=inference_engine.is_loaded,
        model_path=str(inference_engine.model_path),
        metadata_path=str(inference_engine.metadata_path),
    )


@app.get("/model", response_model=ModelInfoResponse)
def model_info() -> ModelInfoResponse:
    """
    Return model metadata and inference contract.

    Returns:
        Model information response.

    Raises:
        HTTPException:
            If metadata has not been loaded.
    """
    metadata = inference_engine.metadata

    if not metadata:
        raise HTTPException(
            status_code=503,
            detail="Model metadata is not loaded.",
        )

    return ModelInfoResponse(
        project=str(metadata.get("project", "PlantGuard")),
        task=str(metadata.get("task", "plant disease classification")),
        model=dict(metadata.get("model", {})),
        input=dict(metadata.get("input", {})),
        output=dict(metadata.get("output", {})),
        preprocessing=dict(metadata.get("preprocessing", {})),
        postprocessing=dict(metadata.get("postprocessing", {})),
    )


@app.post("/predict", response_model=PredictionResponse)
async def predict(
    file: UploadFile = File(...),
    top_k: int = Query(
        default=DEFAULT_TOP_K,
        ge=1,
        le=MAX_TOP_K,
        description="Number of highest-confidence predictions to return.",
    ),
) -> PredictionResponse:
    """
    Predict plant disease from an uploaded image.

    Args:
        file:
            Uploaded image file.
        top_k:
            Number of ranked predictions to return.

    Returns:
        Prediction response with top-k class probabilities.

    Raises:
        HTTPException:
            If the file is missing, empty, not an image, or inference fails.
    """
    if not inference_engine.is_loaded:
        raise HTTPException(
            status_code=503,
            detail="Model is not loaded.",
        )

    if file.content_type and not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=400,
            detail=f"Expected an image upload, received: {file.content_type}",
        )

    image_bytes = await file.read()

    if not image_bytes:
        raise HTTPException(
            status_code=400,
            detail="Uploaded file is empty.",
        )

    try:
        predictions = inference_engine.predict(
            image_bytes=image_bytes,
            top_k=top_k,
        )
    except ValueError as error:
        raise HTTPException(
            status_code=400,
            detail=str(error),
        ) from error
    except Exception as error:
        LOGGER.exception("Prediction failed")
        raise HTTPException(
            status_code=500,
            detail="Prediction failed.",
        ) from error

    return PredictionResponse(
        filename=file.filename or "uploaded_image",
        model=inference_engine.model_run_name,
        top_k=top_k,
        predictions=predictions,
    )


if __name__ == "__main__":
    import sys

    import uvicorn

    project_root = Path(__file__).resolve().parents[1]

    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    os.chdir(project_root)

    uvicorn.run(
        "api.main:app",
        host="127.0.0.1",
        port=8000,
        reload=True,
        reload_dirs=[str(project_root / "api")],
    )