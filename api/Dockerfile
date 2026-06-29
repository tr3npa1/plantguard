# PlantGuard FastAPI inference container.
#
# The container serves the exported ONNX model with ONNX Runtime.
# It does not include PyTorch, training scripts, MLflow logs, datasets, or
# evaluation artifacts.

FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# ONNX Runtime requires OpenMP runtime libraries on slim Debian images.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements-api.txt ./requirements-api.txt

RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements-api.txt

COPY api ./api
COPY models/onnx ./models/onnx

ENV PLANTGUARD_ONNX_MODEL_PATH=/app/models/onnx/plantguard_resnet50_cross_entropy.onnx
ENV PLANTGUARD_METADATA_PATH=/app/models/onnx/plantguard_resnet50_cross_entropy_metadata.json

EXPOSE 8000

# Use ${PORT:-8000} so the same image works locally and on platforms like Render.
CMD ["sh", "-c", "exec uvicorn api.main:app --host 0.0.0.0 --port ${PORT:-8000}"]