# PlantGuard

**PlantGuard** is a production-oriented plant disease classification system built with PyTorch, ONNX Runtime, FastAPI, and Docker.

The project trains, adapts, evaluates, explains, exports, and serves a deep learning model across multiple crop disease datasets. It is designed to go beyond clean benchmark accuracy by testing real-world robustness, domain shift, class expansion, and explanation quality.

The final system includes:

* PlantVillage base training
* PlantDoc domain adaptation
* PlantWild_v2 expanded 132-class training
* FieldPlant external evaluation
* MLflow experiment tracking
* Grad-CAM model auditing
* ONNX export and verification
* FastAPI image-upload inference API
* Dockerized API deployment

## Quickstart

This quickstart runs the final PlantGuard inference API locally.

PlantGuard’s source code, dataset-preparation scripts, training pipeline, evaluation pipeline, ONNX export utility, FastAPI API, and Docker configuration are included in this repository.

Large generated artifacts are intentionally not committed to Git, including:

```text
models/
*.pth
*.pt
*.onnx
data/raw/
data/train/
data/val/
data/test/
evaluation_results/
explain/results/
mlruns/
mlartifacts/
```

To run inference, the exported ONNX model and metadata must exist locally under:

```text
models/onnx/
├── plantguard_resnet50_cross_entropy.onnx
└── plantguard_resnet50_cross_entropy_metadata.json
```

### 1. Clone the repository

```powershell
git clone https://github.com/tr3npa1/plantguard.git
cd plantguard
```

### 2. Create and activate a Python environment

```powershell
python -m venv .venv
.\.venv\Scripts\activate
```

Install the API dependencies:

```powershell
pip install -r requirements-api.txt
```

### 3. Prepare ONNX inference artifacts

If the ONNX model and metadata already exist locally, confirm them with:

```powershell
Test-Path .\models\onnx\plantguard_resnet50_cross_entropy.onnx
Test-Path .\models\onnx\plantguard_resnet50_cross_entropy_metadata.json
```

Both commands should return:

```text
True
```

If the ONNX artifacts are missing, export the selected model after placing the trained checkpoint in the expected local model directory:

```powershell
python export\to_onnx.py
```

A successful export prints:

```text
ONNX graph validation passed
ONNX Runtime verification passed
Export completed successfully
```

### 4. Run the FastAPI inference server

```powershell
python api\main.py
```

Or run with Uvicorn directly:

```powershell
uvicorn api.main:app --host 127.0.0.1 --port 8000 --reload
```

Open the interactive API docs:

```text
http://127.0.0.1:8000/docs
```

### 5. Test the API

Health check:

```powershell
curl.exe http://127.0.0.1:8000/health
```

Model metadata:

```powershell
curl.exe http://127.0.0.1:8000/model
```

Prediction request:

```powershell
curl.exe -X POST "http://127.0.0.1:8000/predict?top_k=5" `
  -F "file=@data\test\Apple___Apple_scab\0340dc35-5215-48ab-8db7-06af99fcb358___FREC_Scab 2966.JPG"
```

The API returns top-k class predictions and confidence scores as JSON.

### 6. Run with Docker

Build the Docker image from the repository root:

```powershell
docker build -f api\Dockerfile -t plantguard-api .
```

Run the container:

```powershell
docker run --rm -p 8000:8000 plantguard-api
```

Then open:

```text
http://127.0.0.1:8000/docs
```

or test from another terminal:

```powershell
curl.exe http://127.0.0.1:8000/health
curl.exe http://127.0.0.1:8000/model
```

The Docker image copies the local ONNX model and metadata from:

```text
models/onnx/
```

So those files must exist before building the image.


## Final Model

The selected final model is:

```text
resnet50_cross_entropy_plantdoc_finetuned_plantwild_expanded
```

Architecture and training objective:

```text
ResNet50 + Cross Entropy
```

This model was selected because it achieved the strongest overall balance of quantitative robustness and qualitative explanation quality.

It had:

* the highest combined final test score
* the strongest hard-set mean macro F1
* the best FieldPlant macro F1
* the best PlantDoc macro F1
* strong PlantWild_v2 performance
* the most trustworthy Grad-CAM focus
* better class-specific Grad-CAM behavior on incorrect predictions

EfficientNet-B3 Focal Loss achieved the best PlantWild_v2 macro F1, but its Grad-CAMs were often broad and less class-discriminative. ResNet50 Cross Entropy produced more compact, disease-focused heatmaps, making it the stronger final choice for a reliability-focused plant disease recognition system.

## Project Motivation

Many crop disease classifiers perform extremely well on clean benchmark datasets such as PlantVillage but fail on real-world images because of domain shift.

PlantGuard directly investigates this problem by evaluating across:

* clean lab-style images
* real-world mobile/camera images
* field-style images
* expanded disease categories
* external datasets with different image distributions

The goal is not only to build a classifier, but to build a realistic machine learning system that is evaluated honestly before deployment.

## Current Status

Implemented:

* PlantVillage dataset download and train/validation/test split generation
* PlantDoc download support and label mapping
* PlantWild_v2 expanded label-space preparation
* FieldPlant external test-set conversion
* dataset-specific normalization
* custom PyTorch datasets and dataloaders
* EfficientNet-B0, EfficientNet-B3, and ResNet50 training
* Cross Entropy, Weighted Cross Entropy, and Focal Loss comparison
* PlantDoc fine-tuning
* PlantWild_v2 132-class expanded training with replay
* final evaluation across PlantVillage, PlantDoc, PlantWild_v2, and FieldPlant
* per-class precision, recall, F1-score, confusion matrices, and top-confusion reports
* MLflow experiment tracking
* Grad-CAM explainability and visual model audit
* ONNX export for the selected model
* PyTorch vs ONNX Runtime numerical verification
* FastAPI inference API with image upload
* Dockerized inference service

Not committed to Git:

* raw datasets
* train/validation/test image folders
* model checkpoints
* ONNX model artifacts
* MLflow runs
* evaluation outputs
* Grad-CAM generated images

## Pipeline Overview

PlantGuard is organized into five major parts.

### 1. PlantVillage Base Training

The first stage trains 38-class models on PlantVillage.

Architectures compared:

* EfficientNet-B0
* EfficientNet-B3
* ResNet50

Loss functions compared:

* CrossEntropyLoss
* Weighted CrossEntropyLoss
* Focal Loss

This creates a 3 × 3 experiment matrix across model architecture and class-imbalance strategy.

### 2. PlantDoc Domain Adaptation

PlantDoc is used for real-world domain adaptation.

PlantDoc contains more varied real-world images than PlantVillage, including different backgrounds, lighting, camera angles, leaf positions, and image quality.

Each PlantVillage-trained checkpoint is fine-tuned on PlantDoc train-adapt data and validated on a PlantDoc validation split.

The best PlantDoc-adapted checkpoint is selected using validation macro F1.

### 3. PlantWild_v2 Expanded Training

PlantWild_v2 expands PlantGuard from the original 38 PlantVillage labels to a 132-class label space.

The expanded label convention is:

```text
indices 0-37   -> original PlantVillage / PlantGuard labels
indices 38-131 -> new PlantWild_v2 labels
```

Expanded training uses replay to reduce catastrophic forgetting:

```text
80% PlantWild_v2
12% PlantDoc replay
8% PlantVillage replay
```

The classifier head is expanded from 38 classes to 132 classes:

* compatible backbone weights are copied
* classifier rows 0-37 are copied from the PlantDoc-fine-tuned checkpoint
* classifier rows 38-131 are randomly initialized

Model selection during expanded training uses:

```text
selection_score =
  0.80 * PlantWild validation macro F1
+ 0.12 * PlantDoc validation macro F1
+ 0.08 * PlantVillage validation macro F1
```

### 4. Final Evaluation and Grad-CAM Audit

Final expanded checkpoints are evaluated on:

* PlantVillage test
* PlantDoc test
* PlantWild_v2 test
* FieldPlant compatible external test

The final test score is computed as:

```text
final_test_score =
  0.50 * PlantWild test macro F1
+ 0.20 * PlantDoc test macro F1
+ 0.20 * FieldPlant test macro F1
+ 0.10 * PlantVillage test macro F1
```

Grad-CAM visualizations are then generated from saved prediction CSVs to inspect whether models focus on diseased plant regions or irrelevant shortcuts such as backgrounds, borders, full leaf silhouettes, or image artifacts.

### 5. ONNX, FastAPI, and Docker Inference

The selected ResNet50 Cross Entropy checkpoint is exported to ONNX.

The export utility:

* loads the selected PyTorch checkpoint
* rebuilds the matching model architecture
* exports the model to ONNX
* validates the ONNX graph
* verifies ONNX Runtime output against PyTorch output
* saves inference metadata as JSON

The FastAPI service:

* loads the ONNX model and metadata at startup
* accepts image uploads
* decodes images with Pillow
* resizes and normalizes images using saved metadata
* runs ONNX Runtime inference
* applies softmax
* returns top-k predictions as JSON

The Docker image packages the API, runtime dependencies, ONNX model, and metadata into a portable inference container.

## Datasets

### PlantVillage

PlantVillage is used for original supervised training, validation, and internal testing.

Local split structure:

```text
data/train/
data/val/
data/test/
```

PlantVillage is clean and lab-style. It is useful for base training, but it is not sufficient for real-world robustness evaluation.

### PlantDoc

PlantDoc is used as an external real-world dataset.

It is mapped into the PlantGuard label space using manually reviewed label mappings. It is used for:

* zero-shot external evaluation
* PlantDoc fine-tuning
* PlantDoc replay during PlantWild_v2 expanded training
* held-out external testing

Initial zero-shot evaluation showed a large performance drop compared to PlantVillage, confirming strong domain shift.

### PlantWild_v2

PlantWild_v2 is used to expand the PlantGuard label space.

Final expanded label space:

```text
38 original PlantVillage labels
94 new PlantWild_v2 labels
132 total PlantGuard labels
```

Generated split files:

```text
data/splits/plantwild_train.csv
data/splits/plantwild_val.csv
data/splits/plantwild_test.csv
```

### FieldPlant

FieldPlant is an external YOLO/object-detection-style dataset converted into a compatible image-level classification test set.

It is not used for training.

Images are skipped if they contain:

* missing label files
* empty label files
* mixed classes
* invalid class IDs
* unmapped classes

Generated compatible test file:

```text
data/splits/fieldplant_test.csv
```

FieldPlant is one of the hardest evaluation sets and is especially important for separating models that look equally strong on cleaner datasets.

## Final Model Ranking

PlantVillage performance is not useful for selecting the final model because almost all models reach very high macro F1 on PlantVillage.

The real separation comes from:

* FieldPlant
* PlantDoc
* PlantWild_v2
* Grad-CAM focus quality

| Model                         | Final Score | Hard-Set Mean Macro F1 | FieldPlant F1 | PlantDoc F1 | PlantWild F1 | Grad-CAM Quality       |
| ----------------------------- | ----------: | ---------------------: | ------------: | ----------: | -----------: | ---------------------- |
| ResNet50 + Cross Entropy      |      0.6106 |                 0.5397 |        0.2830 |      0.7102 |       0.6260 | Best                   |
| EfficientNet-B3 + Weighted CE |      0.6073 |                 0.5291 |        0.2686 |      0.6821 |       0.6365 | Weak / broad           |
| EfficientNet-B3 + Focal Loss  |      0.6061 |                 0.5220 |        0.2412 |      0.6775 |       0.6471 | Weak / broad           |
| ResNet50 + Focal Loss         |      0.5939 |                 0.5236 |        0.2756 |      0.6932 |       0.6021 | Good, but lower F1     |
| ResNet50 + Weighted CE        |      0.5754 |                 0.4927 |        0.1954 |      0.6793 |       0.6034 | Good CAM, poor metrics |

## Final Decision

The selected final model is:

```text
resnet50_cross_entropy_plantdoc_finetuned_plantwild_expanded
```

This model is preferred because it has the strongest balance of numerical robustness and explanation quality.

It achieved:

* highest final test score
* highest hard-set mean macro F1
* best FieldPlant macro F1
* best PlantDoc macro F1
* strong PlantWild performance
* more meaningful Grad-CAM focus
* better class-specific Grad-CAM behavior on wrong predictions

The runner-up is:

```text
efficientnet_b3_weighted_cross_entropy_plantdoc_finetuned_plantwild_expanded
```

That model may be considered if deployment size or inference speed becomes more important than explanation quality and external-set robustness.

## Grad-CAM Explainability Audit

PlantGuard includes a custom Grad-CAM pipeline implemented with raw PyTorch hooks.

Grad-CAM is used to evaluate whether the model focuses on meaningful disease regions rather than shortcuts such as:

* image borders
* background texture
* full leaf silhouettes
* labels or artifacts
* random non-disease regions

### Grad-CAM Findings

The ResNet50 Cross Entropy model produced the most trustworthy heatmaps overall.

Its Grad-CAMs were generally:

* more compact
* more disease-region focused
* better aligned with leaf streaks, spots, rust, blight, and affected tissue
* less distracted by background regions
* more class-specific on incorrect predictions

EfficientNet-B3 Weighted CE and EfficientNet-B3 Focal Loss achieved competitive numerical scores, but their heatmaps were often too broad. Many explanations activated large background regions, borders, full leaf shapes, or non-disease areas.

In wrong predictions, the predicted-class and true-class Grad-CAMs for EfficientNet models were often very similar, suggesting weaker class-discriminative behavior.

This visual audit was a major reason ResNet50 Cross Entropy was selected as the final model.

## API Inference

The FastAPI inference service is implemented in:

```text
api/main.py
```

Available endpoints:

| Endpoint   | Method | Description                                   |
| ---------- | ------ | --------------------------------------------- |
| `/`        | GET    | Basic service information                     |
| `/health`  | GET    | API and model-loading status                  |
| `/model`   | GET    | Model metadata and inference contract         |
| `/predict` | POST   | Upload an image and receive top-k predictions |
| `/docs`    | GET    | Interactive Swagger UI                        |

Run locally:

```powershell
python api\main.py
```

Or with Uvicorn:

```powershell
uvicorn api.main:app --host 127.0.0.1 --port 8000 --reload
```

Open the interactive API docs:

```text
http://127.0.0.1:8000/docs
```

Example prediction request:

```powershell
curl.exe -X POST "http://127.0.0.1:8000/predict?top_k=5" `
  -F "file=@data\test\Apple___Apple_scab\0340dc35-5215-48ab-8db7-06af99fcb358___FREC_Scab 2966.JPG"
```

Example successful prediction:

```json
{
  "filename": "0340dc35-5215-48ab-8db7-06af99fcb358___FREC_Scab 2966.JPG",
  "model": "resnet50_cross_entropy_plantdoc_finetuned_plantwild_expanded",
  "top_k": 5,
  "predictions": [
    {
      "rank": 1,
      "class_index": 0,
      "class_name": "Apple___Apple_scab",
      "confidence": 0.999994158744812
    }
  ]
}
```

## Dockerized API

The inference API can be built and run as a Docker container.

Docker files:

```text
api/Dockerfile
.dockerignore
requirements-api.txt
```

Build the image from the repository root:

```powershell
docker build -f api\Dockerfile -t plantguard-api .
```

Run the container:

```powershell
docker run --rm -p 8000:8000 plantguard-api
```

Then test:

```powershell
curl.exe http://127.0.0.1:8000/health
curl.exe http://127.0.0.1:8000/model
```

Prediction test:

```powershell
curl.exe -X POST "http://127.0.0.1:8000/predict?top_k=5" `
  -F "file=@data\test\Apple___Apple_scab\0340dc35-5215-48ab-8db7-06af99fcb358___FREC_Scab 2966.JPG"
```

The ONNX model and metadata are copied into the Docker image from:

```text
models/onnx/
```

These generated model artifacts are intentionally not committed to Git.

## ONNX Export

The selected model is exported using:

```text
export/to_onnx.py
```

Run:

```powershell
python export\to_onnx.py
```

Generated local artifacts:

```text
models/onnx/plantguard_resnet50_cross_entropy.onnx
models/onnx/plantguard_resnet50_cross_entropy_metadata.json
```

The export script verifies ONNX Runtime against PyTorch before saving metadata. A successful export prints:

```text
ONNX graph validation passed
ONNX Runtime verification passed
Export completed successfully
```

Generated ONNX files are not committed to Git.

## Evaluation Outputs

The evaluation pipeline saves:

* summary metrics
* per-class precision, recall, and F1
* classification reports
* confusion matrices
* normalized confusion matrices
* top confused class pairs
* prediction distributions
* per-image prediction CSVs

Final evaluation outputs are saved under:

```text
evaluation_results/final_expanded_evaluation/
```

Generated evaluation outputs are not committed to Git.

## Grad-CAM Outputs

Grad-CAM outputs are generated from final evaluation prediction CSVs.

Example command:

```powershell
python explain\gradcam.py --all-checkpoints --datasets plantwild_test fieldplant_test plantdoc_test plantvillage_test --samples-per-dataset 8 --target-mode both
```

Outputs are saved under:

```text
explain/results/gradcam/<run_name>/<dataset_name>/
```

The Grad-CAM audit included contact sheets for:

* FieldPlant
* PlantDoc
* PlantWild_v2
* PlantVillage

Generated Grad-CAM images are not committed by default. Selected representative examples can later be copied into `docs/assets/gradcam/` for README display.

## Repository Structure

```text
plantguard/
├── api/
│   ├── main.py
│   └── Dockerfile
├── data/
│   ├── download.py
│   ├── dataset.py
│   ├── prepare_plantwild.py
│   └── prepare_fieldplant.py
├── export/
│   └── to_onnx.py
├── explain/
│   └── gradcam.py
├── training/
│   ├── train.py
│   ├── finetune_plantdoc.py
│   ├── train_plantwild.py
│   ├── evaluate.py
│   └── config.yaml
├── .dockerignore
├── .gitignore
├── requirements-api.txt
└── README.md
```

## Main Commands

Prepare PlantVillage:

```powershell
python data\download.py --dataset plantvillage
```

Prepare PlantDoc:

```powershell
python data\download.py --dataset plantdoc
```

Prepare PlantWild_v2:

```powershell
python data\download.py --dataset plantwild
python data\prepare_plantwild.py
```

Prepare FieldPlant:

```powershell
python data\download.py --dataset fieldplant
python data\prepare_fieldplant.py
```

Train PlantVillage base models:

```powershell
python training\train.py
```

Fine-tune on PlantDoc:

```powershell
python training\finetune_plantdoc.py
```

Train expanded PlantWild models:

```powershell
python training\train_plantwild.py
```

Run final evaluation:

```powershell
python training\evaluate.py
```

Export ONNX:

```powershell
python export\to_onnx.py
```

Run FastAPI locally:

```powershell
python api\main.py
```

Build Docker image:

```powershell
docker build -f api\Dockerfile -t plantguard-api .
```

Run Docker container:

```powershell
docker run --rm -p 8000:8000 plantguard-api
```

Generate Grad-CAM explanations:

```powershell
python explain\gradcam.py --all-checkpoints --datasets plantwild_test fieldplant_test plantdoc_test plantvillage_test --samples-per-dataset 8 --target-mode both
```

Launch MLflow UI:

```powershell
mlflow ui --backend-store-uri sqlite:///mlflow.db --host 127.0.0.1 --port 5000
```

## Tech Stack

* Python
* PyTorch
* torchvision
* scikit-learn
* pandas
* NumPy
* Pillow
* matplotlib
* MLflow
* YAML configs
* Grad-CAM
* ONNX
* ONNX Runtime
* FastAPI
* Uvicorn
* Docker

## What Is Not Committed

The following are generated artifacts and should not be committed:

```text
data/raw/
data/train/
data/val/
data/test/
models/
*.pth
*.pt
*.onnx
evaluation_results/
explain/results/
mlruns/
mlartifacts/
mlflow.db
__pycache__/
```

The repository commits code, metadata, mappings, split manifests, and deployment configuration. Large model artifacts and generated outputs are kept local.


## Research and Evaluation Highlights

This project intentionally avoids choosing a model based only on clean benchmark accuracy.

Important findings:

1. PlantVillage accuracy is saturated and does not meaningfully separate models.
2. External datasets reveal large robustness differences.
3. FieldPlant and PlantDoc are more useful for final model selection than PlantVillage.
4. Grad-CAM quality can change the final model choice.
5. A model with slightly lower PlantWild_v2 macro F1 can still be a better final model if it is more robust and more explainable.
6. ResNet50 Cross Entropy produced the best combined result across metrics and explanation quality.
7. ONNX Runtime inference was verified against PyTorch before API integration.
8. The API can serve predictions locally and inside Docker.

## Project Direction

PlantGuard is a production-oriented and research-minded computer vision project.

The focus is not simply high benchmark accuracy. The focus is building a realistic ML system that includes:

* reproducible training
* experiment tracking
* external validation
* domain adaptation
* expanded label-space learning
* explainability
* failure-case analysis
* ONNX export
* API serving
* containerized deployment preparation

The central lesson is that clean-dataset accuracy alone is not enough. A useful crop disease model must be evaluated against real-world images, external datasets, and visual explanation quality before deployment.
