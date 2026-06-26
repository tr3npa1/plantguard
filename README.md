# PlantGuard

**PlantGuard** is a production-oriented plant disease classification system built with PyTorch. The project goes beyond clean benchmark accuracy by training, adapting, evaluating, and explaining deep learning models across multiple real-world crop disease datasets.

The final model selection is based on a combination of quantitative robustness metrics and qualitative Grad-CAM explainability audits.

## Final Model Recommendation

```text
resnet50_cross_entropy_plantdoc_finetuned_plantwild_expanded
```

The final recommended model is:

```text
ResNet50 + Cross Entropy
```

This model was selected because it achieved the best overall balance of:

* highest combined final test score
* strongest hard-set macro F1
* best FieldPlant macro F1
* best PlantDoc macro F1
* more trustworthy Grad-CAM focus
* better class-specific behavior on incorrect predictions

While EfficientNet-B3 Focal Loss achieved the best PlantWild_v2 macro F1, its Grad-CAMs were often too broad and less class-discriminative. ResNet50 Cross Entropy produced more compact and disease-focused heatmaps, making it the stronger final choice for reliability-focused deployment and research presentation.

## Project Summary

PlantGuard is an end-to-end computer vision pipeline for plant disease recognition.

It includes:

* dataset download and preparation scripts
* custom PyTorch datasets and dataloaders
* PlantVillage base training
* PlantDoc domain adaptation
* PlantWild_v2 expanded-label training
* FieldPlant external evaluation preparation
* MLflow experiment tracking
* final multi-dataset evaluation
* Grad-CAM explainability
* production-oriented code structure
* planned ONNX, FastAPI, and Docker deployment

The main objective is to build a crop disease classifier that is not only accurate on clean benchmark images, but also evaluated honestly on harder real-world datasets.

## Why This Project Matters

Many crop disease classifiers achieve very high accuracy on clean datasets such as PlantVillage, but fail to generalize to real-world images because of domain shift.

PlantGuard explicitly tests this problem by evaluating across:

* clean lab-style images
* field-style images
* real-world mobile/camera images
* expanded disease categories
* external datasets with different image distributions

The project is designed around the principle that a useful machine learning model must be evaluated on realistic failure cases, not only on clean benchmark splits.

## Current Status

Implemented:

* PlantVillage dataset download and train/validation/test split generation
* PlantDoc download support and label mapping
* PlantWild_v2 expanded label-space preparation
* FieldPlant external test-set conversion
* dataset-specific normalization
* custom PyTorch dataset and dataloader utilities
* EfficientNet-B0, EfficientNet-B3, and ResNet50 training
* Cross Entropy, Weighted Cross Entropy, and Focal Loss comparison
* PlantDoc fine-tuning
* PlantWild_v2 132-class expanded training with replay
* final evaluation across four datasets
* per-class metrics, confusion matrices, prediction CSVs, and top-confusion reports
* MLflow experiment tracking
* Grad-CAM explainability generation
* visual Grad-CAM model audit

Planned next:

* ONNX export
* PyTorch vs ONNX Runtime verification
* FastAPI inference API
* Docker packaging
* selected Grad-CAM examples added to README
* final deployment-oriented project polish

## Pipeline Overview

PlantGuard is organized into four major stages.

### Stage A: PlantVillage Base Training

The first stage trains 38-class models on PlantVillage.

Architectures compared:

* EfficientNet-B0
* EfficientNet-B3
* ResNet50

Loss functions compared:

* CrossEntropyLoss
* Weighted CrossEntropyLoss
* Focal Loss

This creates a 3 × 3 experiment matrix across architecture choice and class-imbalance strategy.

### Stage B: PlantDoc Domain Adaptation

PlantDoc is used for real-world domain adaptation.

PlantDoc contains more varied real-world images than PlantVillage, including different backgrounds, lighting, camera angles, leaf positions, and image quality.

Each PlantVillage-trained checkpoint is fine-tuned on PlantDoc train-adapt data and validated on a PlantDoc validation split.

The best PlantDoc-adapted checkpoint is selected using validation macro F1.

### Stage C: PlantWild_v2 Expanded Training

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

The model classifier head is expanded from 38 classes to 132 classes:

* compatible backbone weights are copied
* classifier rows 0-37 are copied from the PlantDoc-fine-tuned checkpoint
* classifier rows 38-131 are randomly initialized

Model selection during PlantWild training uses:

```text
selection_score =
  0.80 * PlantWild validation macro F1
+ 0.12 * PlantDoc validation macro F1
+ 0.08 * PlantVillage validation macro F1
```

### Stage D: Final Evaluation and Grad-CAM Audit

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

After final evaluation, Grad-CAM visualizations are generated from each model’s saved prediction CSVs. The final recommendation is based on both numerical performance and visual explanation quality.

## Datasets

### PlantVillage

PlantVillage is used for original supervised training, validation, and internal testing.

Local split structure:

```text
data/train/
data/val/
data/test/
```

PlantVillage is clean and lab-style, so it is useful for base training but not sufficient for real-world robustness evaluation.

### PlantDoc

PlantDoc is used as an external real-world dataset.

It is mapped into the PlantGuard label space using a manually reviewed mapping. It is used for:

* zero-shot external evaluation
* PlantDoc fine-tuning
* PlantDoc replay during PlantWild-expanded training
* held-out external testing

Initial zero-shot evaluation showed a large performance drop compared to PlantVillage, confirming a strong domain shift.

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

PlantVillage performance is not useful for choosing the final model because almost all models reach very high macro F1 on PlantVillage.

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

Generated Grad-CAM images are not committed by default. Selected representative examples can be copied into `docs/assets/gradcam/` for README display.

## Repository Structure

```text
plantguard/
├── data/
│   ├── download.py
│   ├── dataset.py
│   ├── prepare_plantwild.py
│   └── prepare_fieldplant.py
├── training/
│   ├── train.py
│   ├── finetune_plantdoc.py
│   ├── train_plantwild.py
│   ├── evaluate.py
│   └── config.yaml
├── explain/
│   └── gradcam.py
├── export/
│   └── to_onnx.py
├── api/
│   ├── main.py
│   └── Dockerfile
├── notebooks/
│   └── class_imbalance.ipynb
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
* matplotlib
* MLflow
* YAML configs
* Grad-CAM
* ONNX Runtime
* FastAPI
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

## Engineering Highlights

PlantGuard demonstrates:

* end-to-end ML project structure
* reproducible experiment tracking with MLflow
* config-driven training
* dataset-specific preprocessing
* class imbalance handling
* domain adaptation
* expanded label-space training
* external robustness evaluation
* explainability-based model auditing
* honest reporting of domain shift
* deployment-oriented project planning

## Research and Evaluation Highlights

This project intentionally avoids choosing a model based only on clean benchmark accuracy.

Important findings:

1. PlantVillage accuracy is saturated and does not meaningfully separate models.
2. External datasets reveal large robustness differences.
3. FieldPlant and PlantDoc are more useful for final model selection than PlantVillage.
4. Grad-CAM quality can change the final model choice.
5. A model with slightly lower PlantWild macro F1 can still be a better final model if it is more robust and more explainable.
6. ResNet50 Cross Entropy produced the best combined result across metrics and explanation quality.

## Next Steps

Remaining production steps:

1. Export the selected ResNet50 Cross Entropy model to ONNX.
2. Save class-name metadata alongside the exported model.
3. Compare PyTorch and ONNX Runtime predictions on the same images.
4. Build a FastAPI inference API.
5. Add `/health` and `/predict` endpoints.
6. Return top-k predictions and confidence scores.
7. Add Docker support.
8. Add selected Grad-CAM examples to the README.
9. Prepare final GitHub polish for PlantGuard v1.0.

## Project Direction

PlantGuard is being developed as a production-oriented and research-minded computer vision project.

The focus is not simply high benchmark accuracy. The focus is building a realistic ML system that includes:

* reproducible training
* experiment tracking
* external validation
* domain adaptation
* expanded label-space learning
* explainability
* failure-case analysis
* deployment preparation

The central lesson is that clean-dataset accuracy alone is not enough. A useful crop disease model must be evaluated against real-world images, external datasets, and visual explanation quality before deployment.
