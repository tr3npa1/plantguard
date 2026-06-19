# PlantGuard

PlantGuard is an end-to-end crop disease classification project built with PyTorch. The project trains and compares multiple deep learning models on PlantVillage, evaluates them on both clean internal test data and real-world PlantDoc images, and is being extended toward explainability, ONNX export, and API deployment.

The long-term goal is to build a production-style crop disease detection system where a user can upload a leaf image, receive a disease prediction, and view a GradCAM heatmap showing which part of the leaf influenced the model’s decision.

## Current Status

The core training and evaluation pipeline is implemented.

Completed so far:

* Kaggle API dataset download for PlantVillage
* PlantVillage train/validation/test split generation
* PlantDoc download support for external real-world evaluation
* Custom PyTorch dataset and dataloader utilities
* PlantDoc-to-PlantVillage label mapping
* Image augmentation and dataset-specific normalization
* Config-driven training pipeline
* MLflow experiment tracking
* EfficientNet-B0, EfficientNet-B3, and ResNet50 training
* CrossEntropy, weighted CrossEntropy, and focal loss comparison
* Best-checkpoint saving for each experiment
* Evaluation on PlantVillage test split
* Zero-shot external evaluation on PlantDoc
* Per-class precision, recall, F1-score, confusion matrices, and top-confusion reports

## Experiment Design

The project compares three architectures:

* EfficientNet-B0
* EfficientNet-B3
* ResNet50

Each architecture is trained with three loss functions:

* CrossEntropyLoss
* Weighted CrossEntropyLoss
* Focal Loss

This creates a 3 × 3 comparison across architecture choice and class-imbalance handling strategy.

## Datasets

### PlantVillage

PlantVillage is used for training, validation, and internal testing.

The dataset is split into:

* `data/train`
* `data/val`
* `data/test`

The model is trained only on the training split. The validation split is used for selecting the best checkpoint, and the test split is used for final internal evaluation.

### PlantDoc

PlantDoc is used as an external real-world evaluation dataset.

Unlike PlantVillage, PlantDoc contains more varied real-world images with different backgrounds, lighting, camera angles, and leaf positions. This makes it useful for testing whether a model trained on clean PlantVillage images can generalize to more realistic conditions.

Initial zero-shot evaluation on PlantDoc showed a large performance drop compared to PlantVillage, revealing a significant domain shift. This result motivates the next stage of the project: PlantDoc fine-tuning and domain adaptation.

## Evaluation

The evaluation pipeline reports:

* Accuracy
* Balanced accuracy
* Macro precision
* Macro recall
* Macro F1
* Weighted precision
* Weighted recall
* Weighted F1
* Per-class precision, recall, and F1-score
* Confusion matrix
* Normalized confusion matrix
* Top confused class pairs
* Prediction distribution

Evaluation results are logged to MLflow and also saved locally under `evaluation_results/`.

Generated evaluation outputs are not committed to Git.

## Key Finding So Far

Models achieve very high performance on the PlantVillage test split, but zero-shot performance on PlantDoc is much lower. This shows that high benchmark accuracy on clean datasets does not automatically imply real-world robustness.

This is an important project finding: PlantGuard is not just a classifier, but an experiment in measuring and improving crop disease model generalization.

## Next Steps

Planned next steps:

* Fine-tune the best PlantVillage-trained models on PlantDoc train data
* Evaluate fine-tuned models on PlantDoc test data
* Measure whether PlantDoc fine-tuning improves real-world robustness
* Check whether fine-tuning causes performance loss on PlantVillage
* Add GradCAM and GradCAM++ visual explanations
* Generate visual failure-case analysis on PlantDoc
* Export the selected model to ONNX
* Verify PyTorch vs ONNX Runtime predictions
* Build FastAPI inference endpoints
* Add Docker support
* Deploy a live demo

## Planned Repository Structure

```text
plantguard/
├── data/
│   ├── download.py
│   └── dataset.py
├── training/
│   ├── train.py
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

## Tech Stack

* Python
* PyTorch
* torchvision
* scikit-learn
* pandas
* matplotlib
* MLflow
* FastAPI
* ONNX Runtime
* Docker

## Project Direction

PlantGuard is being developed as a production-oriented computer vision project. The focus is not only on achieving high accuracy, but also on:

* robust evaluation
* class imbalance handling
* external dataset testing
* model explainability
* deployment readiness
* reproducible experiment tracking
* honest reporting of domain shift and failure cases
