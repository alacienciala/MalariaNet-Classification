# MalariaNet: Blood Smear Classification with Multi-Architecture CNN Benchmarking

A deep learning project for binary classification of malaria-infected vs. uninfected blood smear images. Seven CNN architectures are benchmarked using automated hyperparameter tuning (Optuna) and experiment tracking (Comet ML), with the best model exported to ONNX for inference.

---

## Overview

This project explores automated malaria detection from microscopy images using convolutional neural networks. The core idea is to run a fair, reproducible comparison across both custom and standard Keras architectures, letting Optuna find the best hyperparameters for each, and tracking all runs in Comet ML for analysis.

The final pipeline:
1. Loads and preprocesses a balanced train/val/test split from the [Malaria Detection Dataset](https://www.kaggle.com/datasets/....) on Kaggle
2. Runs Optuna tuning (3 trials per model) across 7 architectures
3. Retrains the best overall model on the combined train+val set
4. Exports the trained model to ONNX
5. Provides a ready-to-use `MalariaInference` class for single-image prediction

---

## Architectures Compared

| Model | Type |
|---|---|
| `simple_cnn` | Custom shallow CNN |
| `deep_cnn` | Custom 3-block CNN |
| `cellnet_cnn` | Custom CNN with BatchNorm + GAP |
| `cell_attention_block` | Custom multi-scale feature concatenation |
| `mobilenetv2` | Keras application (no pretrained weights) |
| `resnet50_custom` | Keras application (no pretrained weights) |
| `efficientnetb0` | Keras application (no pretrained weights) |

---

## Key Results

- Best model: **ResNet50** — ~63% validation accuracy
- `deep_cnn` outperformed all Keras applications on accuracy despite being a custom model
- Several models achieved 100% recall, which is medically relevant (minimizing missed positives)
- MobileNetV2 and EfficientNetB0 scored lower on accuracy but high on recall
- All results tracked and visualized via Comet ML dashboards

---

## Stack

- **TensorFlow / Keras** — model building and training
- **Optuna** — hyperparameter search with MedianPruner
- **Comet ML** — experiment tracking (metrics, params, timing)
- **tf2onnx + ONNX Runtime** — model export and CPU inference
- **scikit-learn** — evaluation (confusion matrix, classification report)

---

## Dataset

[Malaria Detection Dataset](https://www.kaggle.com/datasets/...) — blood smear images split into:
- `train/` — Uninfected & Parasitized folders
- `valid/`
- `test/`

Subsampled to 500 samples/class (train), 100/class (val/test) for faster experimentation.

---

## Inference

```python
from malaria_inference import MalariaInference

predictor = MalariaInference("best_model.onnx")
result = predictor.predict("path/to/blood_smear.jpg")
# {'label': 'Parasitized', 'confidence': 0.87}
```

---

## Limitations & Next Steps

- Only 5 epochs per Optuna trial — models may not have fully converged
- Weights trained from scratch (no transfer learning) — pretrained weights could significantly improve results
- Precision was not tracked alongside recall — worth adding for a fuller clinical picture
- Increasing trial count and epochs would likely improve best-model selection
