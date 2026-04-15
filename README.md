# Pneumonia Detection from Chest X-Rays Using Deep Learning

![Python](https://img.shields.io/badge/Python-3.11-blue) ![TensorFlow](https://img.shields.io/badge/TensorFlow-2.16.2-orange) ![Accuracy](https://img.shields.io/badge/Accuracy-88.0%25-green)

Machine learning coursework project that trains a deep learning classifier to detect pneumonia from pediatric chest X-ray images. Compares transfer learning (ResNet50, VGG16) against training from scratch.

---

## Results

| Model | Accuracy | Pneumonia Recall | Normal Recall |
|---|---|---|---|
| ResNet50 (ImageNet pretrained) | **88.0%** | 95.9% | 74.8% |
| VGG16 (ImageNet pretrained) | 85.1% | 86.4% | 82.9% |
| ResNet50 (from scratch) | 83.5% | 82.3% | 85.5% |

ResNet50 with ImageNet pretraining achieves the best overall accuracy and highest pneumonia recall, making it well-suited for clinical screening where minimizing missed cases is critical.

---

## Dataset

[Kaggle Chest X-Ray Images (Pneumonia)](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia) — pediatric chest X-rays from Guangzhou Women and Children's Medical Center.

- 5,216 training images
- 624 test images
- 16 validation images
- ~72% pneumonia, ~28% normal (class imbalanced)

> The dataset is not included in this repo. Download it from Kaggle and place it at `chest_xray/chest_xray/` with `train/`, `val/`, and `test/` subdirectories, each containing `NORMAL/` and `PNEUMONIA/` folders.

---

## Project Pipeline

1. **Data preprocessing** — resize to 224×224, rescale pixels to 0–1
2. **Augmentation** — random flip, zoom, rotation on training set only
3. **Transfer learning** — ResNet50 pretrained on ImageNet, top layer removed
4. **Two-phase training:**
   - Phase 1: freeze base, train custom classification head
   - Phase 2: unfreeze last 30 layers, fine-tune at LR/10
5. **Class weighting** — Normal weighted 3× to counter 72/28 imbalance
6. **Evaluation** — accuracy, recall, confusion matrix, ROC/AUC
7. **Comparison** — VGG16 and ResNet50 from scratch baselines

---

## Model Architecture

```
ResNet50 base (pretrained, ImageNet)
        ↓
GlobalAveragePooling2D
        ↓
Dense(128, relu)
        ↓
Dropout(0.5)
        ↓
Dense(1, sigmoid)  →  0 = Normal, 1 = Pneumonia
```

---

## Requirements

```bash
pip install tensorflow numpy matplotlib seaborn scikit-learn
```

Tested on Python 3.11, TensorFlow 2.16.2, Apple M4 Max (Metal GPU backend).

---

## Usage

1. Clone the repo and download the dataset from Kaggle
2. Update `BASE_DIR` in the notebook to point to your local dataset path
3. Run all cells in `Pneumonia_Final_ML_Asma.ipynb`

---

## Key Findings

- Transfer learning converges ~4× faster than training from scratch (~45s/epoch vs ~130s/epoch)
- Fine-tuning the upper ResNet layers is essential — phase 1 alone reaches only ~52% accuracy
- ResNet50 (ImageNet) achieves highest pneumonia recall (95.9%), minimizing missed diagnoses
- VGG16 provides the most balanced recall across both classes (86.4% / 82.9%)
- Scratch-trained ResNet50 shows highly unstable validation AUC (ranging 0.42–0.94 across epochs), indicating unreliable generalization despite reasonable test accuracy

---

## Limitations

- Validation set is only 16 images — val metrics are noisy
- Single-hospital dataset may not generalize across different X-ray equipment
- Binary classification only (does not distinguish bacterial vs viral pneumonia)
- Decision threshold not optimized for clinical sensitivity targets

---

## Future Work

- Larger validation split for more reliable early stopping
- Grad-CAM visualization to highlight lung regions driving predictions
- Threshold tuning to optimize clinical recall targets
- Multi-class extension: bacterial vs viral pneumonia
- Ensemble methods combining ResNet50 and VGG16
