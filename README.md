# Egg Sex Classification via LSCI and Deep Neural Networks

Code for the paper:

> **Detection of non-invasive sexing of early chick embryos in intact eggs using Laser Speckle Contrast Imaging (LSCI) and Deep Neural Network (DNN)**
> Simon Mahler\*, Anika Arora\*, Carol Readhead\*, Siyuan Yin\*, et al.
> California Institute of Technology

---

## Overview

This repository contains the preprocessing and model training code used to evaluate whether laser speckle contrast imaging (LSCI) of extraembryonic blood vessels can be used to classify the sex of early-stage chicken embryos. Two DNN architectures were evaluated — ResNetBiT and YOLOv5 — using 5-fold cross-validation on a dataset of 1,251 eggs imaged at day 3 and day 4 of incubation.

**Result:** Neither model achieved accuracy significantly above chance, suggesting that vascular patterns alone are insufficient for reliable early sex identification.

---

## Repository Structure

```
├── 1_data_preprocessing.ipynb          # Raw image → pickle pipeline
├── 2_resnetbit_cross_validation.ipynb  # ResNetBiT 5-fold CV training
├── 3_yolov5_classification.ipynb       # YOLOv5 5-fold CV training
├── GT_labels_eggs_HH19.csv             # Ground-truth sex labels for HH19 dataset
├── GT_labels_eggs_HH25.csv             # Ground-truth sex labels for HH25 dataset
└── README.md
```

---

## Requirements

All notebooks are designed to run on **Google Colab** with GPU acceleration enabled (`Runtime → Change runtime type → GPU`).

Dependencies are installed at the top of each notebook. Key packages:

| Package | Purpose |
|---|---|
| `timm` | ResNetBiT pretrained model |
| `torchmetrics` | AUROC metric |
| `torcheval` | BinaryAccuracy metric |
| `torch_optimizer` | Lookahead optimizer (available but not used in final runs) |
| `ultralytics/yolov5` | YOLOv5 classification |

---

## Data

The ground-truth label files (`GT_labels_eggs_HH19.csv`, `GT_labels_eggs_HH25.csv`) are included in this repository.

CSV columns: `image_number, label (0=male/1=female), egg_id, grade, real, stage, sub_folder`

The raw LSCI images (~28 GB) are not included. Images are 16-bit PNG files (~4096×3000 px) organized under `HH19/` and `HH25/` subdirectories, with paths specified per image in the CSV files.

**Data availability:** The raw image dataset is available from the corresponding author upon reasonable request.

---

## How to Run

### Step 1 — Preprocess images (`1_data_preprocessing.ipynb`)

Converts raw 16-bit LSCI images into processed pickle files.

**Run twice** with different `IMG_SIZE` values to produce datasets for each model:

| Run | `IMG_SIZE` | Used by |
|---|---|---|
| 1st | `256` | ResNetBiT |
| 2nd | `640` | YOLOv5 |

**Pipeline per image:**
1. Convert 16-bit → 8-bit
2. Center-crop to square
3. Zoom to the brightest region (proxy for embryonic heart)
4. Resize to `IMG_SIZE × IMG_SIZE`

Output: `./processed_data/{HH19,HH25}/{training,validation,testing}/` pickle files, saved to Google Drive.

### Step 2a — Train ResNetBiT (`2_resnetbit_cross_validation.ipynb`)

- Loads 256×256 pickle files
- Adapts pretrained `resnetv2_50x1_bit.goog_in21k` for single-channel grayscale input
- Trains with 5-fold cross-validation (splits by egg ID to prevent leakage)
- Augmentation: random rotation ±180°, translation ±5%, scaling ×0.9–1.1
- Enhancement: unsharp masking applied at load time
- Early stopping after 10 epochs without improvement (starts at epoch 40), max 100 epochs
- Saves training curves and confusion matrices

### Step 2b — Train YOLOv5 (`3_yolov5_classification.ipynb`)

- Loads 640×640 pickle files and writes them to a YOLOv5-compatible directory structure
- Fine-tunes `yolov5s-cls.pt` for binary classification (male/female)
- Trains with 5-fold cross-validation (splits by egg ID)
- Max 300 epochs per fold

---

## Key Design Decisions

**Split by egg ID:** All images of the same egg are placed in the same fold. This prevents data leakage, since a single egg can have multiple images taken at different developmental stages.

**Per-egg prediction aggregation (ResNetBiT):** At evaluation time, predictions across all images of the same egg are aggregated — mean probability for eggs with an even number of images, majority voting for eggs with an odd number.

**Grayscale adaptation:** ResNetBiT's first convolutional layer (originally 3-channel) is replaced with a 1-channel layer initialized by averaging the pretrained RGB weights.

---

## Results Summary

5-fold cross-validated accuracy:

| Model | Day 3 | Day 4 |
|---|---|---|
| ResNetBiT | 59% ± 5% (p < 0.3) | 61% ± 3% (p < 0.04) |
| YOLOv5 | 55% ± 3% (p < 0.3) | 53% ± 3% (p < 0.5) |

Neither model achieves accuracy significantly above chance across all folds, indicating that LSCI-derived vascular patterns alone are not sufficient for reliable early sex identification.
