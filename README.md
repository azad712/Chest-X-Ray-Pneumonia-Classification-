# Chest X-Ray Pneumonia Classification

A MobileNetV2-based image classifier for two labels: `NORMAL` and `PNEUMONIA`. This repository is intended for learning and research, not clinical use or diagnosis.

## Dataset

The dataset is not included. Download the [Chest X-Ray Images (Pneumonia) dataset from Kaggle](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia) and place it so the folders are arranged as follows:

```text
chest_xray/chest_xray/
├── train/{NORMAL,PNEUMONIA}/
├── val/{NORMAL,PNEUMONIA}/
└── test/{NORMAL,PNEUMONIA}/
```

## Setup

Requires Python 3.8 or newer. From the repository directory:

```bash
python -m venv .venv
```

Activate the environment, then install dependencies:

```bash
# Windows PowerShell
.venv\Scripts\Activate.ps1

# macOS/Linux
source .venv/bin/activate

pip install -r requirements.txt
```

## Train

```bash
python train_pneumonia_classifier.py
```

The script reads the dataset from `chest_xray/chest_xray` and saves model outputs under `models/`.

## Predict

Download the optional pretrained model [from Google Drive](https://drive.google.com/file/d/19ET2TCEcbegPWtgR2mZ7ogVO1BxQ7_tl/view?usp=sharing), then run:

```bash
python predict_pneumonia.py \
  --model models/pneumonia_classifier_20260527_011206/best_model.h5 \
  --image path/to/xray.jpg \
  --visualize
```

For a directory of images, use `--directory path/to/images/` and optionally `--output predictions.json` instead of `--image`.

## Evaluate

```bash
python evaluate_pneumonia.py \
  --model models/pneumonia_classifier_20260527_011206/best_model.h5 \
  --test_dir chest_xray/chest_xray/test \
  --output evaluation_results
```

The model uses transfer learning with MobileNetV2 and 224×224 input images. Predictions are experimental and must not be used to make medical decisions.
