# Chest X-ray Pneumonia Classification using MobileNetV2

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.10%2B-orange)
![License](https://img.shields.io/badge/License-MIT-green)
![Status](https://img.shields.io/badge/Status-Active-success)

A lightweight deep learning solution for classifying chest X-ray images to detect pneumonia using **MobileNetV2**, an efficient pretrained model optimized for medical image analysis.

## 🎯 Project Overview

This project implements an automated pneumonia detection system from chest X-ray images using transfer learning with MobileNetV2. The model classifies X-rays into two categories:
- **NORMAL**: Healthy chest X-rays
- **PNEUMONIA**: X-rays showing pneumonia

## 📊 Dataset


### 📥 Dataset Setup

**Important**: The dataset is NOT included in this repository due to size constraints.

#### Download Instructions:

1. **Download the dataset** from Kaggle:
   - 🔗 [Chest X-ray Images (Pneumonia) Dataset](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia)
   - You'll need a Kaggle account (free)

2. **Extract the dataset**:
   ```bash
   # Extract to the project directory
   unzip chest-xray-pneumonia.zip -d chest_xray/
   ```

3. **Verify the structure**:
   ```
   chest_xray/chest_xray/
   ├── train/
   │   ├── NORMAL/      (1,341 images)
   │   └── PNEUMONIA/   (3,875 images)
   ├── val/
   │   ├── NORMAL/      (8 images)
   │   └── PNEUMONIA/   (8 images)
   └── test/
       ├── NORMAL/      (234 images)
       └── PNEUMONIA/   (390 images)
   ```

4. **Dataset Statistics**:
   - Total Images: 5,856
   - Training: 5,216 images
   - Validation: 16 images
   - Test: 624 images
   - Image Size: Variable (will be resized to 224×224)

- **Source**: Chest X-ray Images (Pneumonia) Dataset
- **Classes**: 2 (NORMAL, PNEUMONIA)
- **Training samples**: 5,216 images
- **Validation samples**: 16 images  
- **Test samples**: 624 images
- **Image format**: JPEG/PNG chest X-ray images

## 🚀 Features

✅ **Lightweight Model**: MobileNetV2 (~2.4M parameters) - fast and efficient  
✅ **Transfer Learning**: Pretrained on ImageNet for better feature extraction  
✅ **Data Augmentation**: Rotation, shift, zoom, flip for robust training  
✅ **Two-Phase Training**: Classification head training + fine-tuning  
✅ **Comprehensive Metrics**: Accuracy, Precision, Recall, AUC, F1-Score  
✅ **Easy Prediction**: Single image or batch inference  
✅ **Visualization**: Training history and prediction results  

## 📁 Project Structure

```
Chest x ray pneumonia/
├── train_pneumonia_classifier.py    # Main training script
├── predict_pneumonia.py              # Prediction/inference script
├── evaluate_pneumonia.py             # Model evaluation script
├── requirements.txt                  # Python dependencies
├── README.md                         # This file
├── chest_xray/                       # Dataset directory
│   └── chest_xray/
│       ├── train/
│       │   ├── NORMAL/
│       │   └── PNEUMONIA/
│       ├── val/
│       │   ├── NORMAL/
│       │   └── PNEUMONIA/
│       └── test/
│           ├── NORMAL/
│           └── PNEUMONIA/
└── models/                           # Saved models (created during training)
    └── pneumonia_classifier_YYYYMMDD_HHMMSS/
        ├── best_model.h5
        ├── final_model.h5
        ├── class_names.json
        ├── history.json
        └── training_history.png
```

## 📥 Pre-trained Model Download

**Important**: Due to GitHub's file size limitations (100MB), the trained model files (.h5) are NOT included in this repository.

### Model Information:

- **Model Name**: `pneumonia_classifier_20260527_011206`
- **Architecture**: MobileNetV2 + Custom Classification Head
- **Model Size**: ~9 MB (best_model.h5), ~9 MB (final_model.h5)
- **Training Date**: May 27, 2026

### Performance Metrics:

| Metric | Training | Validation |
|--------|----------|------------|
| Accuracy | 95.44% | 75.00% |
| Precision | 95.44% | 75.00% |
| Recall | 95.44% | 75.00% |
| AUC | 98.96% | 85.94% |
| Loss | 0.129 | 0.691 |

### Download Options:

#### Option 1: Train Your Own Model (Recommended)
```bash
# Follow the training instructions below
python train_pneumonia_classifier.py
```

#### Option 2: Download Pre-trained Model
If you want to use the pre-trained model without training:

1. **Kaggle**: [https://www.kaggle.com/models/google/mobilenet-v2] 

After downloading, place the model files in:
```
models/pneumonia_classifier_20260527_011206/
├── best_model.h5           # Download this
├── final_model.h5          # Download this
├── class_names.json        # Already in repo
├── history.json            # Already in repo
└── training_history.png    # Already in repo
```

### Using the Pre-trained Model:

```bash
# Predict with downloaded model
python predict_pneumonia.py \
    --model models/pneumonia_classifier_20260527_011206/best_model.h5 \
    --image path/to/xray.jpg \
    --visualize

# Evaluate on test set
python evaluate_pneumonia.py \
    --model models/pneumonia_classifier_20260527_011206/best_model.h5 \
    --test_dir chest_xray/chest_xray/test
```


## 🛠️ Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Required Packages
- TensorFlow >= 2.10.0
- NumPy >= 1.21.0
- OpenCV >= 4.5.0
- Matplotlib >= 3.4.0
- Scikit-learn >= 1.0.0
- Pandas >= 1.3.0
- Seaborn >= 0.11.0

## 🎓 Model Architecture

```
Input (224×224×3)
    ↓
MobileNetV2 Base (Pretrained on ImageNet)
    ↓
Global Average Pooling
    ↓
Dropout (0.3)
    ↓
Dense (128, ReLU)
    ↓
Batch Normalization
    ↓
Dropout (0.4)
    ↓
Dense (2, Softmax) → [NORMAL, PNEUMONIA]
```

**Model Statistics:**
- Total Parameters: 2,422,722
- Trainable Parameters: 164,482 (Phase 1)
- Input Size: 224×224×3
- Output: 2 classes (NORMAL, PNEUMONIA)

## 🏋️ Training

### Quick Start

```bash
python train_pneumonia_classifier.py
```

### Training Process

The training happens in **two phases**:

#### Phase 1: Train Classification Head (30 epochs)
- Freeze MobileNetV2 base model
- Train only custom classification layers
- Learning rate: 1e-3
- Optimizer: Adam

#### Phase 2: Fine-tune Top Layers (20 epochs)
- Unfreeze top 20 layers of MobileNetV2
- Fine-tune with lower learning rate
- Learning rate: 1e-5
- Optimizer: Adam

### Training Configuration

Edit `train_pneumonia_classifier.py` to customize:

```python
DATA_DIR = "chest_xray/chest_xray"  # Dataset path
IMG_SIZE = (224, 224)                # Input image size
BATCH_SIZE = 32                      # Batch size
EPOCHS = 30                          # Phase 1 epochs
FINE_TUNE_EPOCHS = 20               # Phase 2 epochs
FINE_TUNE_LAYERS = 20               # Layers to unfreeze
```

### Training Output

```
models/pneumonia_classifier_20260527_011206/
├── best_model.h5              # Best model (highest val accuracy)
├── final_model.h5             # Final model after all training
├── class_names.json           # Class labels
├── history.json               # Training metrics history
└── training_history.png       # Training curves visualization
```

### Callbacks

- **ModelCheckpoint**: Saves best model based on validation accuracy
- **EarlyStopping**: Stops training if no improvement (patience=10)
- **ReduceLROnPlateau**: Reduces learning rate on plateau (factor=0.5, patience=5)

## 🔮 Prediction

### Single Image Prediction

```bash
python predict_pneumonia.py \
    --model models/pneumonia_classifier_20260527_011206/best_model.h5 \
    --image path/to/xray.jpg \
    --visualize
```

**Output:**
```
Prediction Result:
  Class: PNEUMONIA
  Confidence: 94.23%

  All Probabilities:
    NORMAL: 5.77%
    PNEUMONIA: 94.23%
```

### Batch Prediction

```bash
python predict_pneumonia.py \
    --model models/pneumonia_classifier_20260527_011206/best_model.h5 \
    --directory path/to/xray_images/ \
    --output predictions.json
```

### Using in Python Code

```python
from predict_pneumonia import PneumoniaPredictor

# Initialize predictor
predictor = PneumoniaPredictor(
    model_path="models/pneumonia_classifier_20260527_011206/best_model.h5"
)

# Predict single image
result = predictor.predict_single("xray.jpg", return_probs=True)
print(f"Prediction: {result['class']} ({result['confidence']:.2%})")

# Visualize prediction
predictor.visualize_prediction("xray.jpg", save_path="prediction.png")
```

## 📈 Evaluation

### Run Evaluation

```bash
python evaluate_pneumonia.py \
    --model models/pneumonia_classifier_20260527_011206/best_model.h5 \
    --test_dir chest_xray/chest_xray/test \
    --output evaluation_results
```

### Evaluation Metrics

The evaluation script generates:

1. **Basic Metrics**
   - Loss
   - Accuracy
   - Precision
   - Recall
   - AUC (Area Under Curve)

2. **Confusion Matrix**
   - Counts and percentages
   - Visual heatmap

3. **Classification Report**
   - Per-class metrics
   - Support (number of samples)
   - Macro and weighted averages

### Evaluation Output

```
evaluation_results/
├── confusion_matrix.png           # Confusion matrix visualization
├── classification_report.txt      # Detailed text report
├── classification_report.csv      # Metrics in CSV format
└── evaluation_summary.json        # Summary statistics
```

## 📊 Expected Results

Based on the Chest X-ray Pneumonia dataset:

| Metric | Expected Range |
|--------|---------------|
| Training Accuracy | 90-95% |
| Validation Accuracy | 85-92% |
| Test Accuracy | 85-90% |
| Precision (PNEUMONIA) | 88-93% |
| Recall (PNEUMONIA) | 90-95% |
| AUC | 0.92-0.96 |

**Note**: Results may vary based on:
- Dataset quality and distribution
- Training hyperparameters
- Hardware (CPU vs GPU)
- Random initialization

## 💡 Tips for Best Results

### Data Preparation
- Ensure balanced classes or use class weights
- Use high-quality X-ray images
- Remove corrupted or low-quality images
- Verify correct folder structure

### Training
- **GPU Recommended**: Training on GPU is 10-50x faster
- **Batch Size**: Increase if you have more GPU memory (16, 32, 64)
- **Monitor Metrics**: Watch for overfitting (val_loss increasing)
- **Early Stopping**: Let the callback stop training automatically

### Hyperparameter Tuning

```python
# Experiment with:
IMG_SIZE = (224, 224)      # or (192, 192) for faster training
BATCH_SIZE = 32            # 16, 32, 64
EPOCHS = 30                # 20-50
FINE_TUNE_LAYERS = 20      # 10-30
Dropout rates = 0.3, 0.4   # 0.2-0.5
```

## 🔧 Advanced Usage

### Custom Data Augmentation

Edit `train_pneumonia_classifier.py`:

```python
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,        # Increase for more rotation
    width_shift_range=0.2,    # Horizontal shift
    height_shift_range=0.2,   # Vertical shift
    shear_range=0.2,          # Shear transformation
    zoom_range=0.2,           # Zoom in/out
    horizontal_flip=True,     # Mirror flip
    brightness_range=[0.8, 1.2],  # Brightness adjustment
    fill_mode='nearest'
)
```

### Transfer Learning from Your Model

```python
from tensorflow import keras

# Load your trained model
base_model = keras.models.load_model('best_model.h5')

# Freeze all layers
for layer in base_model.layers:
    layer.trainable = False

# Add new classification head for different task
# ... (add your custom layers)
```

## 🐛 Troubleshooting

### Out of Memory Error
```python
# Reduce batch size
BATCH_SIZE = 16  # or 8
```

### Slow Training
- Use GPU if available
- Reduce image size: `IMG_SIZE = (192, 192)`
- Reduce batch size if using CPU
- Close other applications

### Poor Performance
- Check class balance in dataset
- Increase training epochs
- Try different fine-tuning layers (10-30)
- Verify data quality
- Check for data leakage

### Import Errors
```bash
# Reinstall dependencies
pip install --upgrade -r requirements.txt
```

### Protobuf Version Error
```bash
# Upgrade protobuf
pip install --upgrade protobuf
```

## 📚 References

- **MobileNetV2 Paper**: [MobileNetV2: Inverted Residuals and Linear Bottlenecks](https://arxiv.org/abs/1801.04381)
- **TensorFlow Documentation**: https://www.tensorflow.org/
- **Keras Applications**: https://keras.io/api/applications/

## 🤝 Contributing

Contributions are welcome! Feel free to:
- Report bugs
- Suggest improvements
- Add new features
- Share your results

## 📄 License

This project is provided as-is for educational and research purposes.

## 🙏 Acknowledgments

- **MobileNetV2**: Sandler et al., Google Research
- **TensorFlow/Keras**: Google Brain Team
- **Dataset**: Chest X-ray Images (Pneumonia) Dataset

## 📞 Support

For issues or questions:
1. Check the troubleshooting section
2. Review the code comments
3. Verify dataset structure
4. Check TensorFlow/Keras documentation

---

## 🎯 Quick Command Reference

```bash
# Install dependencies
pip install -r requirements.txt

# Train model
python train_pneumonia_classifier.py

# Predict single image
python predict_pneumonia.py --model models/best_model.h5 --image xray.jpg --visualize

# Predict directory
python predict_pneumonia.py --model models/best_model.h5 --directory images/ --output results.json

# Evaluate model
python evaluate_pneumonia.py --model models/best_model.h5 --test_dir chest_xray/chest_xray/test
```

---

**Happy Classifying! 🏥🔬**

*Automated Pneumonia Detection from Chest X-rays*
