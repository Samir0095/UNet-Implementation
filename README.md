# Brain Tumor Segmentation using 2D U-Net (BraTS 2018)

[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c.svg)](https://pytorch.org/)
[![Python](https://img.shields.io/badge/Python-3.8%2B-3776ab.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

A Deep Learning pipeline built with PyTorch for automated semantic segmentation of brain tumors from 4-modality 3D MRI scans using a custom 2D U-Net architecture. Evaluated on the **MICCAI BraTS 2018** (Brain Tumor Segmentation) dataset.

---

## 📌 Project Overview

Brain tumor segmentation from Magnetic Resonance Imaging (MRI) is essential for clinical diagnosis, treatment planning, and monitoring. This project implements an end-to-end deep learning framework that:
- Processes multi-modal 3D MRI scans (`FLAIR`, `T1`, `T1CE`, `T2`).
- Applies patient-level dataset splitting to eliminate data leakage across 2D axial slices.
- Normalizes individual MRI modalities slice-by-slice using foreground Z-score intensity normalization.
- Trains a custom 2D U-Net model using a combined **Dice + Binary Cross-Entropy (BCE)** loss function.
- Evaluates segmentation metrics (**Dice Score**, **IoU / Jaccard Index**, **Precision**, **Recall / Sensitivity**, **Specificity**).
- Includes decision threshold analysis and generates 5-panel side-by-side visual comparison plots.

---

## 🔬 Model Architecture

The segmentation model is based on the **U-Net** architecture (Ronneberger et al.), adapted for 4-channel multi-sequence 2D MRI inputs:

- **Input Shape**: `(Batch, 4, 240, 240)` representing `[FLAIR, T1, T1CE, T2]` modalities.
- **Encoder Path**:
  - `Encoder 1`: 4 → 16 channels (`DoubleConv`: Conv3x3 + BatchNorm + ReLU)
  - `Encoder 2`: 16 → 32 channels
  - `Encoder 3`: 32 → 64 channels
  - `Encoder 4`: 64 → 128 channels
- **Bottleneck**: 128 → 256 channels
- **Decoder Path**:
  - Transposed Convolution upsampling concatenated with encoder skip connections.
  - `Decoder 4`: 256 → 128 channels
  - `Decoder 3`: 128 → 64 channels
  - `Decoder 2`: 64 → 32 channels
  - `Decoder 1`: 32 → 16 channels
- **Output**: 1x1 Convolution mapping 16 channels to a single binary logit segmentation map.

---

## 📁 Repository Structure

```
BrainTumor_UNet/
├── Datasets/
│   └── BraTS2018/                         # BraTS 2018 Dataset (NIfTI format)
│       ├── MICCAI_BraTS_2018_Data_Training/
│       │   ├── HGG/                        # High-Grade Glioma patient folders
│       │   └── LGG/                        # Low-Grade Glioma patient folders
│       └── MICCAI_BraTS_2018_Data_Validation/
└── UNet/
    ├── check_gpu.py                        # System & CUDA GPU diagnostic check
    ├── dataset.py                          # PyTorch BraTSDataset class & slice loader
    ├── dataloader.py                       # Patient-level split & DataLoader factory
    ├── model.py                            # PyTorch 4-channel U-Net neural network
    ├── split_dataset.py                    # Patient train/val split & leakage verification
    ├── train.py                            # Model training loop (Dice + BCE loss)
    ├── evaluate.py                         # Validation evaluation pipeline
    ├── threshold_analysis.py               # Post-processing decision threshold optimization
    ├── analyze_slices.py                   # Dataset slice distribution & pixel ratio analysis
    ├── visualize_predictions.py            # Overlay visualization figure generator
    ├── best_model.pth                      # Trained U-Net model checkpoint
    └── prediction_visualizations/          # Generated output overlay images (.png)
```

---

## 📊 Dataset & Preprocessing

The project utilizes the **MICCAI BraTS 2018** training set containing **HGG** (High-Grade Glioma) and **LGG** (Low-Grade Glioma) patient data.

### Data Processing Pipeline
1. **Patient-Level Split**: Patients are randomly assigned (80% train / 20% validation, Seed 42). All 2D axial slices of a single patient remain strictly within the same split to avoid data leakage.
2. **Modality Stacking**: For each slice, 4 MRI modalities (`FLAIR`, `T1`, `T1-Contrast-Enhanced`, `T2`) are stacked into a 4-channel input tensor.
3. **Z-score Normalization**: Brain foreground intensity values per modality slice are normalized using zero-mean and unit-variance ($\frac{x - \mu}{\sigma}$), keeping background pixels at zero.
4. **Binary Target Creation**: Multi-class BraTS ground truth annotations (labels 1, 2, 4) are converted into a binary tumor mask (`1 = Tumor`, `0 = Background`).

---

## ⚡ Installation & Setup

### Prerequisites
- Python 3.8 or higher
- NVIDIA GPU with CUDA support (Recommended for training)

### Install Dependencies
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install nibabel numpy matplotlib tqdm
```

---

## 🚀 Usage Guide

Navigate to the `UNet` workspace directory:
```bash
cd UNet
```

### 1. Verify GPU & Environment Setup
Check CUDA availability, PyTorch version, and VRAM memory:
```bash
python check_gpu.py
```

### 2. Verify Dataset Split & Leakage Prevention
Confirm patient-level division and ensure zero overlap between train and validation splits:
```bash
python split_dataset.py
```

### 3. Analyze Dataset Slices
Calculate total slice counts, tumor vs empty slice ratios, and pixel statistics:
```bash
python analyze_slices.py
```

### 4. Train the U-Net Model
Start model training using combined Dice + BCE Loss:
```bash
python train.py
```
*The best performing model weights are automatically saved to `best_model.pth` based on validation Dice score.*

### 5. Evaluate Model Performance
Run full metric evaluation across validation set slices:
```bash
python evaluate.py
```

### 6. Perform Threshold Analysis
Sweep decision probability thresholds from 0.10 to 0.90 to optimize segmentation accuracy:
```bash
python threshold_analysis.py
```

### 7. Generate Prediction Visualizations
Export 5-panel side-by-side figures (FLAIR, Ground Truth, Prediction, Prediction Probability, Ground Truth + Prediction Overlay):
```bash
python visualize_predictions.py
```
Visualizations are saved to `UNet/prediction_visualizations/`.

---

## 📈 Evaluation Metrics

The segmentation performance is measured using standard medical imaging metrics:

- **Dice Score / F1-Score**: Measure of overlap between predicted tumor mask and ground truth annotation.
  $$\text{Dice} = \frac{2 \times TP}{2 \times TP + FP + FN}$$
- **IoU (Intersection over Union / Jaccard Index)**: Ratio of overlap area to union area.
  $$\text{IoU} = \frac{TP}{TP + FP + FN}$$
- **Precision**: Accuracy of positive tumor pixel predictions.
- **Recall (Sensitivity)**: Proportion of true tumor pixels correctly segmented.
- **Specificity**: Proportion of non-tumor/background pixels correctly identified.

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
