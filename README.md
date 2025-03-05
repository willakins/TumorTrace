# Diagnosing Patients Through Magnetic Resonance Imaging

## Team Name: TumorTrace  
**Team Members:** Jimmy Vu, William Akins, Chen Zhang  
**Date:** February 10th, 2025  

## Project Overview
Magnetic Resonance Imaging (MRI) is an essential tool in medical diagnostics. However, analyzing MRI scans for accurate diagnoses requires significant expertise and time. This project aims to develop a deep learning-based model to assist in diagnosing medical conditions from MRI scans, specifically focusing on brain tumor classification. By leveraging publicly available MRI datasets, we seek to improve the efficiency and accuracy of medical image analysis.

## Features
- **Deep Learning-Based Classification**: Utilizes 3D CNNs, Inception, and ResNet models to classify MRI scans.
- **Preprocessing Pipeline**: Converts groups of 2D MRI slices into 3D data points for more accurate classification.
- **Cross-Validation**: Implements 5-fold cross-validation for robust evaluation.
- **Performance Metrics**: Evaluates models using accuracy, F1-score, and confusion matrices.
- **Ablation Studies**: Analyzes the impact of different preprocessing techniques, architecture choices, and hyperparameters.

## Directory Structure
```plaintext
TumorTrace/
│── data/                      # Contains datasets and preprocessing scripts
│   ├── raw/                   # Original MRI dataset (e.g., downloaded from Kaggle)
│   ├── processed/             # Preprocessed MRI scans (after transformations)
│   ├── data_preprocessing.py  # Script to preprocess MRI images
│
│── models/                    # Contains different model implementations
│   ├── 3D_CNN/                # 3D Convolutional Neural Network implementation
│   │   ├── model.py           # Model architecture
│   │   ├── train.py           # Training script
│   │   ├── evaluate.py        # Evaluation script
│   │   ├── checkpoints/       # Saved model weights
│   ├── Inception/             # Inception model implementation
│   │   ├── model.py
│   │   ├── train.py
│   │   ├── evaluate.py
│   │   ├── checkpoints/
│   ├── ResNet/                # ResNet model implementation
│   │   ├── model.py
│   │   ├── train.py
│   │   ├── evaluate.py
│   │   ├── checkpoints/
│
│── results/                   # Stores output reports, visualizations, and analysis
│   ├── figures/               # Plots, confusion matrices, loss curves
│   ├── performance_metrics.csv# Stores accuracy, F1-score, etc.
│
│── utils/                     # Utility functions
│   ├── image_loader.py        # MRI image loading functions
│   ├── visualization.py       # Functions for visualizing MRI scans
│
│── docs/                      # Documentation and reports
│   ├── proposal.pdf           # Original project proposal
│   ├── README.md              # Project overview and instructions
|
│── requirements.txt           # Dependencies (TensorFlow, PyTorch, etc.)
```

## Installation & Setup
1. **Clone the repository**:
   ```sh
   git clone https://github.com/yourusername/TumorTrace.git
   cd TumorTrace
   ```
2. **Set up a virtual environment (optional but recommended)**:
   ```sh
   python3 -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```
3. **Install dependencies**:
   ```sh
   pip install -r requirements.txt
   ```

## Usage
### Preprocess Data
Run the following command to preprocess the MRI scans:
```sh
python data/data_preprocessing.py --input data/raw --output data/processed
```

### Train Model
To train the deep learning models:
```sh
python models/desiredmodel/train.py --config config.yaml
```

### Evaluate Model
To evaluate a trained model:
```sh
python models/desiredmodel/evaluate.py
```

### Run Inference
To make predictions on new MRI scans:
```sh
python models/desiredmodel/inference.py --image path/to/image
```

## Dataset
We use a publicly available brain MRI dataset from Kaggle:
[Brain Tumor Classification MRI Images](https://www.kaggle.com/datasets/jarvisgroot/brain-tumor-classification-mri-images)

## Ethical Considerations
Medical AI models must prioritize safety, accuracy, and transparency. To mitigate automation bias and misdiagnosis risks, our model provides probability-based results rather than definitive diagnoses. This ensures that medical professionals remain the primary decision-makers.

## Contributors
- **Jimmy Vu** – TBD
- **William Akins** – TBD
- **Chen Zhang** – TBD
