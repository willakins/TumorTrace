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
│   ├── processed/             # Preprocessed MRI scans
│   │    ├── test/             # mri images used for testing
│   │    └── train/            # mri images used for training
│   ├── data_transforms.py     # Functions that allow for creation of synthetic data
|   └── image_loader.py        # Dataset class for mri images
|
│── docs/                      # Documentation and reports
│   ├── proposal.pdf           # Original project proposal
│   ├── midterm.pdf            # Midterm project checkpoint
│   └── README.md              # Project overview and instructions
|
│── environments/              # Stores any environment files used
│   └── environment.yaml       # Main environment file to run ipynb
│
│── notebooks/                 # Stores ipynb that we use
│   └── main.ipynb             # Main notebook for running models
│
│── results/                   # Stores output reports, visualizations, and analysis
│
│── src/                       # Contains different model implementations
│   ├── models/                # 3D Convolutional Neural Network implementation
│   │   ├── CNN_3D/            # Home for all files related to 3D convolutional neural network model
│   │   │   └── model.py       # Class implementation of the model
│   │   ├── ResNet/            # Home for all files related to ResNet model
│   │   │   └── model.py       # Class implementation of the model
│   │   ├── Inception/         # Home for all files related to Inception model
│   │   │   └── model.py       # Class implementation of the model
│   ├── optimizer.py           # Helper function for creating an optimizer
│   └── runner.py              # Class for training the models
|
│── utils/                     # Utility functions
│   ├── confusion_matrix.py    # Functions for creating & visualizing confusion matrices
│   ├── dataset_utils.py       # Functions for retrieving and processing the dataset
│   └── utils.py               # Misc. general use functions
│
│── .gitignore                 # Git files not to track
```

## Installation & Setup

1. **Clone the repository**:
   ```sh
   git clone https://github.com/yourusername/TumorTrace.git
   cd TumorTrace
   ```
2. **Install dependencies**:
   ```sh
   conda env create -f environments/environment.yaml
   ```
3. **Activate Environment**:
   ```sh
   conda activate TumorTrace           
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
