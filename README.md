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
│-- data/                      # Dataset and preprocessing scripts
│   │-- raw/                   # Original MRI scan images
│   │-- processed/             # Preprocessed images ready for training
│-- models/                    # Trained model checkpoints and architecture implementations
│-- notebooks/                 # Jupyter notebooks for exploratory data analysis
│-- src/                       # Source code for the deep learning pipeline
│   │-- preprocessing.py       # Data preprocessing and augmentation
│   │-- train.py               # Training script for model training
│   │-- evaluate.py            # Model evaluation and performance metrics
│   │-- inference.py           # Script for making predictions on new data
│-- results/                   # Model performance results, confusion matrices, and logs
│-- docs/                      # Project documentation and references
│-- requirements.txt           # List of required dependencies
│-- README.md                  # Project overview and setup instructions
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
python src/preprocessing.py --input data/raw --output data/processed
```

### Train Model
To train the deep learning models:
```sh
python src/train.py --config config.yaml
```

### Evaluate Model
To evaluate a trained model:
```sh
python src/evaluate.py --model models/best_model.pth
```

### Run Inference
To make predictions on new MRI scans:
```sh
python src/inference.py --image path/to/image
```

## Dataset
We use a publicly available brain MRI dataset from Kaggle:
[Brain Tumor Classification MRI Images](https://www.kaggle.com/datasets/jarvisgroot/brain-tumor-classification-mri-images)

## Ethical Considerations
Medical AI models must prioritize safety, accuracy, and transparency. To mitigate automation bias and misdiagnosis risks, our model provides probability-based results rather than definitive diagnoses. This ensures that medical professionals remain the primary decision-makers.

## Contributors
- **Jimmy Vu** – Model development & implementation
- **William Akins** – Data preprocessing & evaluation
- **Chen Zhang** – Documentation & project structure

## License
This project is licensed under the MIT License - see the LICENSE file for details.

## References
Refer to our full list of references in the `docs/` folder.

