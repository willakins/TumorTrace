"""
Utilities that make retrieving and processing the dataset easier
"""

import os
import numpy as np
import pickle
from PIL import Image
from typing import Tuple
from sklearn.model_selection import train_test_split
import zipfile
import subprocess

from utils.utils import compute_mean_and_std


def download_kaggle_dataset(dataset: str, download_path: str):
    if not os.path.exists(download_path):
        os.makedirs(download_path)
    
    print(f"[INFO] Downloading Kaggle dataset: {dataset} to {download_path}")
    result = subprocess.run(
        ["kaggle", "datasets", "download", "-d", dataset, "-p", download_path],
        capture_output=True,
        text=True
    )

    if result.returncode != 0:
        raise RuntimeError(f"Failed to download dataset from Kaggle:\n{result.stderr}")

    # Find and unzip the dataset file
    for file in os.listdir(download_path):
        if file.endswith(".zip"):
            zip_path = os.path.join(download_path, file)
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(download_path)
            print(f"[INFO] Unzipped {file} into {download_path}")
            os.remove(zip_path)
            break
def check_kaggle_auth():
    kaggle_path = os.path.expanduser("~/.kaggle/kaggle.json")
    if not os.path.exists(kaggle_path):
        raise FileNotFoundError(
            "Kaggle API token not found. Please follow these steps:\n"
            "1. Visit https://www.kaggle.com/account\n"
            "2. Click 'Create New API Token'\n"
            "3. type the following into your terminal if on MacOS/Linux:\n"
            "mkdir -p ~/.kaggle\n"
            "mv /path/to/downloaded/kaggle.json ~/.kaggle/\n"
            "chmod 600 ~/.kaggle/kaggle.json\n"
            "4. type the following into your terminal if on Windows:\n"
            "mkdir %USERPROFILE%\.kaggle\n"
            "move %USERPROFILE%\Downloads\kaggle.json %USERPROFILE%\.kaggle\ \n"
        )
    else:
        print("[Info] Kaggle API token found.")

def prepare_dataset(raw_data_path: str, pickle_path: str, processed_data_path: str) -> Tuple[float, float]:
    # Step 0: Make sure the user has a Kaggle API key
    check_kaggle_auth()

    # Step 1: Download raw dataset from Kaggle if needed
    if not os.path.exists(raw_data_path) or not os.listdir(raw_data_path):
        download_kaggle_dataset(
            dataset="jarvisgroot/brain-tumor-classification-mri-images",
            download_path=raw_data_path
        )
    else:
        print("[INFO] Raw data already exists. Skipping Kaggle download.")

    # Step 2: Convert pickle to folder structure
    if not os.path.exists(processed_data_path):
        print("[INFO] Converting pickle to folder structure...")
        convert_pickle_to_folder(pickle_path, processed_data_path)
    else:
        print("[INFO] Processed data directory already exists. Skipping conversion.")

    # Step 3: Load or compute dataset mean and std
    mean_std_cache = os.path.join(processed_data_path, "mean_std.npy")
    if os.path.exists(mean_std_cache):
        print("[INFO] Loading cached dataset mean and std...")
        dataset_mean, dataset_std = np.load(mean_std_cache)
    else:
        print("[INFO] Computing dataset mean and std from scratch...")
        dataset_mean, dataset_std = compute_mean_and_std(processed_data_path)
        np.save(mean_std_cache, np.array([dataset_mean, dataset_std]))
        print(f"[INFO] Mean: {dataset_mean:.4f}, Std: {dataset_std:.4f} (saved to cache)")

    return dataset_mean, dataset_std


def convert_pickle_to_folder(pickle_path, output_dir):
    if os.path.exists(output_dir):
        print(f"[INFO] Skipping conversion. Folder '{output_dir}' already exists.")
        return
    
    with open(pickle_path, 'rb') as f:
        data = pickle.load(f)  # List of (image, label) tuples

    images, labels = zip(*data)
    X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, stratify=labels)
    
    def save(split, images, labels):
        class_names = ["Pituitary", "Meningioma", "Glioma"]
        for i, (img, label) in enumerate(zip(images, labels)):
            label_dir = os.path.join(output_dir, split, class_names[label - 1])
            os.makedirs(label_dir, exist_ok=True)

            # Normalize and save image as PNG
            img = (img - np.min(img)) / (np.max(img) - np.min(img)) * 255
            img = Image.fromarray(np.uint8(img)).convert("L")
            img.save(os.path.join(label_dir, f"img_{i}.png"))

    save("train", X_train, y_train)
    save("test", X_test, y_test)