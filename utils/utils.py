"""
Utilities to be used along with the deep model
"""

import os
import glob
import torch
import numpy as np
import pickle
from PIL import Image
from typing import Union, Tuple
from sklearn.model_selection import train_test_split

from src.models.ResNet.model import MyResNet

def compute_accuracy(logits: torch.Tensor, labels: torch.Tensor) -> float:
    best_guesses = torch.argmax(logits, axis=1)
    diff = torch.where(best_guesses == labels, 1, 0)
    batch_accuracy = float(torch.sum(diff) / labels.shape[0])

    return batch_accuracy


def compute_loss(
    model: Union[MyResNet],
    model_output: torch.Tensor,
    target_labels: torch.Tensor,
    is_normalize: bool = True,
) -> torch.Tensor:
    loss = model.loss_criterion(model_output, target_labels)
    if (is_normalize):
        loss /= target_labels.shape[0]

    return loss

def save_trained_model_weights(
    model: Union[MyResNet], out_dir: str
) -> None:
    class_name = model.__class__.__name__
    state_dict = model.state_dict()

    assert class_name in set(
        ["MyResNet"]
    ), "Please save only supported models"

    save_dict = {"class_name": class_name, "state_dict": state_dict}
    torch.save(save_dict, f"{out_dir}/trained_{class_name}_final.pt")

def compute_mean_and_std(dir_name: str) -> Tuple[float, float]:
    pixel_values = []

    for img_path in glob.glob(os.path.join(dir_name, "**/**/*.png"), recursive=True):
        img = Image.open(img_path).convert("L")
        img_array = np.array(img) / 255.0
        pixel_values.extend(img_array.flatten())

    pixel_values = np.array(pixel_values)
    if len(pixel_values) == 0:
        raise ValueError(f"No images found in {dir_name}. Check your folder structure and file extensions.")

    mean = np.mean(pixel_values)
    std = np.std(pixel_values, ddof=1)

    return mean, std

def prepare_dataset(pickle_path: str, processed_dir: str) -> Tuple[float, float]:
    # Convert pickle to folder structure if not already done
    if not os.path.exists(processed_dir):
        print("[INFO] Converting pickle to folder structure...")
        convert_pickle_to_folder(pickle_path, processed_dir)
    else:
        print("[INFO] Processed data directory already exists. Skipping conversion.")

    # Cache path for mean and std
    mean_std_cache = os.path.join(processed_dir, "mean_std.npy")

    if os.path.exists(mean_std_cache):
        print("[INFO] Loading cached dataset mean and std...")
        dataset_mean, dataset_std = np.load(mean_std_cache)
    else:
        print("[INFO] Computing dataset mean and std from scratch...")
        dataset_mean, dataset_std = compute_mean_and_std(processed_dir)
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