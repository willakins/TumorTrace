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
from src.models import (
    CNN_3D,
    MyResNet,
    MyInception,
)

def compute_accuracy(logits: torch.Tensor, labels: torch.Tensor) -> float:
    best_guesses = torch.argmax(logits, axis=1)
    diff = torch.where(best_guesses == labels, 1, 0)
    batch_accuracy = float(torch.sum(diff) / labels.shape[0])

    return batch_accuracy


def compute_loss(
    model: Union[CNN_3D, MyResNet, MyInception],
    model_output: torch.Tensor,
    target_labels: torch.Tensor,
    is_normalize: bool = True,
) -> torch.Tensor:
    loss = model.loss_criterion(model_output, target_labels)
    if (is_normalize):
        loss /= target_labels.shape[0]

    return loss

def save_trained_model_weights(
    model: Union[CNN_3D, MyResNet, MyInception], out_dir: str
) -> None:
    class_name = model.__class__.__name__
    state_dict = model.state_dict()

    assert class_name in set(
        ["CNN_3D", "MyResNet", "MyInception"]
    ), "Please save only supported models"

    save_dict = {"class_name": class_name, "state_dict": state_dict}
    path = f"{out_dir}/trained_{class_name}_final.pt"
    torch.save(save_dict, path)
    print(f"Saved training plots to {path}")

def compute_mean_and_std(dir_name: str) -> Tuple[float, float]:
    count = 0
    mean = 0.0
    M2 = 0.0  # sum of squares of differences from the current mean

    img_paths = glob.glob(os.path.join(dir_name, "**", "*.png"), recursive=True)
    if len(img_paths) == 0:
        raise ValueError(f"No images found in {dir_name}. Check your folder structure and file extensions.")

    for img_path in img_paths:
        img = Image.open(img_path).convert("L")
        img_array = np.array(img).astype(np.float32) / 255.0
        flat = img_array.flatten()

        for x in flat:
            count += 1
            delta = x - mean
            mean += delta / count
            delta2 = x - mean
            M2 += delta * delta2

    if count < 2:
        return mean, 0.0
    variance = M2 / (count - 1)
    std = np.sqrt(variance)
    return mean, std

def convert_pickle_to_folder(pickle_path, output_dir):
    if os.path.exists(output_dir):
        print(f"[INFO] Skipping conversion. Folder '{output_dir}' already exists.")
        return
    
    with open(pickle_path, 'rb') as f:
        data = pickle.load(f)  # List of (image, label) tuples

    images, labels = zip(*data)
    X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, stratify=labels)

    def save(split, images, labels):
        for i, (img, label) in enumerate(zip(images, labels)):
            label_dir = os.path.join(output_dir, split, f"class_{label}")
            os.makedirs(label_dir, exist_ok=True)

            # Normalize and save image as PNG
            img = (img - np.min(img)) / (np.max(img) - np.min(img)) * 255
            img = Image.fromarray(np.uint8(img)).convert("L")
            img.save(os.path.join(label_dir, f"img_{i}.png"))

    save("train", X_train, y_train)
    save("test", X_test, y_test)