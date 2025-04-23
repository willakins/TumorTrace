"""
Utilities to be used along with the deep model
"""
from typing import Union, Tuple

import torch
from models.ResNet.model import MyResNet
import glob
import os
import numpy as np
from PIL import Image


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

    for img_path in glob.glob(os.path.join(dir_name, "**/**/*.jpg"), recursive=True):
        img = Image.open(img_path).convert("L")
        img_array = np.array(img) / 255.0
        pixel_values.extend(img_array.flatten())

    pixel_values = np.array(pixel_values)

    mean = np.mean(pixel_values)
    std = np.std(pixel_values, ddof=1)
    return mean, std
