import os
from typing import Sequence, Tuple

import numpy as np
import torch
from matplotlib import pyplot as plt
from torch import nn
from torch.utils.data import DataLoader, Subset
import seaborn as sns
from sklearn.metrics import confusion_matrix as sklearn_confusion_matrix
from data.image_loader import ImageLoader


def generate_confusion_data(
    model: nn.Module,
    dataset: ImageLoader,
    use_cuda: bool = False,
) -> Tuple[Sequence[int], Sequence[int], Sequence[str]]:
    batch_size = 32
    cuda = use_cuda and torch.cuda.is_available()
    dataloader_args = {"num_workers": 1, "pin_memory": True} if cuda else {}
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, **dataloader_args)

    preds = []
    targets = []

    class_dict = dataset.get_classes()  # Assuming this returns a dict like {"glioma": 0, ...}
    class_labels = [label for label, _ in sorted(class_dict.items(), key=lambda x: x[1])]

    model.eval()
    with torch.no_grad():
        for inputs, labels in loader:
            if cuda:
                inputs, labels = inputs.cuda(), labels.cuda()
            outputs = model(inputs)
            predictions = torch.argmax(outputs, dim=1)
            preds.extend(predictions.tolist())
            targets.extend(labels.tolist())
    model.train()

    return np.array(targets), np.array(preds), class_labels


def generate_confusion_matrix(
    targets: np.ndarray, preds: np.ndarray, num_classes: int, normalize=True
) -> np.ndarray:
    cm = sklearn_confusion_matrix(targets, preds, labels=list(range(num_classes)))
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1, keepdims=True)
        cm[np.isnan(cm)] = 0  # Avoid NaNs from division by zero
    return cm


def plot_confusion_matrix(
    confusion_matrix: np.ndarray, class_labels: Sequence[str], model_name: str, path: str
) -> None:
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        confusion_matrix,
        annot=True,
        fmt=".2f",
        cmap="Blues",
        xticklabels=class_labels,
        yticklabels=class_labels,
        square=True,
        linewidths=0.5,
        cbar_kws={"shrink": 0.8}
    )

    plt.title(f"{model_name} Confusion Matrix", fontsize=14)
    plt.xlabel("Predicted Label", fontsize=12)
    plt.ylabel("True Label", fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    os.makedirs(path, exist_ok=True)
    plt.savefig(os.path.join(path, f"{model_name}_confusion_matrix.png"))
    plt.show()

def get_pred_images_for_target(
    model: nn.Module,
    dataset: ImageLoader,
    predicted_class: int,
    target_class: int,
    use_cuda: bool = False,
) -> Sequence[str]:
    model.eval()
    dataset_list = dataset.dataset
    indices = []
    image_paths = []
    for i, (image_path, class_label) in enumerate(dataset_list):
        if class_label == target_class:
            indices.append(i)
            image_paths.append(image_path)
    subset = Subset(dataset, indices)
    dataloader_args = {"num_workers": 1, "pin_memory": True} if use_cuda else {}
    loader = DataLoader(subset, batch_size=32, shuffle=False, **dataloader_args)
    preds = []
    for i, (inp, _) in enumerate(loader):
        if use_cuda:
            inp = inp.cuda()
        logits = model(inp)
        p = torch.argmax(logits, dim=1)
        preds.append(p)
    predictions = torch.cat(preds, dim=0).cpu().tolist()
    valid_image_paths = [
        image_paths[i] for i, p in enumerate(predictions) if p == predicted_class
    ]
    model.train()
    return valid_image_paths


def generate_and_plot_confusion_matrix(
    model: nn.Module, dataset: ImageLoader, path: str, use_cuda: bool = False
) -> None:
    targets, predictions, class_labels = generate_confusion_data(
        model, dataset, use_cuda=use_cuda
    )
    cm = generate_confusion_matrix(targets, predictions, len(class_labels))
    plot_confusion_matrix(cm, class_labels, model.__class__.__name__, path)
