import os
from typing import Sequence, Tuple

import numpy as np
import torch
from matplotlib import pyplot as plt
from data.image_loader import ImageLoader
from torch import nn
from torch.utils.data import DataLoader, Subset
import seaborn as sns


def generate_confusion_data(
    model: nn.Module,
    dataset: ImageLoader,
    use_cuda: bool = False,
) -> Tuple[Sequence[int], Sequence[int], Sequence[str]]:
    batch_size = 32
    cuda = use_cuda and torch.cuda.is_available()
    dataloader_args = {"num_workers": 1, "pin_memory": True} if cuda else {}
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, **dataloader_args)

    preds = np.zeros(len(dataset)).astype(np.int32)
    targets = np.zeros(len(dataset)).astype(np.int32)
    label_to_idx = dataset.get_classes()
    class_labels = [""] * len(label_to_idx)

    model.eval()

    preds = []
    targets = []
    with torch.no_grad():
        for inputs, labels in loader:
            if cuda:
                inputs, labels = inputs.cuda(), labels.cuda()
            outputs = model(inputs)
            predictions = torch.argmax(outputs, dim=1)
            preds.extend(predictions.tolist())
            targets.extend(labels.tolist())

    targets = torch.tensor(targets)
    preds = torch.tensor(preds)
    model.train()

    return targets.cpu().numpy(), preds.cpu().numpy(), class_labels


def generate_confusion_matrix(
    targets: np.ndarray, preds: np.ndarray, num_classes: int, normalize=True
) -> np.ndarray:
    confusion_matrix = np.zeros((num_classes, num_classes))

    for target, prediction in zip(targets, preds):
        confusion_matrix[target, prediction] += 1

    if normalize:
        row_sums = confusion_matrix.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1
        confusion_matrix = confusion_matrix / row_sums
    
    return confusion_matrix


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
    plt.savefig(os.path.join(path, f"{model_name}_confusion_matrix.png"))
    plt.show()

def generate_and_plot_confusion_matrix(
    model: nn.Module, dataset: ImageLoader, path: str, use_cuda: bool = False
) -> None:
    targets, predictions, class_labels = generate_confusion_data(
        model, dataset, use_cuda=use_cuda
    )

    confusion_matrix = generate_confusion_matrix(
        np.array(targets, dtype=np.int32),
        np.array(predictions, np.int32),
        len(class_labels),
    )

    plot_confusion_matrix(confusion_matrix, class_labels, model.__class__.__name__, path)


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


def generate_accuracy_data(
    model: nn.Module,
    dataset: ImageLoader,
    num_attributes: int,
    use_cuda: bool = False,
) -> Tuple[Sequence[int], Sequence[int], Sequence[str]]:
    batch_size = 32
    cuda = use_cuda and torch.cuda.is_available()
    dataloader_args = {"num_workers": 1, "pin_memory": True} if cuda else {}
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, **dataloader_args)

    preds = np.zeros((len(dataset), num_attributes)).astype(np.int32)
    targets = np.zeros((len(dataset), num_attributes)).astype(np.int32)

    model.eval()

    preds = []
    targets = []
    with torch.no_grad():
        for inputs, labels in loader:
            if cuda:
                inputs, labels = inputs.cuda(), labels.cuda()
            outputs = model(inputs)
            predictions = torch.round(torch.sigmoid(outputs))
            preds.extend(predictions.numpy())
            targets.extend(labels.numpy())
    
    preds = torch.tensor(preds)
    targets = torch.tensor(targets)
    model.train()

    return targets.cpu().numpy(), preds.cpu().numpy()


def generate_accuracy_table(
    targets: np.ndarray, preds: np.ndarray, num_attributes: int
) -> np.ndarray:
    accuracy_table = np.zeros(num_attributes)
    correct = np.sum(targets == preds, axis=0)
    total = targets.shape[0]
    accuracy_table = correct / total

    return accuracy_table


def plot_accuracy_table(accuracy_table: np.ndarray, attribute_labels: Sequence[str]) -> None:
    plt.figure(figsize=(10, 2))
    sns.heatmap(
        accuracy_table[np.newaxis, :],
        annot=True,
        fmt=".2f",
        cmap="Greens",
        xticklabels=attribute_labels,
        yticklabels=["Accuracy"],
        cbar=False,
        linewidths=0.3
    )

    plt.title("Attribute-wise Accuracy", fontsize=14)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.show()


def generate_and_plot_accuracy_table(
    model: nn.Module, 
    dataset: ImageLoader, 
    num_attributes = int, 
    attribute_labels = Sequence[str],
    use_cuda: bool = False
) -> None:
    targets, predictions = generate_accuracy_data(
        model, dataset, num_attributes, use_cuda=use_cuda
    )

    accuracy_table = generate_accuracy_table(
        np.array(targets, dtype=np.int32),
        np.array(predictions, np.int32),
        num_attributes
    )

    plot_accuracy_table(accuracy_table, attribute_labels)