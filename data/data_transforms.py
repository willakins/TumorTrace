"""
Contains functions with different data transforms
"""

from typing import Sequence, Tuple
import torchvision.transforms as transforms


def get_fundamental_transforms(inp_size: Tuple[int, int]) -> transforms.Compose:
    fundamental_transforms = transforms.Compose([
        transforms.Resize(inp_size),
        transforms.ToTensor(),
    ])

    return fundamental_transforms


def get_fundamental_augmentation_transforms(
    inp_size: Tuple[int, int]
) -> transforms.Compose:
    fund_aug_transforms = transforms.Compose([
        transforms.Resize(inp_size),
        transforms.ToTensor(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
    ])

    return fund_aug_transforms


def get_fundamental_normalization_transforms(
    inp_size: Tuple[int, int], pixel_mean: Sequence[float], pixel_std: Sequence[float]
) -> transforms.Compose:
    fund_norm_transforms = transforms.Compose([
        transforms.Resize(inp_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=pixel_mean, std=pixel_std),
    ])
    return fund_norm_transforms


def get_all_transforms(
    inp_size: Tuple[int, int], pixel_mean: Sequence[float], pixel_std: Sequence[float]
) -> transforms.Compose:
    all_transforms = transforms.Compose([
        transforms.Resize(inp_size),
        transforms.ToTensor(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.Normalize(mean=pixel_mean, std=pixel_std),
    ])
    return all_transforms

def get_3d_transforms(inp_size: Tuple[int, int], mean: float, std: float) -> transforms.Compose:
    return transforms.Compose([
        transforms.Pad(padding=int(0.1 * inp_size[0])),
        transforms.RandomCrop(inp_size),
        transforms.RandomResizedCrop(inp_size, scale=(0.6, 1.0), ratio=(0.8, 1.2)),
        transforms.RandomRotation(degrees=25),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.0, hue=0.0),
        transforms.ToTensor(),
        transforms.Normalize(mean=[mean], std=[std]),
    ])
