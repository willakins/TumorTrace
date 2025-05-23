import os
from typing import Tuple, List, Dict
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms


class ImageLoader(Dataset):
    """Dataset class for loading MRI images from disk."""

    train_folder = "train"
    test_folder = "test"

    def __init__(
        self,
        root_dir: str,
        split: str = "train",
        transform: transforms.Compose = None,
        is_3d: bool = False,
        n_slices: int = 1,
    ):
        self.root = os.path.expanduser(root_dir)
        self.transform = transform
        self.split = split
        self.is_3d = is_3d
        self.n_slices = n_slices

        # Determine the folder based on split (train or test)
        if split == "train":
            self.curr_folder = os.path.join(root_dir, self.train_folder)
        elif split == "test":
            self.curr_folder = os.path.join(root_dir, self.test_folder)

        # Get class labels (directories in the root folder)
        self.class_dict = self.get_classes()
        # Load dataset based on class labels
        self.dataset = self.load_imagepaths_with_labels(self.class_dict)

    def get_classes(self) -> Dict[str, int]:
        """Get class names from folder names and create a mapping to labels."""
        class_names = sorted([
            d for d in os.listdir(self.curr_folder)
            if os.path.isdir(os.path.join(self.curr_folder, d))
        ])
        return {name: idx for idx, name in enumerate(class_names)}

    def load_imagepaths_with_labels(self, class_labels):
        """Collect image paths per class and group them into slices (for 3D data)."""
        paths = []
        for class_name, label in class_labels.items():
            class_folder = os.path.join(self.curr_folder, class_name)
            fpaths = sorted([
                os.path.join(class_folder, fn)
                for fn in os.listdir(class_folder)
                if fn.lower().endswith((".png", ".jpg"))
            ])
            # Group into volumes of n_slices
            for i in range(0, len(fpaths), self.n_slices):
                group = fpaths[i:i+self.n_slices]
                if len(group) == self.n_slices:
                    paths.append((group, label))
        return paths

    def load_img_from_path(self, path: str) -> Image:
        """Load image and convert to grayscale."""
        return Image.open(path).convert(mode='L')  # grayscale

    def __getitem__(self, index):
        """Get image and label at index."""
        if self.is_3d:
            # 3D data: get multiple slices for the same volume
            paths, label = self.dataset[index]
            imgs = [self.load_img_from_path(p) for p in paths]
            if self.transform:
                imgs = [self.transform(img) for img in imgs]
            else:
                imgs = [transforms.ToTensor()(img) for img in imgs]
            img_tensor = torch.stack(imgs, dim=0)  # Stack slices into [n_slices, H, W]
            return img_tensor, label
        else:
            # 2D data: single image per entry
            path, label = self.dataset[index]
            img = self.load_img_from_path(path[0])
            if self.transform:
                img = self.transform(img)
            return img, label

    def __len__(self) -> int:
        """Return the length of the dataset."""
        return len(self.dataset)
