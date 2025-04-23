"""
This file should preprocess the raw data and place the final data into the preprocessed folder.
"""
import os
from typing import Dict, List, Tuple

import torch
import torchvision
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms


class MRIDataset(Dataset):
    def __init__(self, images, labels, transformation, model_type=None):
        self.images = images
        self.labels = labels
        self.transformation = transformation
        self.model_type = model_type

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        # Assuming images[idx] is a 3D numpy array of shape (D, H, W)
        image = torch.tensor(self.images[idx], dtype=torch.float32).unsqueeze(0)

        label = torch.tensor(self.labels[idx], dtype=torch.float32)

        # Depending on what model, resize the image
        if self.model_type == "resnet":
            resize = transforms.Resize((224, 224))
            image = resize(image)
        elif self.model_type == "inception":
            resize = transforms.Resize((299, 299))
            image = resize(image)

        # Applies the randomized transformations to the images
        if self.transformation:
            image = self.transformation(image)

        return image, label

# Dataset class for MRI image scans
class ImageLoader(Dataset):
    """Class for data loading"""

    train_folder = "train"
    test_folder = "test"

    def __init__(
        self,
        root_dir: str,
        split: str = "train",
        transform: torchvision.transforms.Compose = None,
    ):
        self.root = os.path.expanduser(root_dir)
        self.transform = transform
        self.split = split
        if split == "train":
            self.curr_folder = os.path.join(root_dir, self.train_folder)
        elif split == "test":
            self.curr_folder = os.path.join(root_dir, self.test_folder)
        self.class_dict = self.get_classes()
        self.dataset = self.load_imagepaths_with_labels(self.class_dict)

    def load_imagepaths_with_labels(
        self, class_labels: Dict[str, int]
    ) -> List[Tuple[str, int]]:

        img_paths = []  # a list of (filename, class index)
        for class_name, label in class_labels.items():
            class_folder = os.path.join(self.curr_folder, class_name)
            for file in os.listdir(class_folder):
                if file.endswith(".jpg"):
                    img_paths.append((os.path.join(class_folder, file), label))

        return img_paths

    def get_classes(self) -> Dict[str, int]:
        classes = dict()
        names = []
        for d in os.listdir(self.curr_folder):
            if os.path.isdir(os.path.join(self.curr_folder, d)):
                names.append(d)
                
        names.sort()
        classes = {name: idx for idx, name in enumerate(names)}

        return classes

    def load_img_from_path(self, path: str) -> Image:
        return Image.open(path).convert(mode='L')

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int]:
        path, class_idx = self.dataset[index]
        img = self.load_img_from_path(path)
        if self.transform:
            img = self.transform(img)

        return img, class_idx

    def __len__(self) -> int:
        return len(self.dataset)
