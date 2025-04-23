import os
from typing import Tuple, List, Dict
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

    def get_classes(self) -> Dict[str, int]:
        class_names = sorted([
            d for d in os.listdir(self.curr_folder)
            if os.path.isdir(os.path.join(self.curr_folder, d))
        ])
        return {name: idx for idx, name in enumerate(class_names)}

    def load_imagepaths_with_labels(self, class_labels: Dict[str, int]) -> List[Tuple[str, int]]:
        paths = []
        for class_name, label in class_labels.items():
            class_folder = os.path.join(self.curr_folder, class_name)
            for fname in os.listdir(class_folder):
                if fname.endswith(".png") or fname.endswith(".jpg"):
                    paths.append((os.path.join(class_folder, fname), label))
        return paths

    def load_img_from_path(self, path: str) -> Image:
        return Image.open(path).convert(mode='L')  # grayscale

    def __getitem__(self, index: int) -> Tuple:
        path, label = self.dataset[index]
        img = self.load_img_from_path(path)
        if self.transform:
            img = self.transform(img)
        return img, label

    def __len__(self) -> int:
        return len(self.dataset)