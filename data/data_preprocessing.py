"""
This file should preprocess the raw data and place the final data into the preprocessed folder.
"""
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from torchvision.transforms import InterpolationMode

# Values from Inception V3 standard calculations
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# Dataset class for MRI image scans
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

        label = torch.tensor(self.labels[idx], dtype=torch.long)

        # Depending on what model, resize the image
        if self.model_type == "resnet":
            resize = transforms.Resize((224, 224))
            image = resize(image)
        elif self.model_type == "inception":
            image_3d = self.images[idx]
            middle_slice_idx = image_3d.shape[0] // 2
            image_2d = image_3d[middle_slice_idx, :, :]

            # Convert to tensor and add channel dimension (grayscale)
            # Shape becomes (1, H, W)

            model_transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Lambda(lambda x: x.repeat(3, 1, 1)),
                transforms.Resize((299, 299), interpolation=InterpolationMode.BICUBIC, antialias=True),
                transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
            ])
            image = model_transform(image_2d)
  

        # Applies the randomized transformations to the images
        if self.transformation:
            image = self.transformation(image).to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        label -= 1

        return image, label
    
