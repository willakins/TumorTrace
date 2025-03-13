"""
This file should preprocess the raw data and place the final data into the preprocessed folder.
"""
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms

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
        # from HxWxC to CxHxW
        image = torch.tensor(self.images[idx]).permute(2, 0, 1)

        label = torch.tensor(self.labels[idx], dtype=torch.float32)

        # Depending on what model, resize the image
        if (self.model_type == "resnet"):
            resize = transforms.Resize((224, 224))
            image = resize(image)
        elif (self.model_type == "inception"):
            resize = transforms.Resize((299, 299))
            image = resize(image)

        # Applies the randomized transformations to the images
        if self.transformation:
            image = self.transformation(image)
            
        return image, label