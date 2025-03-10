"""
This file should preprocess the raw data and place the final data into the preprocessed folder.
"""
import pickle
from torch.utils.data import Dataset, DataLoader, random_split
import torchvision.transforms as transforms
import torch
data_path = 'archive\\brain_tumor_mri\\new_dataset\\training_data.pickle'


# Data preloaded from pickle file
with open(data_path, 'rb') as file:
    loaded_data = pickle.load(file)

# Unpacking the data into the images and their corresponding labels
images, labels = zip(*loaded_data)

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

# Applying random transformations to vary data
transformations = transforms.Compose([
    transforms.RandomRotation(15),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip()
])

dataset = MRIDataset(images, labels, transformations, model_type=None)

# Splitting the dataset into training and testing
training_size = int(.8 * len(dataset))
testing_size = len(dataset) - training_size
training_dataset, testing_dataset = random_split(dataset, [training_size, testing_size])

# Two separate loaders for training and testing
train_loader = DataLoader(training_dataset, batch_size=16, shuffle=True)
testing_loader = DataLoader(testing_dataset, batch_size=16, shuffle=True)



for sample_image, sample_label in train_loader:
    print(f"Image shape: {sample_image.shape}")
    print(f"Label: {sample_label}")
    break