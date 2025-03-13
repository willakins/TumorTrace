import torch
import torch.nn as nn
import torch.nn.functional as F

class CNN_3D(nn.Module):
    def __init__(self, num_classes=3): # Pituitary, Meningioma, and Glioma Tumor
        super(CNN_3D, self).__init__()

        self.layers = nn.Sequential(
            nn.Conv3d(1, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=2, stride=2),
            
            nn.Conv3d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=2, stride=2),
            
            nn.Conv3d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=2, stride=2)
        )
        
        # Fully connected layers
        self.fc1 = nn.Linear(64 * 1 * 1 * 1, 512)
        self.fc2 = nn.Linear(512, num_classes)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        
        print('pooling:', x.shape)
        x = x.view(x.size(0), -1)  # Flatten the tensor to (batch_size, flattened_features)
        
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        
        return x
