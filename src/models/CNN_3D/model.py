import torch
import torch.nn as nn
import torch.nn.functional as F

class CNN_3D(nn.Module):
    def __init__(self, inp_size, num_classes=3): # Pituitary, Meningioma, and Glioma Tumor
        super(CNN_3D, self).__init__()

        self.conv_layers = nn.Sequential( # Not set up for 3D, just filler for testing
            nn.Conv2d(in_channels=1, out_channels=10, kernel_size=5, padding=2),
            nn.BatchNorm2d(10),
            nn.MaxPool2d(kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=10, out_channels=15, kernel_size=5, padding=2),
            nn.BatchNorm2d(15),
            nn.ReLU(),
            nn.Conv2d(in_channels=15, out_channels=20, kernel_size=5, padding=2),
            nn.BatchNorm2d(20),
            nn.ReLU(),
            nn.Dropout(),
            nn.Conv2d(in_channels=20, out_channels=25, kernel_size=5, padding=0),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3)
        )

        # Dynamically calculate fully connected layer node size
        dummy_input = torch.zeros(1, 1, *inp_size)
        conv_output = self.conv_layers(dummy_input)
        flattened_size = conv_output.view(1, -1).shape[1]

        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flattened_size, 100),
            nn.ReLU(),
            nn.Linear(100, num_classes)
        )

        self.loss_criterion = nn.CrossEntropyLoss(reduction='mean')

    def forward(self, x):
        x = self.conv_layers(x)
        model_output = self.fc_layers(x)
        
        return model_output
    
    def count_parameters(self):
        return sum(p.numel() for p in self.conv_layers.parameters()) + sum(p.numel() for p in self.fc_layers.parameters())