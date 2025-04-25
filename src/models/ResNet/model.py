import torch.nn as nn
import torchvision.models as models
import random

class MyResNet(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        self.resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        unfreeze_layers = [
            self.resnet.layer4
        ]

        # Freeze all parameters in the resnet model
        for param in self.resnet.parameters():
            param.requires_grad = False

        # Unfreeze specific layers
        for layer in unfreeze_layers:
            for param in layer.parameters():
                if random.random() < 0.1: # Reduces params from 8 mil to 2 mil probably (pun intended)
                    param.requires_grad = True

        self.resnet.fc = nn.Sequential(
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )

        for param in self.resnet.fc.parameters():
            param.requires_grad = True

        self.loss_criterion = nn.CrossEntropyLoss(reduction='mean')

    def forward(self, x):
        if x.shape[1] == 1:  # Grayscale image with 1 channel (maybe unnecessary)
            x = x.repeat(1, 3, 1, 1)
        out = self.resnet(x)
        return out

    def count_parameters(self):
        # Count parameters for the entire model
        total_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return total_params
