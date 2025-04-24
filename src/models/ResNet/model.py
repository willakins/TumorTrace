import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights

class MyResNet(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        # Use the new 'weights' argument
        pretrained_model = resnet18(weights=ResNet18_Weights.DEFAULT)
        for param in pretrained_model.parameters():
            param.requires_grad = False

        self.conv_layers = nn.Sequential(*list(pretrained_model.children())[:-1])

        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512, 100),
            nn.ReLU(),
            nn.Linear(100, num_classes)
        )

        self.loss_criterion = nn.CrossEntropyLoss(reduction='mean')

    def forward(self, x):
        if x.shape[1] == 1:  # Grayscale image with 1 channel (maybe unnecessary)
            x = x.repeat(1, 3, 1, 1)
        x = self.conv_layers(x)
        model_output = self.fc_layers(x)
        return model_output

    def count_parameters(self):
        return sum(p.numel() for p in self.conv_layers.parameters()) + sum(p.numel() for p in self.fc_layers.parameters())
