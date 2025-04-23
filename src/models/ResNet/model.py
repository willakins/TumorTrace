import torch.nn as nn
from torchvision.models import resnet18

class MyResNet(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        pretrained_model = resnet18(pretrained=True)
        for param in pretrained_model.parameters():
            param.requires_grad = False

        self.conv_layers = nn.Sequential(*list(pretrained_model.children())[:-1])

        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512, num_classes),
        )

        self.loss_criterion = nn.CrossEntropyLoss(reduction='mean')

    def forward(self, x):
        x = x.repeat(1, 3, 1, 1)  # as ResNet accepts 3-channel color images

        x = self.conv_layers(x)
        model_output = self.fc_layers(x)

        return model_output
    
    def count_parameters(self):
        return sum(p.numel() for p in self.conv_layers.parameters()) + sum(p.numel() for p in self.fc_layers.parameters())