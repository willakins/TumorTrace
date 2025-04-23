import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import inception_v3, Inception_V3_Weights

class MyInception(nn.Module):
    def __init__(self, num_classes):
        super(MyInception, self).__init__()

        # Use the updated 'weights' argument
        weights = Inception_V3_Weights.DEFAULT
        pretrained_model = inception_v3(weights=weights, aux_logits=False)

        # Freeze all layers for feature extraction
        for param in pretrained_model.parameters():
            param.requires_grad = False

        # Remove the final fully connected layer
        self.features = nn.Sequential(*list(pretrained_model.children())[:-1])  # Exclude original fc layer

        # Define a new classification head
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(2048, num_classes),
        )

        self.loss_criterion = nn.CrossEntropyLoss(reduction='mean')

    def forward(self, x):
        if x.shape[1] == 1:  # Grayscale image with 1 channel (maybe unnecessary)
            x = x.repeat(1, 3, 1, 1)
        x = self.features(x)
        model_output = self.classifier(x)
        return model_output

    def count_parameters(self):
        return sum(p.numel() for p in self.features.parameters()) + sum(p.numel() for p in self.classifier.parameters())