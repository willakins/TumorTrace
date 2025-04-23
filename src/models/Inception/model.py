import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import inception_v3

class MyInception(nn.Module):
    def __init__(self, num_classes):
        super(MyInception, self).__init__()

        # Load pretrained model
        pretrained_model = inception_v3(pretrained=True, aux_logits=False)
        
        # Freeze all layers for feature extraction
        for param in pretrained_model.parameters():
            param.requires_grad = False

        # Remove the final fully connected layer
        self.features = nn.Sequential(*list(pretrained_model.children())[:-1]) # [-1] excludes the original fc layer

        # TODO: Define a new classification head, add hidden layers or change sizes, etc.
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(2048, num_classes),
        )

        self.loss_criterion = nn.CrossEntropyLoss(reduction='mean')

    def forward(self, x):
        x = x.repeat(1, 3, 1, 1)  # as Inception accepts 3-channel color images and our images are grey-scale
        x = self.features(x)
        model_output = self.classifier(x)
        return model_output
    
    def count_parameters(self):
        return sum(p.numel() for p in self.features.parameters()) + sum(p.numel() for p in self.classifier.parameters())