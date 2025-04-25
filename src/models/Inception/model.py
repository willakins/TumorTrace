import torch.nn as nn
import torchvision.models as models

class MyInception(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        self.inception = models.inception_v3(weights='DEFAULT', aux_logits=True)

        freeze = True
        for name, param in self.inception.named_parameters():
            if "fc" in name or "conv2d" in name:  # Only keep the fully connected layers trainable
                freeze = False
            param.requires_grad = not freeze

        num_features = self.inception.fc.in_features
        self.inception.fc = nn.Linear(num_features, 512) 
        self.inception.fc_out = nn.Linear(512, num_classes)

        self.dropout = nn.Dropout(0.1)
        self.loss_criterion = nn.CrossEntropyLoss(reduction='mean')

    def forward(self, x):
        if x.shape[1] == 1:  # Grayscale image with 1 channel
            x = x.repeat(1, 3, 1, 1)  # Convert to 3 channels
        x = self.inception(x)
        x = self.dropout(x)
        x = self.inception.fc_out(x)
        return x

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
