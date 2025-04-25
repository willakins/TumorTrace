import torch.nn as nn
import torchvision.models as models

class MyInception(nn.Module):
    def __init__(self, num_classes):
        super().__init__()


        self.inception = models.inception_v3(weights=models.inception_v3(weights='DEFAULT'), aux_logits=True)
        # List of layers to unfreeze
        unfreeze_layers = [
            self.inception.Mixed_6a
        ]

        # Freeze all parameters in the inception model
        for param in self.inception.parameters():
            param.requires_grad = False 

        # Unfreeze specific layers
        for layer in unfreeze_layers:
            for param in layer.parameters():
                param.requires_grad = True

        
        
        self.inception.fc = nn.Sequential(
            nn.Linear(2048, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )

        for param in self.inception.fc.parameters():
            param.requires_grad = True

        if self.inception.aux_logits:
            for param in self.inception.AuxLogits.fc.parameters():
                param.requires_grad = True    
        self.dropout = nn.Dropout(0.125)
        
        self.loss_criterion = nn.CrossEntropyLoss(reduction='mean')

    def forward(self, x):
        if x.shape[1] == 1:  # Grayscale image with 1 channel (maybe unnecessary)
            x = x.repeat(1, 3, 1, 1)
        if self.training and self.inception.aux_logits:
            out, aux_out = self.inception(x)
            out = self.dropout(out)
            return out, aux_out
        else:
            out = self.inception(x)
            out = self.dropout(out)
            return out
        
    def count_parameters(self):
        # Count parameters for the main network and auxiliary logits
        base_params = sum(p.numel() for p in self.inception.parameters() if p.requires_grad)
        aux_params = sum(p.numel() for p in self.inception.fc.parameters() if p.requires_grad)
        return base_params + aux_params
