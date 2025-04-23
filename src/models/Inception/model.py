import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import inception_v3, Inception_V3_Weights

class MyInception(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        # Load the pretrained model with default weights
        self.base = inception_v3(weights=Inception_V3_Weights.DEFAULT, aux_logits=True) # aux_logits must be true but kinda in the way

        # Freeze all layers
        for param in self.base.parameters():
            param.requires_grad = False

        # Replace classifiers
        self.base.fc = nn.Linear(2048, num_classes)
        self.base.AuxLogits.fc = nn.Linear(768, num_classes)

        self.loss_criterion = nn.CrossEntropyLoss()

    def forward(self, x):
        # Make sure input is 3-channel and resized to (299, 299)
        if x.shape[1] == 1:  # Grayscale to RGB
            x = x.repeat(1, 3, 1, 1)
        x = F.interpolate(x, size=(299, 299), mode='bilinear', align_corners=False)

        # Inception is weird and can return either a tuple or a scalar
        if self.training and self.base.aux_logits:
            out, aux_out = self.base(x)
            return out, aux_out
        else:
            out = self.base(x)
            return out

    def compute_loss(self, output, target):
        if isinstance(output, tuple):
            main_out, aux_out = output
            loss_main = self.loss_fn(main_out, target)
            loss_aux = self.loss_fn(aux_out, target)
            return loss_main + 0.4 * loss_aux
        else:
            return self.loss_fn(output, target)
        
    def count_parameters(self):
        # Count parameters for the main network and auxiliary logits
        base_params = sum(p.numel() for p in self.base.parameters() if p.requires_grad)
        aux_params = sum(p.numel() for p in self.base.AuxLogits.parameters() if p.requires_grad)
        return base_params + aux_params
