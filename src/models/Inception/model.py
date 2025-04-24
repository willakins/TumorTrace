import torch.nn as nn
import torchvision.models as models

class Inception(nn.Module):
    def __init__(self, num_classes):
        super(Inception, self).__init__()


        self.inception = models.inception_v3(weights='DEFAULT', aux_logits=True)
        num_features = self.inception.fc.in_features
        self.inception.fc = nn.Linear(num_features, num_classes)
        self.dropout = nn.Dropout(0.1)
        
        

    def forward(self, x):
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
