import torch.nn as nn
import torchvision.models as models

class Inception(nn.Module):
    def __init__(self, num_classes=3): # Pituitary, Meningioma, and Glioma Tumor
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

      