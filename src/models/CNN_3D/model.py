import torch
import torch.nn as nn

class CNN_3D(nn.Module):
    def __init__(self, inp_size, num_classes=3):
        super().__init__()

        kernel_size_depth = min(inp_size[0], 3)  # ideally use kernel size depth 3, but if n_slices < 3, then need to reduce kernel size depth
        kernel_size = 5
        padding_size = (kernel_size - 1) // 2

        self.conv_layers = nn.Sequential( 
            nn.Conv3d(in_channels=1, out_channels=10, kernel_size=(kernel_size_depth, kernel_size, kernel_size), padding=(0, padding_size, padding_size)),
            nn.BatchNorm3d(10),
            nn.MaxPool3d(kernel_size=(kernel_size_depth, kernel_size, kernel_size), padding=(0, padding_size, padding_size)),
            nn.ReLU(),
            nn.Conv3d(in_channels=10, out_channels=15, kernel_size=(kernel_size_depth, kernel_size, kernel_size), padding=(0, padding_size, padding_size)),
            nn.BatchNorm3d(15),
            nn.ReLU(),
            nn.Conv3d(in_channels=15, out_channels=20, kernel_size=(kernel_size_depth, kernel_size, kernel_size), padding=(0, padding_size, padding_size)),
            nn.BatchNorm3d(20),
            nn.ReLU(),
            nn.Dropout(),
            nn.Conv3d(in_channels=20, out_channels=25, kernel_size=(kernel_size_depth, kernel_size, kernel_size), padding=(0, padding_size, padding_size)),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(kernel_size_depth, kernel_size, kernel_size), padding=(0, padding_size, padding_size))
        )

        # Dynamically calculate the fully connected layer node size based on the output of Conv3D layers
        dummy_input = torch.zeros(1, 1, *inp_size)
        conv_output = self.conv_layers(dummy_input)
        flattened_size = conv_output.view(1, -1).shape[1] 

        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flattened_size, 100),
            nn.ReLU(),
            nn.Linear(100, num_classes)
        )

        self.loss_criterion = nn.CrossEntropyLoss(reduction='mean')

    def forward(self, x):
        x = self.conv_layers(x)
        model_output = self.fc_layers(x)
        return model_output
    
    def count_parameters(self):
        return sum(p.numel() for p in self.conv_layers.parameters()) + sum(p.numel() for p in self.fc_layers.parameters())