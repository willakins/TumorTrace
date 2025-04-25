import torch
import torch.nn as nn

class BasicBlock3D(nn.Module):
    """
    A basic 3D residual block with two 3x3x3 convolutions and a skip connection.
    """
    expansion = 1

    def __init__(self, in_channels, out_channels, depth_kernel, stride=(1,1,1), downsample=None):
        super().__init__()
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=(depth_kernel,3,3),
                               stride=stride, padding=(depth_kernel//2,1,1), bias=False)
        self.bn1 = nn.BatchNorm3d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=(depth_kernel,3,3),
                               stride=1, padding=(depth_kernel//2,1,1), bias=False)
        self.bn2 = nn.BatchNorm3d(out_channels)
        self.downsample = downsample

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

class CNN_3D(nn.Module):
    def __init__(self, inp_size, num_classes=3):
        super().__init__()
        depth, height, width = inp_size
        depth_kernel = min(depth, 3)

        # Initial convolution
        self.conv1 = nn.Conv3d(1, 32, kernel_size=(depth_kernel,3,3), padding=(depth_kernel//2,1,1), bias=False)
        self.bn1 = nn.BatchNorm3d(32)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(kernel_size=(1,2,2), stride=(1,2,2))

        # Residual layers
        self.layer1 = self._make_layer(32, 64, depth_kernel, blocks=2)
        self.layer2 = self._make_layer(64, 128, depth_kernel, blocks=2)
        self.layer3 = self._make_layer(128, 256, depth_kernel, blocks=2)

        # Adaptive pooling & classifier
        self.avgpool = nn.AdaptiveAvgPool3d((1,1,1))
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(256 * BasicBlock3D.expansion, num_classes)

        self.loss_criterion = nn.CrossEntropyLoss()

        self._initialize_weights()

    def _make_layer(self, in_channels, out_channels, depth_kernel, blocks, stride=(1,1,1)):
        downsample = None
        if in_channels != out_channels:
            downsample = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm3d(out_channels)
            )

        layers = []
        layers.append(BasicBlock3D(in_channels, out_channels, depth_kernel, stride=stride, downsample=downsample))
        for _ in range(1, blocks):
            layers.append(BasicBlock3D(out_channels, out_channels, depth_kernel))

        layers.append(nn.MaxPool3d(kernel_size=(1,2,2), stride=(1,2,2)))
        return nn.Sequential(*layers)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, mode='fan_in', nonlinearity='linear')
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.fc(x)

        return x

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
