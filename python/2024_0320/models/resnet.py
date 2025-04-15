import torch
import torch.nn as nn

class ResNet152D(nn.Module):
    """
    ResNet152-D with modifications:
    1. Deep Stem
    2. Average Downsampling
    3. Improved Bottleneck
    """
    def __init__(self, input_dim=768, num_classes=2):
        super().__init__()
        
        # Deep Stem
        self.stem = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.LayerNorm(input_dim),
            nn.GELU(),
            nn.Dropout(0.1)
        )
        
        # Residual Blocks
        self.layer1 = self._make_layer(input_dim, input_dim, 3)
        self.layer2 = self._make_layer(input_dim, input_dim*2, 8)
        self.layer3 = self._make_layer(input_dim*2, input_dim*4, 36)
        self.layer4 = self._make_layer(input_dim*4, input_dim*8, 3)
        
        # Classifier Head
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(input_dim*8, num_classes)
        
    def _make_layer(self, in_channels, out_channels, blocks):
        layers = []
        layers.append(Bottleneck(in_channels, out_channels))
        for _ in range(1, blocks):
            layers.append(Bottleneck(out_channels, out_channels))
        return nn.Sequential(*layers)
        
    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = self.fc(x)
        return x

class Bottleneck(nn.Module):
    expansion = 4
    
    def __init__(self, in_channels, out_channels):
        super().__init__()
        
        self.conv1 = nn.Linear(in_channels, out_channels)
        self.bn1 = nn.LayerNorm(out_channels)
        self.conv2 = nn.Linear(out_channels, out_channels)
        self.bn2 = nn.LayerNorm(out_channels)
        self.conv3 = nn.Linear(out_channels, out_channels * self.expansion)
        self.bn3 = nn.LayerNorm(out_channels * self.expansion)
        self.relu = nn.GELU()
        
        self.downsample = None
        if in_channels != out_channels * self.expansion:
            self.downsample = nn.Sequential(
                nn.Linear(in_channels, out_channels * self.expansion),
                nn.LayerNorm(out_channels * self.expansion)
            )
            
    def forward(self, x):
        identity = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        
        out = self.conv3(out)
        out = self.bn3(out)
        
        if self.downsample is not None:
            identity = self.downsample(x)
            
        out += identity
        out = self.relu(out)
        
        return out 