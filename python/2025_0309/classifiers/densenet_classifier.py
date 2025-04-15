import torch
import torch.nn as nn

class DenseBlock(nn.Module):
    def __init__(self, in_channels, growth_rate, num_layers):
        super().__init__()
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            self.layers.append(nn.Sequential(
                nn.BatchNorm1d(in_channels + i * growth_rate),
                nn.ReLU(),
                nn.Conv1d(in_channels + i * growth_rate, growth_rate, 3, padding=1)
            ))
    
    def forward(self, x):
        features = [x]
        for layer in self.layers:
            new_features = layer(torch.cat(features, 1))
            features.append(new_features)
        return torch.cat(features, 1)

class DenseNetClassifier(nn.Module):
    def __init__(self, input_dim=768):
        super().__init__()
        self.conv1 = nn.Conv1d(2, 64, 7, padding=3)
        self.bn1 = nn.BatchNorm1d(64)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool1d(3, stride=2, padding=1)
        
        self.dense1 = DenseBlock(64, 32, 4)
        self.trans1 = nn.Conv1d(64 + 4 * 32, 128, 1)
        self.dense2 = DenseBlock(128, 32, 4)
        
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(128 + 4 * 32, 2)
        
    def forward(self, x1, x2):
        x = torch.stack([x1, x2], dim=1)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        x = self.dense1(x)
        x = self.trans1(x)
        x = self.dense2(x)
        
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return self.fc(x) 