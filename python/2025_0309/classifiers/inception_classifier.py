import torch
import torch.nn as nn

class InceptionBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        
        self.branch1 = nn.Conv1d(in_channels, out_channels//4, 1)
        
        self.branch2 = nn.Sequential(
            nn.Conv1d(in_channels, out_channels//4, 1),
            nn.Conv1d(out_channels//4, out_channels//4, 3, padding=1)
        )
        
        self.branch3 = nn.Sequential(
            nn.Conv1d(in_channels, out_channels//4, 1),
            nn.Conv1d(out_channels//4, out_channels//4, 5, padding=2)
        )
        
        self.branch4 = nn.Sequential(
            nn.MaxPool1d(3, stride=1, padding=1),
            nn.Conv1d(in_channels, out_channels//4, 1)
        )
        
    def forward(self, x):
        branch1 = self.branch1(x)
        branch2 = self.branch2(x)
        branch3 = self.branch3(x)
        branch4 = self.branch4(x)
        return torch.cat([branch1, branch2, branch3, branch4], 1)

class InceptionClassifier(nn.Module):
    def __init__(self, input_dim=768):
        super().__init__()
        self.conv1 = nn.Conv1d(2, 64, 7, padding=3)
        self.bn1 = nn.BatchNorm1d(64)
        self.relu = nn.ReLU()
        
        self.inception1 = InceptionBlock(64, 256)
        self.inception2 = InceptionBlock(256, 512)
        
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(512, 2)
        
    def forward(self, x1, x2):
        x = torch.stack([x1, x2], dim=1)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        
        x = self.inception1(x)
        x = self.inception2(x)
        
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        return self.fc(x) 