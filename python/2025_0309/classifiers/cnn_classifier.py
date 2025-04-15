from .base_classifier import BaseClassifier
import torch.nn as nn

class CNNClassifier(BaseClassifier):
    def __init__(self):
        super().__init__()
        
        self.cnn = nn.Sequential(
            nn.Linear(self.input_dim, self.hidden_dim),  # 先降维
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
        )
        
        self.fc = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim)
        )
    
    def forward_once(self, x):
        # x shape: [batch_size, input_dim]
        x = self.cnn(x)
        x = self.fc(x)
        return x 