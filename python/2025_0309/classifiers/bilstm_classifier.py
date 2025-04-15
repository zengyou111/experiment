from .base_classifier import BaseClassifier
import torch
import torch.nn as nn

class BiLSTMClassifier(BaseClassifier):
    def __init__(self):
        super().__init__()
        
        self.lstm = nn.LSTM(
            input_size=self.input_dim,
            hidden_size=self.hidden_dim,
            num_layers=2,
            batch_first=True,
            bidirectional=True
        )
        
        self.fc = nn.Sequential(
            nn.Linear(self.hidden_dim * 2, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim)
        )
    
    def forward_once(self, x):
        # x shape: [batch_size, input_dim]
        x = x.unsqueeze(1)  # 添加序列维度 [batch_size, 1, input_dim]
        _, (h_n, _) = self.lstm(x)
        # 连接双向LSTM的最后一个隐藏状态
        x = torch.cat((h_n[-2,:,:], h_n[-1,:,:]), dim=1)
        x = self.fc(x)
        return x 