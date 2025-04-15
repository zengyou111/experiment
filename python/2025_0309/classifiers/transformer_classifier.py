import torch
import torch.nn as nn

class TransformerClassifier(nn.Module):
    def __init__(self, input_dim=768):
        super().__init__()
        self.transformer_encoder = nn.TransformerEncoderLayer(
            d_model=input_dim,
            nhead=8,
            dim_feedforward=2048
        )
        self.fc = nn.Linear(input_dim, 2)
    
    def forward(self, x1, x2):
        x = torch.stack([x1, x2], dim=1)
        out = self.transformer_encoder(x)
        return self.fc(out.mean(dim=1)) 