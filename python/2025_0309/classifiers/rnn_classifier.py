import torch
import torch.nn as nn

class RNNClassifier(nn.Module):
    def __init__(self, input_dim=768, hidden_dim=256):
        super().__init__()
        self.rnn = nn.GRU(input_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_dim*2, 2)
    
    def forward(self, x1, x2):
        # Stack embeddings as sequence
        x = torch.stack([x1, x2], dim=1)
        out, _ = self.rnn(x)
        return self.fc(out[:, -1, :]) 