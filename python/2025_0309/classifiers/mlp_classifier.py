import torch
import torch.nn as nn

class MLPClassifier(nn.Module):
    def __init__(self, input_dim=768*2):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 2)
        )
    
    def forward(self, x1, x2):
        # Concatenate the two code embeddings
        x = torch.cat((x1, x2), dim=1)
        return self.network(x) 