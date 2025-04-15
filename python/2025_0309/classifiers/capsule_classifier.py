import torch
import torch.nn as nn
import torch.nn.functional as F

class CapsuleLayer(nn.Module):
    def __init__(self, num_capsules, num_route_nodes, in_channels, out_channels, kernel_size=1, stride=1, padding=0):
        super().__init__()
        self.num_route_nodes = num_route_nodes
        self.num_capsules = num_capsules
        self.route_weights = nn.Parameter(
            torch.randn(num_capsules, num_route_nodes, in_channels, out_channels)
        )
    
    def squash(self, tensor):
        squared_norm = (tensor ** 2).sum(dim=-1, keepdim=True)
        scale = squared_norm / (1 + squared_norm)
        return scale * tensor / torch.sqrt(squared_norm + 1e-8)
    
    def forward(self, x):
        priors = torch.matmul(x[None, :, :, None, :], self.route_weights[:, None, :, :, :])
        logits = torch.zeros_like(priors[:, :, :, 0])
        
        for _ in range(3):  # routing iterations
            attn = F.softmax(logits, dim=2)
            outputs = self.squash((attn[:, :, :, None] * priors).sum(dim=2))
            logits = logits + (outputs * priors).sum(dim=-1)
        
        return outputs.squeeze()

class CapsuleClassifier(nn.Module):
    def __init__(self, input_dim=768):
        super().__init__()
        self.conv1 = nn.Conv1d(2, 256, kernel_size=9)
        self.primary_capsules = CapsuleLayer(8, 32, 256, 16)
        self.digit_capsules = CapsuleLayer(2, 8, 16, 16)
        
    def forward(self, x1, x2):
        x = torch.stack([x1, x2], dim=1)
        x = F.relu(self.conv1(x))
        x = self.primary_capsules(x)
        x = self.digit_capsules(x)
        return x.norm(dim=-1) 