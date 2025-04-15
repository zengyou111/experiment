import torch
import torch.nn as nn

class ViT(nn.Module):
    """
    Vision Transformer (ViT) adapted for sequence classification
    """
    def __init__(self, input_dim=768, num_classes=2, num_heads=8, 
                 num_layers=12, mlp_ratio=4, dropout=0.1):
        super().__init__()
        
        # Input embedding
        self.embedding = nn.Linear(input_dim, input_dim)
        self.pos_embedding = nn.Parameter(torch.randn(1, input_dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, input_dim))
        
        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=input_dim,
            nhead=num_heads,
            dim_feedforward=input_dim * mlp_ratio,
            dropout=dropout,
            activation='gelu',
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, 
            num_layers=num_layers
        )
        
        # MLP Head
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, input_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(input_dim // 2, num_classes)
        )
        
    def forward(self, x):
        # Add position embeddings and CLS token
        b = x.shape[0]
        cls_tokens = self.cls_token.expand(b, -1, -1)
        x = self.embedding(x)
        x = x + self.pos_embedding
        x = torch.cat((cls_tokens, x), dim=1)
        
        # Apply transformer
        x = self.transformer(x)
        
        # Use CLS token for classification
        x = x[:, 0]
        x = self.mlp_head(x)
        return x 