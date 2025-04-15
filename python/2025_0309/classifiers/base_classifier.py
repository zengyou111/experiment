import torch
import torch.nn as nn
from config import Config

class BaseClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.input_dim = Config.INPUT_DIM
        self.hidden_dim = Config.HIDDEN_DIM
        
    def forward_once(self, x):
        """单个输入的前向传播，由子类实现"""
        raise NotImplementedError
        
    def forward(self, x1, x2):
        """处理一对输入"""
        # 获取特征
        feat1 = self.forward_once(x1)
        feat2 = self.forward_once(x2)
        
        # 计算相似度
        sim = torch.cosine_similarity(feat1, feat2, dim=1)
        output = torch.stack([1 - sim, sim], dim=1)  # [batch_size, 2]
        
        return output, feat1  # 返回输出和特征 