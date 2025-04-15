import torch
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def calculate_metrics(outputs, targets):
    """计算各种评估指标"""
    predictions = torch.argmax(outputs, dim=1).cpu().numpy()
    targets = targets.cpu().numpy()
    
    return {
        'accuracy': accuracy_score(targets, predictions),
        'precision': precision_score(targets, predictions, average='weighted'),
        'recall': recall_score(targets, predictions, average='weighted'),
        'f1': f1_score(targets, predictions, average='weighted')
    }

def calculate_similarity_matrix(embeddings_1, embeddings_2):
    """计算余弦相似度矩阵"""
    normalized_1 = F.normalize(embeddings_1, p=2, dim=1)
    normalized_2 = F.normalize(embeddings_2, p=2, dim=1)
    return torch.mm(normalized_1, normalized_2.t()) 