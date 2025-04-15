import torch
from torch.utils.data import Dataset
import numpy as np
from sklearn.model_selection import train_test_split
import os
import random

class CodePairDataset(Dataset):
    def __init__(self, samples):
        self.samples = samples
        
    def __len__(self):
        return len(self.samples)
        
    def __getitem__(self, idx):
        sample = self.samples[idx]
        x1 = sample['code_embedding']
        
        # 随机选择一个正样本
        pos_idx = random.randint(0, len(sample['positive_embeddings']) - 1)
        x2 = sample['positive_embeddings'][pos_idx].squeeze()
        
        # 随机选择一个负样本
        neg_idx = random.randint(0, len(sample['negative_embeddings']) - 1)
        x3 = sample['negative_embeddings'][neg_idx].squeeze()
        
        return x1, x2, x3

def prepare_data(samples, train_size=0.6, val_size=0.2, random_state=42):
    """Split data into train, validation and test sets"""
    train_samples, temp_samples = train_test_split(
        samples, train_size=train_size, random_state=random_state
    )
    val_samples, test_samples = train_test_split(
        temp_samples, test_size=0.5, random_state=random_state
    )
    
    return (
        CodePairDataset(train_samples),
        CodePairDataset(val_samples),
        CodePairDataset(test_samples)
    )

def compute_similarity_matrix(embeddings1, embeddings2):
    """Compute cosine similarity matrix between two sets of embeddings"""
    norm1 = torch.norm(embeddings1, dim=1, keepdim=True)
    norm2 = torch.norm(embeddings2, dim=1, keepdim=True)
    return torch.mm(embeddings1, embeddings2.t()) / (norm1 * norm2.t())

def set_seed(seed):
    """Set random seed for reproducibility"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def save_model(model, path, name):
    """Save model checkpoint"""
    os.makedirs(path, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(path, f'{name}.pth'))

def load_model(model, path, name):
    """Load model checkpoint"""
    model.load_state_dict(torch.load(os.path.join(path, f'{name}.pth')))
    return model

def encode_samples(samples, tokenizer, model, device, max_length=512):
    """编码样本数据"""
    encoded_samples = []
    for sample in samples:
        # 编码原始代码
        code_inputs = tokenizer(sample['code'], return_tensors='pt', 
                              padding='max_length', truncation=True,
                              max_length=max_length).to(device)
        with torch.no_grad():
            code_embedding = model(**code_inputs).last_hidden_state.mean(dim=1).cpu()
        
        # 编码正样本
        positive_embeddings = []
        for positive_sample in sample['positive']['code_positive']:
            pos_inputs = tokenizer(positive_sample['code'], return_tensors='pt',
                                 padding='max_length', truncation=True,
                                 max_length=max_length).to(device)
            with torch.no_grad():
                pos_embedding = model(**pos_inputs).last_hidden_state.mean(dim=1).cpu()
            positive_embeddings.append(pos_embedding.squeeze())  # 确保维度正确
        
        # 编码负样本
        negative_embeddings = []
        for negative_sample in sample['negative']['code_negative']:
            neg_inputs = tokenizer(negative_sample['code'], return_tensors='pt',
                                 padding='max_length', truncation=True,
                                 max_length=max_length).to(device)
            with torch.no_grad():
                neg_embedding = model(**neg_inputs).last_hidden_state.mean(dim=1).cpu()
            negative_embeddings.append(neg_embedding.squeeze())  # 确保维度正确
        
        encoded_samples.append({
            'code_embedding': code_embedding.squeeze(),  # 确保维度正确
            'positive_embeddings': positive_embeddings,
            'negative_embeddings': negative_embeddings,
            'language': sample['language']
        })
    
    return encoded_samples 