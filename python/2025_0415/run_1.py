import os
import json
import torch
import torch.nn.functional as F
import random
import time
from transformers import AutoTokenizer, AutoModel
from torch.utils.data import DataLoader, Dataset
from torch import nn, optim
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score, roc_curve, auc, roc_auc_score
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import numpy as np
from tqdm import tqdm

# 设置设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 参数配置
projectionHeadEpoch = 50
classifyEpoch=70
Temperature = 0.2
projectionHeadPatience = 2
Trainsize = 100
lr1 = 3e-6
lr2 = 1e-4  # 分类器学习率
ProjectionBatchSize = 1
BatchSize = 64
modelPath = '../../../model/unixcoder/'
langPath = '../../../Datasets/code_pairs_java_python.jsonl'

# 加载模型和 tokenizer
tokenizer = AutoTokenizer.from_pretrained(modelPath)
model = AutoModel.from_pretrained(modelPath).to(device)


# 加载样本数据
def load_samples_from_jsonl(file_path, max_samples=Trainsize):
    samples = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            record = json.loads(line.strip())
            samples.append(record)
            if len(samples) >= max_samples:
                break
    return samples


samples = load_samples_from_jsonl(langPath)


# 编码样本
def encode_sample(sample, tokenizer, model, device, max_length=512):
    code_inputs = tokenizer(sample['code'], return_tensors='pt', padding='max_length', truncation=True,
                            max_length=max_length).to(device)
    with torch.no_grad():
        code_embedding = model(**code_inputs).last_hidden_state.mean(dim=1).cpu()
    positive_embeddings = []
    for positive_sample in sample['positive']['code_positive']:
        pos_inputs = tokenizer(positive_sample['code'], return_tensors='pt', padding='max_length', truncation=True,
                               max_length=max_length).to(device)
        with torch.no_grad():
            pos_embedding = model(**pos_inputs).last_hidden_state.mean(dim=1).cpu()
        positive_embeddings.append(pos_embedding)
    negative_embeddings = []
    for negative_sample in sample['negative']['code_negative']:
        neg_inputs = tokenizer(negative_sample['code'], return_tensors='pt', padding='max_length', truncation=True,
                               max_length=max_length).to(device)
        with torch.no_grad():
            neg_embedding = model(**neg_inputs).last_hidden_state.mean(dim=1).cpu()
        negative_embeddings.append(neg_embedding)
    return {
        'code_embedding': code_embedding,
        'positive_embeddings': positive_embeddings,
        'negative_embeddings': negative_embeddings
    }


encoded_samples = [encode_sample(sample, tokenizer, model, device) for sample in samples]


# 数据集类
class CodeCloneDataset(Dataset):
    def __init__(self, samples):
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        anchor = sample['code_embedding'].view(1, -1)
        positives = [pos.view(1, -1) for pos in sample['positive_embeddings']]
        negatives = [neg.view(1, -1) for neg in sample['negative_embeddings']]
        return anchor, positives, negatives


# 分割数据集
from sklearn.model_selection import train_test_split

train_samples, remaining_samples = train_test_split(encoded_samples, test_size=0.4, random_state=42)
val_samples, test_samples = train_test_split(remaining_samples, test_size=0.5, random_state=42)


# SimCLR 对比损失函数
def simclr_contrastive_loss(anchor, positives, negatives, temperature=Temperature):
    anchor = F.normalize(anchor, dim=1)
    positives = [F.normalize(pos, dim=1) for pos in positives]
    negatives = [F.normalize(neg, dim=1) for neg in negatives]
    positive_loss = 0
    for pos in positives:
        pos_similarity = torch.exp(torch.mm(anchor, pos.t()) / temperature)
        neg_similarity = sum(torch.exp(torch.mm(anchor, neg.t()) / temperature) for neg in negatives)
        positive_loss += -torch.log(pos_similarity / (pos_similarity + neg_similarity)).mean()
    return positive_loss / len(positives)


# 原始 ConbaLayer
class ConbaLayer(nn.Module):
    def __init__(self, input_dim):
        super(ConbaLayer, self).__init__()
        self.selective_fc = nn.Linear(input_dim, input_dim)
        self.swish = nn.SiLU()
        self.control_weight = nn.Parameter(torch.ones(input_dim))
        self.feedback_weight = nn.Parameter(torch.ones(input_dim))
        self.A = nn.Linear(input_dim, input_dim, bias=False)
        self.B = nn.Linear(input_dim, input_dim, bias=False)
        self.C = nn.Linear(input_dim, input_dim, bias=False)
        self.fc2 = nn.Linear(input_dim, input_dim)

    def forward(self, x, previous_state=None):
        if x.dim() > 2:
            x = x.view(x.size(0), -1)
        if previous_state is None:
            previous_state = torch.zeros_like(x, device=x.device)
        state = self.A(previous_state) + self.B(x)
        selective_output = self.swish(self.selective_fc(x)) * x
        adjusted_output = selective_output * self.control_weight + state * self.feedback_weight
        output = self.C(adjusted_output)
        output = self.fc2(output)
        return output, state


# 消融实验变体
class ConbaNoControl(nn.Module):
    def __init__(self, input_dim):
        super(ConbaNoControl, self).__init__()
        self.selective_fc = nn.Linear(input_dim, input_dim)
        self.swish = nn.SiLU()
        self.feedback_weight = nn.Parameter(torch.ones(input_dim))
        self.A = nn.Linear(input_dim, input_dim, bias=False)
        self.B = nn.Linear(input_dim, input_dim, bias=False)
        self.C = nn.Linear(input_dim, input_dim, bias=False)
        self.fc2 = nn.Linear(input_dim, input_dim)

    def forward(self, x, previous_state=None):
        if x.dim() > 2:
            x = x.view(x.size(0), -1)
        if previous_state is None:
            previous_state = torch.zeros_like(x, device=x.device)
        state = self.A(previous_state) + self.B(x)
        selective_output = self.swish(self.selective_fc(x)) * x
        adjusted_output = selective_output + state * self.feedback_weight
        output = self.C(adjusted_output)
        output = self.fc2(output)
        return output, state


class ConbaNoFeedback(nn.Module):
    def __init__(self, input_dim):
        super(ConbaNoFeedback, self).__init__()
        self.selective_fc = nn.Linear(input_dim, input_dim)
        self.swish = nn.SiLU()
        self.control_weight = nn.Parameter(torch.ones(input_dim))
        self.A = nn.Linear(input_dim, input_dim, bias=False)
        self.B = nn.Linear(input_dim, input_dim, bias=False)
        self.C = nn.Linear(input_dim, input_dim, bias=False)
        self.fc2 = nn.Linear(input_dim, input_dim)

    def forward(self, x, previous_state=None):
        if x.dim() > 2:
            x = x.view(x.size(0), -1)
        if previous_state is None:
            previous_state = torch.zeros_like(x, device=x.device)
        state = self.A(previous_state) + self.B(x)
        selective_output = self.swish(self.selective_fc(x)) * x
        adjusted_output = selective_output * self.control_weight + state
        output = self.C(adjusted_output)
        output = self.fc2(output)
        return output, state


class ConbaNoState(nn.Module):
    def __init__(self, input_dim):
        super(ConbaNoState, self).__init__()
        self.selective_fc = nn.Linear(input_dim, input_dim)
        self.swish = nn.SiLU()
        self.control_weight = nn.Parameter(torch.ones(input_dim))
        self.C = nn.Linear(input_dim, input_dim, bias=False)
        self.fc2 = nn.Linear(input_dim, input_dim)

    def forward(self, x, previous_state=None):
        if x.dim() > 2:
            x = x.view(x.size(0), -1)
        selective_output = self.swish(self.selective_fc(x)) * x
        adjusted_output = selective_output * self.control_weight
        output = self.C(adjusted_output)
        output = self.fc2(output)
        return output, None


# SimCLR 模型
class SimCLRWithLayer(nn.Module):
    def __init__(self, layer_class, input_dim=768):
        super(SimCLRWithLayer, self).__init__()
        self.layer = layer_class(input_dim)

    def forward(self, x, previous_state=None):
        return self.layer(x, previous_state)


# 训练函数
def train_contrastive_model(encoded_samples, val_encoded_samples, contrastive_model, epochs=projectionHeadEpoch,
                            temperature=Temperature, patience=3, exp_name=""):
    print(f"\n{'=' * 50}")
    print(f"Training Model: {exp_name}")
    print(f"{'-' * 50}")
    print(f"Epochs: {epochs} | Batch Size: {ProjectionBatchSize} | Learning Rate: {lr1}")
    print(f"Temperature: {temperature} | Patience: {patience}")
    print(f"{'=' * 50}\n")

    train_dataset = CodeCloneDataset(encoded_samples)
    val_dataset = CodeCloneDataset(val_encoded_samples)
    train_dataloader = DataLoader(train_dataset, batch_size=ProjectionBatchSize, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=ProjectionBatchSize, shuffle=False)
    optimizer = optim.Adam(contrastive_model.parameters(), lr=lr1)
    contrastive_model.train()
    losses = []
    val_losses = []
    best_val_loss = float('inf')
    no_improvement_count = 0

    for epoch in range(epochs):
        total_loss = 0
        contrastive_model.train()
        for anchor, positives, negatives in train_dataloader:
            optimizer.zero_grad()
            anchor = anchor.to(device)
            positives = [pos.to(device) for pos in positives]
            negatives = [neg.to(device) for neg in negatives]
            anchor, state_anchor = contrastive_model(anchor)
            positives = [contrastive_model(pos)[0] for pos in positives]
            negatives = [contrastive_model(neg)[0] for neg in negatives]
            anchor = anchor.view(anchor.size(0), -1)
            positives = [pos.view(pos.size(0), -1) for pos in positives]
            negatives = [neg.view(neg.size(0), -1) for neg in negatives]
            loss = simclr_contrastive_loss(anchor, positives, negatives, temperature)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_train_loss = total_loss / len(train_dataloader)
        losses.append(avg_train_loss)

        # 验证
        contrastive_model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for anchor, positives, negatives in val_dataloader:
                anchor = anchor.to(device)
                positives = [pos.to(device) for pos in positives]
                negatives = [neg.to(device) for neg in negatives]
                anchor, _ = contrastive_model(anchor)
                positives = [contrastive_model(pos)[0] for pos in positives]
                negatives = [contrastive_model(neg)[0] for neg in negatives]
                anchor = anchor.view(anchor.size(0), -1)
                positives = [pos.view(pos.size(0), -1) for pos in positives]
                negatives = [neg.view(neg.size(0), -1) for neg in negatives]
                val_loss = simclr_contrastive_loss(anchor, positives, negatives, temperature)
                total_val_loss += val_loss.item()
        avg_val_loss = total_val_loss / len(val_dataloader)
        val_losses.append(avg_val_loss)

        print(f"Epoch {epoch + 1:03d}/{epochs} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            no_improvement_count = 0
        else:
            no_improvement_count += 1
            if no_improvement_count >= patience:
                print(f"\nEarly stopping triggered at epoch {epoch + 1}")
                break

    return losses, val_losses


# T-SNE 可视化
def plot_tsne(samples, contrastive_model, title, filename):
    embeddings = []
    labels = []
    for sample in samples[:10]:
        anchor = sample['code_embedding'].squeeze().numpy()
        embeddings.append(anchor)
        labels.append('anchor')
        for pos in sample['positive_embeddings']:
            embeddings.append(pos.squeeze().numpy())
            labels.append('positive')
        for neg in sample['negative_embeddings']:
            embeddings.append(neg.squeeze().numpy())
            labels.append('negative')
    if contrastive_model:
        contrastive_model.eval()
        with torch.no_grad():
            embeddings = contrastive_model(torch.tensor(embeddings).to(device))[0].cpu().numpy()
    embeddings = np.array(embeddings)
    tsne = TSNE(n_components=2, random_state=42)
    tsne_results = tsne.fit_transform(embeddings)
    plt.figure(figsize=(8, 6))
    for label in ['anchor', 'positive', 'negative']:
        idx = [i for i, lbl in enumerate(labels) if lbl == label]
        plt.scatter(tsne_results[idx, 0], tsne_results[idx, 1], label=label)
    plt.legend()
    plt.title(title)
    plt.savefig(filename)
    plt.close()


# 修复分类器模型的输入维度问题

class MLPClassifier(nn.Module):
    def __init__(self, input_dim):
        super(MLPClassifier, self).__init__()
        self.fc1 = nn.Linear(input_dim*2, 256)  # 输入是两个向量的拼接
        self.fc2 = nn.Linear(256, 64)
        self.fc3 = nn.Linear(64, 1)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)
        
    def forward(self, x1, x2):
        # Ensure correct input shapes
        if x1.dim() > 2:
            x1 = x1.reshape(x1.size(0), -1)
        if x2.dim() > 2:
            x2 = x2.reshape(x2.size(0), -1)
            
        # 拼接两个代码嵌入
        x = torch.cat([x1, x2], dim=1)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = torch.sigmoid(self.fc3(x))
        return x


class GRUClassifier(nn.Module):
    def __init__(self, input_dim):
        super(GRUClassifier, self).__init__()
        self.gru = nn.GRU(input_dim, 256, batch_first=True)
        self.fc1 = nn.Linear(256*2, 64)  # 两个GRU输出的拼接
        self.fc2 = nn.Linear(64, 1)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)
        
    def forward(self, x1, x2):
        # Ensure correct input shapes
        if x1.dim() > 2:
            x1 = x1.reshape(x1.size(0), -1)
        if x2.dim() > 2:
            x2 = x2.reshape(x2.size(0), -1)
            
        # 确保输入形状正确 [batch, seq_len=1, features]
        x1 = x1.unsqueeze(1)
        x2 = x2.unsqueeze(1)
        _, h1 = self.gru(x1)
        _, h2 = self.gru(x2)
        # 拼接两个输出
        x = torch.cat([h1.squeeze(0), h2.squeeze(0)], dim=1)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = torch.sigmoid(self.fc2(x))
        return x


class CNNClassifier(nn.Module):
    def __init__(self, input_dim):
        super(CNNClassifier, self).__init__()
        # 对于CNN，我们需要调整输入形状
        self.conv1 = nn.Conv1d(1, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool1d(2, stride=2)
        # 计算池化后的特征数量
        pooled_size = input_dim // 2
        self.fc1 = nn.Linear(pooled_size * 64 * 2, 256)  # 两个CNN输出的拼接
        self.fc2 = nn.Linear(256, 64)
        self.fc3 = nn.Linear(64, 1)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)
        self.input_dim = input_dim
        
    def forward(self, x1, x2):
        # Ensure correct input shapes
        if x1.dim() > 2:
            x1 = x1.reshape(x1.size(0), -1)
        if x2.dim() > 2:
            x2 = x2.reshape(x2.size(0), -1)
            
        # 调整输入形状为 [batch, channels=1, features]
        x1 = x1.unsqueeze(1)
        x2 = x2.unsqueeze(1)
        
        x1 = self.pool(self.relu(self.conv1(x1)))
        x2 = self.pool(self.relu(self.conv1(x2)))
        
        x1 = x1.reshape(x1.size(0), -1)
        x2 = x2.reshape(x2.size(0), -1)
        
        x = torch.cat([x1, x2], dim=1)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = torch.sigmoid(self.fc3(x))
        return x


class LSTMClassifier(nn.Module):
    def __init__(self, input_dim):
        super(LSTMClassifier, self).__init__()
        self.lstm = nn.LSTM(input_dim, 256, batch_first=True)
        self.fc1 = nn.Linear(256*2, 64)  # 两个LSTM输出的拼接
        self.fc2 = nn.Linear(64, 1)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)
        
    def forward(self, x1, x2):
        # Ensure correct input shapes
        if x1.dim() > 2:
            x1 = x1.reshape(x1.size(0), -1)
        if x2.dim() > 2:
            x2 = x2.reshape(x2.size(0), -1)
            
        # 确保输入形状正确 [batch, seq_len=1, features]
        x1 = x1.unsqueeze(1)
        x2 = x2.unsqueeze(1)
        
        _, (h1, _) = self.lstm(x1)
        _, (h2, _) = self.lstm(x2)
        
        x = torch.cat([h1.squeeze(0), h2.squeeze(0)], dim=1)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = torch.sigmoid(self.fc2(x))
        return x


class BiLSTMClassifier(nn.Module):
    def __init__(self, input_dim):
        super(BiLSTMClassifier, self).__init__()
        self.bilstm = nn.LSTM(input_dim, 128, batch_first=True, bidirectional=True)
        self.fc1 = nn.Linear(256*2, 64)  # 两个BiLSTM输出的拼接 (128*2方向*2个输入)
        self.fc2 = nn.Linear(64, 1)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)
        
    def forward(self, x1, x2):
        # Ensure correct input shapes
        if x1.dim() > 2:
            x1 = x1.reshape(x1.size(0), -1)
        if x2.dim() > 2:
            x2 = x2.reshape(x2.size(0), -1)
            
        # 确保输入形状正确 [batch, seq_len=1, features]
        x1 = x1.unsqueeze(1)
        x2 = x2.unsqueeze(1)
        
        _, (h1, _) = self.bilstm(x1)
        _, (h2, _) = self.bilstm(x2)
        
        # 双向LSTM有2个方向，需要合并
        h1 = h1.transpose(0, 1).contiguous().view(h1.size(1), -1)
        h2 = h2.transpose(0, 1).contiguous().view(h2.size(1), -1)
        
        x = torch.cat([h1, h2], dim=1)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = torch.sigmoid(self.fc2(x))
        return x


class RNNClassifier(nn.Module):
    def __init__(self, input_dim):
        super(RNNClassifier, self).__init__()
        self.rnn = nn.RNN(input_dim, 256, batch_first=True)
        self.fc1 = nn.Linear(256*2, 64)  # 两个RNN输出的拼接
        self.fc2 = nn.Linear(64, 1)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)
        
    def forward(self, x1, x2):
        # Ensure correct input shapes
        if x1.dim() > 2:
            x1 = x1.reshape(x1.size(0), -1)
        if x2.dim() > 2:
            x2 = x2.reshape(x2.size(0), -1)
            
        # 确保输入形状正确 [batch, seq_len=1, features]
        x1 = x1.unsqueeze(1)
        x2 = x2.unsqueeze(1)
        
        _, h1 = self.rnn(x1)
        _, h2 = self.rnn(x2)
        
        x = torch.cat([h1.squeeze(0), h2.squeeze(0)], dim=1)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = torch.sigmoid(self.fc2(x))
        return x


class ResNetClassifier(nn.Module):
    def __init__(self, input_dim):
        super(ResNetClassifier, self).__init__()
        # 残差块
        self.fc1 = nn.Linear(input_dim, input_dim)
        self.fc2 = nn.Linear(input_dim, input_dim)
        # 分类层
        self.fc3 = nn.Linear(input_dim*2, 256)  # 两个残差输出的拼接
        self.fc4 = nn.Linear(256, 64)
        self.fc5 = nn.Linear(64, 1)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)
        
    def forward(self, x1, x2):
        # Ensure correct input shapes
        if x1.dim() > 2:
            x1 = x1.reshape(x1.size(0), -1)
        if x2.dim() > 2:
            x2 = x2.reshape(x2.size(0), -1)
            
        # 第一个输入的残差连接
        res1 = x1
        x1 = self.relu(self.fc1(x1))
        x1 = self.fc2(x1) + res1
        
        # 第二个输入的残差连接
        res2 = x2
        x2 = self.relu(self.fc1(x2))
        x2 = self.fc2(x2) + res2
        
        # 拼接两个输出
        x = torch.cat([x1, x2], dim=1)
        x = self.relu(self.fc3(x))
        x = self.dropout(x)
        x = self.relu(self.fc4(x))
        x = self.dropout(x)
        x = torch.sigmoid(self.fc5(x))
        return x


# 创建分类器数据集
class CloneClassifierDataset(Dataset):
    def __init__(self, samples, contrastive_model):
        self.contrastive_model = contrastive_model
        self.contrastive_model.eval()
        
        self.code_pairs = []
        self.labels = []
        
        with torch.no_grad():
            for sample in samples:
                anchor = sample['code_embedding'].to(device)
                anchor_emb, _ = self.contrastive_model(anchor)
                
                # 正例
                for pos in sample['positive_embeddings']:
                    pos = pos.to(device)
                    pos_emb, _ = self.contrastive_model(pos)
                    self.code_pairs.append((anchor_emb.cpu(), pos_emb.cpu()))
                    self.labels.append(1)
                
                # 负例
                for neg in sample['negative_embeddings']:
                    neg = neg.to(device)
                    neg_emb, _ = self.contrastive_model(neg)
                    self.code_pairs.append((anchor_emb.cpu(), neg_emb.cpu()))
                    self.labels.append(0)
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        return self.code_pairs[idx][0], self.code_pairs[idx][1], self.labels[idx]


# 训练分类器
def train_classifier(train_dataset, val_dataset, classifier, epochs=classifyEpoch, batch_size=32, patience=3, exp_name=""):
    print(f"\n{'=' * 50}")
    print(f"Training Classifier: {exp_name}")
    print(f"{'-' * 50}")
    print(f"Epochs: {epochs} | Batch Size: {batch_size} | Learning Rate: {lr2}")
    print(f"{'=' * 50}\n")
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    criterion = nn.BCELoss()
    optimizer = optim.Adam(classifier.parameters(), lr=lr2)
    
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    no_improvement_count = 0
    
    for epoch in range(epochs):
        # 训练
        classifier.train()
        total_loss = 0
        for code1, code2, labels in train_loader:
            code1, code2, labels = code1.to(device), code2.to(device), labels.float().to(device)
            
            optimizer.zero_grad()
            outputs = classifier(code1, code2).squeeze()
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        avg_train_loss = total_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        
        # 验证
        classifier.eval()
        total_val_loss = 0
        with torch.no_grad():
            for code1, code2, labels in val_loader:
                code1, code2, labels = code1.to(device), code2.to(device), labels.float().to(device)
                outputs = classifier(code1, code2).squeeze()
                loss = criterion(outputs, labels)
                total_val_loss += loss.item()
        
        avg_val_loss = total_val_loss / len(val_loader)
        val_losses.append(avg_val_loss)
        
        print(f"Epoch {epoch + 1:03d}/{epochs} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")
        
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            no_improvement_count = 0
        else:
            no_improvement_count += 1
            if no_improvement_count >= patience:
                print(f"\nEarly stopping triggered at epoch {epoch + 1}")
                break
    
    return train_losses, val_losses


# 评估分类器
def evaluate_classifier(test_dataset, classifier):
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
    classifier.eval()
    
    all_labels = []
    all_preds = []
    all_probs = []
    
    with torch.no_grad():
        for code1, code2, labels in test_loader:
            code1, code2 = code1.to(device), code2.to(device)
            outputs = classifier(code1, code2).squeeze()
            probs = outputs.cpu().numpy()
            preds = (probs > 0.5).astype(int)
            
            all_labels.extend(labels.numpy())
            all_preds.extend(preds)
            all_probs.extend(probs)
    
    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds)
    recall = recall_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds)
    auc_score = roc_auc_score(all_labels, all_probs)
    
    # 计算ROC曲线
    fpr, tpr, _ = roc_curve(all_labels, all_probs)
    
    return acc, f1, recall, precision, auc_score, fpr, tpr


# 绘制ROC曲线
def plot_roc_curves(roc_data, filename):
    plt.figure(figsize=(10, 8))
    
    for name, (fpr, tpr, auc_score) in roc_data.items():
        plt.plot(fpr, tpr, label=f'{name} (AUC = {auc_score:.3f})')
    
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves for Different Classifiers')
    plt.legend(loc='lower right')
    plt.savefig(filename)
    plt.close()


# 实验运行与结果保存
results = {}
roc_data = {}

# 1. 训练对比学习模型
contrastive_model = SimCLRWithLayer(ConbaLayer).to(device)
losses, val_losses = train_contrastive_model(train_samples, val_samples, contrastive_model, 
                                            epochs=projectionHeadEpoch, 
                                            temperature=Temperature, 
                                            patience=projectionHeadPatience,
                                            exp_name="ConbaLayer")

# 2. 创建分类器数据集
train_classifier_dataset = CloneClassifierDataset(train_samples, contrastive_model)
val_classifier_dataset = CloneClassifierDataset(val_samples, contrastive_model)
test_classifier_dataset = CloneClassifierDataset(test_samples, contrastive_model)

# 3. 训练和评估不同的分类器模型
classifier_models = {
    "MLP": MLPClassifier(768).to(device),
    "GRU": GRUClassifier(768).to(device),
    "CNN": CNNClassifier(768).to(device),
    "LSTM": LSTMClassifier(768).to(device),
    "BiLSTM": BiLSTMClassifier(768).to(device),
    "RNN": RNNClassifier(768).to(device),
    "ResNet": ResNetClassifier(768).to(device)
}

for name, classifier in classifier_models.items():
    print(f"\nTraining {name} Classifier...")
    train_losses, val_losses = train_classifier(
        train_classifier_dataset, 
        val_classifier_dataset, 
        classifier, 
        epochs=10, 
        batch_size=BatchSize, 
        patience=3,
        exp_name=name
    )
    
    # 评估
    acc, f1, recall, precision, auc_score, fpr, tpr = evaluate_classifier(test_classifier_dataset, classifier)
    
    # 保存结果
    results[name] = {
        "accuracy": float(acc),
        "f1_score": float(f1),
        "recall": float(recall),
        "precision": float(precision),
        "auc": float(auc_score)
    }
    
    # 保存ROC数据
    roc_data[name] = (fpr, tpr, auc_score)
    
    print(f"{name} Classifier Results:")
    print(f"Accuracy: {acc:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"AUC: {auc_score:.4f}")

# 4. 消融实验
ablation_models = {
    "ConbaNoControl": SimCLRWithLayer(ConbaNoControl).to(device),
    "ConbaNoFeedback": SimCLRWithLayer(ConbaNoFeedback).to(device),
    "ConbaNoState": SimCLRWithLayer(ConbaNoState).to(device)
}

ablation_results = {}
ablation_roc_data = {}

for name, model in ablation_models.items():
    print(f"\nTraining Ablation Model: {name}...")
    losses, val_losses = train_contrastive_model(
        train_samples, 
        val_samples, 
        model, 
        epochs=projectionHeadEpoch, 
        temperature=Temperature, 
        patience=projectionHeadPatience,
        exp_name=name
    )
    
    # 创建分类器数据集
    train_ablation_dataset = CloneClassifierDataset(train_samples, model)
    val_ablation_dataset = CloneClassifierDataset(val_samples, model)
    test_ablation_dataset = CloneClassifierDataset(test_samples, model)
    
    # 使用最佳分类器进行评估
    best_classifier_name = max(results, key=lambda k: results[k]["f1_score"])
    best_classifier_class = type(classifier_models[best_classifier_name])
    ablation_classifier = best_classifier_class(768).to(device)
    
    train_losses, val_losses = train_classifier(
        train_ablation_dataset, 
        val_ablation_dataset, 
        ablation_classifier, 
        epochs=classifyEpoch,
        batch_size=BatchSize, 
        patience=3,
        exp_name=f"{name} with {best_classifier_name}"
    )
    
    # 评估
    acc, f1, recall, precision, auc_score, fpr, tpr = evaluate_classifier(test_ablation_dataset, ablation_classifier)
    
    # 保存结果
    ablation_results[name] = {
        "accuracy": float(acc),
        "f1_score": float(f1),
        "recall": float(recall),
        "precision": float(precision),
        "auc": float(auc_score)
    }
    
    # 保存ROC数据
    ablation_roc_data[name] = (fpr, tpr, auc_score)
    
    print(f"{name} Results:")
    print(f"Accuracy: {acc:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"AUC: {auc_score:.4f}")

# 5. 可视化结果
# 绘制ROC曲线
plot_roc_curves(roc_data, "classifier_roc_curves.png")
plot_roc_curves(ablation_roc_data, "ablation_roc_curves.png")

# 绘制分类器性能对比图
plt.figure(figsize=(12, 8))
metrics = ["accuracy", "f1_score", "recall", "precision", "auc"]
x = np.arange(len(metrics))
width = 0.1
multiplier = 0

for name, result in results.items():
    offset = width * multiplier
    values = [result[metric] for metric in metrics]
    plt.bar(x + offset, values, width, label=name)
    multiplier += 1

plt.xlabel('Metrics')
plt.ylabel('Scores')
plt.title('Performance Comparison of Different Classifiers')
plt.xticks(x + width * (len(results) - 1) / 2, metrics)
plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), ncol=4)
plt.tight_layout()
plt.savefig("classifier_comparison.png")
plt.close()

# 绘制消融实验性能对比图
plt.figure(figsize=(12, 8))
x = np.arange(len(metrics))
width = 0.15
multiplier = 0

# 添加完整模型结果
full_model_result = {
    "accuracy": results[best_classifier_name]["accuracy"],
    "f1_score": results[best_classifier_name]["f1_score"],
    "recall": results[best_classifier_name]["recall"],
    "precision": results[best_classifier_name]["precision"],
    "auc": results[best_classifier_name]["auc"]
}
ablation_results["Full Model"] = full_model_result
ablation_roc_data["Full Model"] = roc_data[best_classifier_name]

for name, result in ablation_results.items():
    offset = width * multiplier
    values = [result[metric] for metric in metrics]
    plt.bar(x + offset, values, width, label=name)
    multiplier += 1

plt.xlabel('Metrics')
plt.ylabel('Scores')
plt.title('Ablation Study Performance Comparison')
plt.xticks(x + width * (len(ablation_results) - 1) / 2, metrics)
plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), ncol=3)
plt.tight_layout()
plt.savefig("ablation_comparison.png")
plt.close()

# 保存所有结果到JSON文件
all_results = {
    "classifiers": results,
    "ablation_study": ablation_results
}

with open("code_clone_detection_results.json", "w") as f:
    json.dump(all_results, f, indent=4)

print("\nExperiment completed. Results saved to code_clone_detection_results.json")

# 绘制T-SNE可视化
plot_tsne(test_samples, None, "Original Embeddings", "original_tsne.png")
plot_tsne(test_samples, contrastive_model, "ConbaLayer Embeddings", "conba_tsne.png")

# 为每个消融模型绘制T-SNE
for name, model in ablation_models.items():
    plot_tsne(test_samples, model, f"{name} Embeddings", f"{name.lower()}_tsne.png")

# 绘制训练损失曲线
plt.figure(figsize=(10, 6))
plt.plot(losses, label='Training Loss')
plt.plot(val_losses, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Validation Loss for ConbaLayer')
plt.legend()
plt.savefig("contrastive_loss.png")
plt.close()

print("\nAll visualizations have been saved.")