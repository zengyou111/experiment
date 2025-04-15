import os
import json
import torch
import torch.nn.functional as F
import random
import time
from transformers import AutoTokenizer, AutoModel
from torch.utils.data import DataLoader, Dataset
from torch import nn, optim
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import numpy as np
from tqdm import tqdm

# 设置设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 参数配置
projectionHeadEpoch = 10
Temperature = 0.2
projectionHeadPatience = 2
Trainsize = 10
lr1 = 3e-6
ProjectionBatchSize = 1
BatchSize = 64
modelPath = '../../../model/unixcoder/'  # 示例模型路径
langPath = '../../../Datasets/code_pairs_java_python.jsonl'  # 示例数据集

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


# 对比实验变体（7个网络架构）
class MLPLayer(nn.Module):
    def __init__(self, input_dim):
        super(MLPLayer, self).__init__()
        self.fc1 = nn.Linear(input_dim, input_dim)
        self.fc2 = nn.Linear(input_dim, input_dim)
        self.relu = nn.ReLU()

    def forward(self, x, previous_state=None):
        if x.dim() > 2:
            x = x.view(x.size(0), -1)
        output = self.relu(self.fc1(x))
        output = self.fc2(output)
        return output, None


class GRULayer(nn.Module):
    def __init__(self, input_dim):
        super(GRULayer, self).__init__()
        self.gru = nn.GRU(input_dim, input_dim, batch_first=True)
        self.fc = nn.Linear(input_dim, input_dim)

    def forward(self, x, previous_state=None):
        if x.dim() > 2:
            x = x.view(x.size(0), -1)
        if previous_state is None:
            previous_state = torch.zeros(1, x.size(0), x.size(-1), device=x.device)
        output, state = self.gru(x.unsqueeze(1), previous_state)
        output = self.fc(output.squeeze(1))
        return output, state


class CNNLayer(nn.Module):
    def __init__(self, input_dim):
        super(CNNLayer, self).__init__()
        self.conv = nn.Conv1d(in_channels=1, out_channels=1, kernel_size=3, padding=1)
        self.fc = nn.Linear(input_dim, input_dim)
        self.relu = nn.ReLU()

    def forward(self, x, previous_state=None):
        if x.dim() > 2:
            x = x.view(x.size(0), -1)
        x = x.unsqueeze(1)  # [batch_size, 1, embedding_dim]
        output = self.relu(self.conv(x))
        output = self.fc(output.squeeze(1))
        return output, None


class LSTMLayer(nn.Module):
    def __init__(self, input_dim):
        super(LSTMLayer, self).__init__()
        self.lstm = nn.LSTM(input_dim, input_dim, batch_first=True)
        self.fc = nn.Linear(input_dim, input_dim)

    def forward(self, x, previous_state=None):
        if x.dim() > 2:
            x = x.view(x.size(0), -1)
        if previous_state is None:
            previous_state = (torch.zeros(1, x.size(0), x.size(-1), device=x.device),
                              torch.zeros(1, x.size(0), x.size(-1), device=x.device))
        output, (hn, cn) = self.lstm(x.unsqueeze(1), previous_state)
        output = self.fc(output.squeeze(1))
        return output, (hn, cn)


class BiLSTMLayer(nn.Module):
    def __init__(self, input_dim):
        super(BiLSTMLayer, self).__init__()
        self.bilstm = nn.LSTM(input_dim, input_dim // 2, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(input_dim, input_dim)

    def forward(self, x, previous_state=None):
        if x.dim() > 2:
            x = x.view(x.size(0), -1)
        if previous_state is None:
            previous_state = (torch.zeros(2, x.size(0), x.size(-1) // 2, device=x.device),
                              torch.zeros(2, x.size(0), x.size(-1) // 2, device=x.device))
        output, (hn, cn) = self.bilstm(x.unsqueeze(1), previous_state)
        output = self.fc(output.squeeze(1))
        return output, (hn, cn)


class RNNLayer(nn.Module):
    def __init__(self, input_dim):
        super(RNNLayer, self).__init__()
        self.rnn = nn.RNN(input_dim, input_dim, batch_first=True)
        self.fc = nn.Linear(input_dim, input_dim)

    def forward(self, x, previous_state=None):
        if x.dim() > 2:
            x = x.view(x.size(0), -1)
        if previous_state is None:
            previous_state = torch.zeros(1, x.size(0), x.size(-1), device=x.device)
        output, state = self.rnn(x.unsqueeze(1), previous_state)
        output = self.fc(output.squeeze(1))
        return output, state


class ResNetLayer(nn.Module):
    def __init__(self, input_dim):
        super(ResNetLayer, self).__init__()
        self.fc1 = nn.Linear(input_dim, input_dim)
        self.fc2 = nn.Linear(input_dim, input_dim)
        self.relu = nn.ReLU()

    def forward(self, x, previous_state=None):
        if x.dim() > 2:
            x = x.view(x.size(0), -1)
        residual = x
        output = self.relu(self.fc1(x))
        output = self.fc2(output)
        output = output + residual  # 残差连接
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


# 评估函数
def evaluate_clones(samples, model, contrastive_model):
    model.eval()
    contrastive_model.eval()
    all_labels = []
    all_preds = []
    for sample in samples:
        anchor = sample['code_embedding'].to(device)
        positives = [pos.to(device) for pos in sample['positive_embeddings']]
        negatives = [neg.to(device) for neg in sample['negative_embeddings']]
        with torch.no_grad():
            anchor, _ = contrastive_model(anchor)
            positives = [contrastive_model(pos)[0] for pos in positives]
            negatives = [contrastive_model(neg)[0] for neg in negatives]
        for pos in positives:
            similarity = F.cosine_similarity(anchor, pos, dim=-1).item()
            all_labels.append(1)
            all_preds.append(1 if similarity > 0.5 else 0)
        for neg in negatives:
            similarity = F.cosine_similarity(anchor, neg, dim=-1).item()
            all_labels.append(0)
            all_preds.append(0 if similarity <= 0.5 else 1)
    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds)
    recall = recall_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds)
    return acc, f1, recall, precision


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


# 实验运行与结果保存
results = {}

# 1. 完整模型
contrastive_model = SimCLRWithLayer(ConbaLayer).to(device)
losses, val_losses = train_contrastive_model(train_samples, val_samples, contrastive_model, exp_name="Full Conba Model")
acc, f1, recall, precision = evaluate_clones(test_samples, model, contrastive_model)
results['Full_Conba'] = {'train_losses': losses, 'val_losses': val_losses, 'acc': acc, 'f1': f1, 'recall': recall,
                         'precision': precision}
plot_tsne(test_samples, contrastive_model, 'T-SNE of Full Conba', 'tsne_full_conba.png')

# 2. 消融实验
ablation_configs = [
    ('No_Control', ConbaNoControl, "Removed control_weight to disable dynamic feature weighting"),
    ('No_Feedback', ConbaNoFeedback, "Removed feedback_weight to disable state adjustment"),
    ('No_State', ConbaNoState, "Removed state mechanism (A and B matrices) to eliminate historical context")
]
for name, layer_class, desc in ablation_configs:
    print(f"\n{'=' * 50}")
    print(f"Ablation Experiment: {name}")
    print(f"Description: {desc}")
    print(f"{'=' * 50}")
    contrastive_model = SimCLRWithLayer(layer_class).to(device)
    losses, val_losses = train_contrastive_model(train_samples, val_samples, contrastive_model, exp_name=name)
    acc, f1, recall, precision = evaluate_clones(test_samples, model, contrastive_model)
    results[name] = {'train_losses': losses, 'val_losses': val_losses, 'acc': acc, 'f1': f1, 'recall': recall,
                     'precision': precision}
    plot_tsne(test_samples, contrastive_model, f'T-SNE of {name}', f'tsne_{name.lower()}.png')

# 3. 对比实验（7个网络架构）
contrast_configs = [
    ('MLP', MLPLayer, "Replaced ConbaLayer with a Multi-Layer Perceptron"),
    ('GRU', GRULayer, "Replaced ConbaLayer with a Gated Recurrent Unit"),
    ('CNN', CNNLayer, "Replaced ConbaLayer with a 1D Convolutional Neural Network"),
    ('LSTM', LSTMLayer, "Replaced ConbaLayer with a Long Short-Term Memory network"),
    ('BiLSTM', BiLSTMLayer, "Replaced ConbaLayer with a Bidirectional LSTM"),
    ('RNN', RNNLayer, "Replaced ConbaLayer with a Recurrent Neural Network"),
    ('ResNet', ResNetLayer, "Replaced ConbaLayer with a Residual Network")
]
for name, layer_class, desc in contrast_configs:
    print(f"\n{'=' * 50}")
    print(f"Contrast Experiment: {name}")
    print(f"Description: {desc}")
    print(f"{'=' * 50}")
    contrastive_model = SimCLRWithLayer(layer_class).to(device)
    losses, val_losses = train_contrastive_model(train_samples, val_samples, contrastive_model, exp_name=name)
    acc, f1, recall, precision = evaluate_clones(test_samples, model, contrastive_model)
    results[name] = {'train_losses': losses, 'val_losses': val_losses, 'acc': acc, 'f1': f1, 'recall': recall,
                     'precision': precision}
    plot_tsne(test_samples, contrastive_model, f'T-SNE of {name}', f'tsne_{name.lower()}.png')

# 保存结果到 JSON
with open('experiment_results.json', 'w') as f:
    json.dump(results, f, indent=4)

# 绘制折线图（损失）
plt.figure(figsize=(12, 8))
colors = {
    'Full_Conba': 'blue', 'No_Control': 'red', 'No_Feedback': 'green', 'No_State': 'purple',
    'MLP': 'orange', 'GRU': 'brown', 'CNN': 'cyan', 'LSTM': 'pink', 'BiLSTM': 'gray', 'RNN': 'magenta',
    'ResNet': 'yellow'
}
for name, data in results.items():
    epochs = range(1, len(data['train_losses']) + 1)
    plt.plot(epochs, data['val_losses'], label=name, color=colors[name])
plt.xlabel('Epochs')
plt.ylabel('Validation Loss')
plt.title('Validation Loss Across Experiments')
plt.legend(loc='upper right', bbox_to_anchor=(1.15, 1))
plt.savefig('val_loss_comparison.png')
plt.close()

plt.figure(figsize=(12, 8))
for name, data in results.items():
    epochs = range(1, len(data['train_losses']) + 1)
    plt.plot(epochs, data['train_losses'], label=name, color=colors[name])
plt.xlabel('Epochs')
plt.ylabel('Training Loss')
plt.title('Training Loss Across Experiments')
plt.legend(loc='upper right', bbox_to_anchor=(1.15, 1))
plt.savefig('train_loss_comparison.png')
plt.close()

# 性能指标对比图（消融实验和对比实验分开）
metrics = ['acc', 'f1', 'recall', 'precision']
metric_colors = {'acc': 'blue', 'f1': 'red', 'recall': 'green', 'precision': 'purple'}

# 消融实验图
plt.figure(figsize=(10, 6))
ablation_models = ['Full_Conba', 'No_Control', 'No_Feedback', 'No_State']
for metric in metrics:
    values = [results[name][metric] for name in ablation_models]
    plt.plot(ablation_models, values, marker='o', label=metric.capitalize(), color=metric_colors[metric])
plt.xlabel('Model')
plt.ylabel('Performance Metric')
plt.title('Ablation Study: Performance Metrics')
plt.legend(loc='upper right')
plt.xticks(rotation=45)
plt.savefig('ablation_metrics.png')
plt.close()

# 对比实验图
plt.figure(figsize=(12, 8))
contrast_models = ['MLP', 'GRU', 'CNN', 'LSTM', 'BiLSTM', 'RNN', 'ResNet']
for metric in metrics:
    values = [results[name][metric] for name in contrast_models]
    plt.plot(contrast_models, values, marker='o', label=metric.capitalize(), color=metric_colors[metric])
plt.xlabel('Model')
plt.ylabel('Performance Metric')
plt.title('Contrast Study: Performance Metrics')
plt.legend(loc='upper right')
plt.xticks(rotation=45)
plt.savefig('contrast_metrics.png')
plt.close()
