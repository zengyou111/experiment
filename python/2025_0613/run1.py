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
from itertools import product
from sklearn.model_selection import train_test_split

# ==================== 模型参数配置区域 ====================
# 设备配置
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ==================== 数据相关参数 ====================
# 训练数据大小 - 每个数据集加载的样本数量
Trainsize = 10  # 建议范围: 50-500，影响训练时间和模型性能

# 数据分割比例
train_ratio = 0.6    # 训练集比例
val_ratio = 0.2      # 验证集比例  
test_ratio = 0.2     # 测试集比例
# 注意: train_ratio + val_ratio + test_ratio = 1.0

# 数据预处理参数
max_sequence_length = 512  # 代码序列最大长度，影响tokenizer截断

# ==================== 对比学习模型训练参数 ====================
# 对比学习训练轮次
projectionHeadEpoch = 5  # 建议范围: 30-100，过少可能欠拟合，过多可能过拟合

# 对比学习早停耐心值
projectionHeadPatience = 2  # 验证集损失不下降的容忍轮次，建议范围: 2-5

# 对比学习批次大小
ProjectionBatchSize = 1  # 对比学习的批次大小，通常设为1以便处理不同数量的正负样本

# 对比学习学习率
lr1 = 3e-6  # 对比学习模型学习率，建议范围: 1e-6到1e-4

# SimCLR温度系数
Temperature = 0.2  # 对比学习损失函数的温度参数，影响相似度计算的锐度
# 温度越小，模型对相似度差异越敏感，建议范围: 0.1-1.0

# ==================== 分类器训练参数 ====================
# 分类器训练轮次
classifyEpoch = 3  # 建议范围: 30-100

# 分类器早停耐心值
classifyPatience = 3  # 验证集损失不下降的容忍轮次

# 分类器批次大小
BatchSize = 64  # 分类器训练的批次大小，建议范围: 32-128

# 分类器学习率
lr2 = 1e-4  # 分类器学习率，通常比对比学习学习率大一些

# 分类器Dropout率
dropout_rate = 0.2  # 防止过拟合，建议范围: 0.1-0.5

# ==================== 实验设计参数 ====================
# 温度系数实验范围
temperature_range = [0.1, 0.2, 0.3, 0.5, 0.7, 1.0]  # 用于温度系数ROC分析

# 噪声鲁棒性测试参数
noise_types = ['synonym', 'insertion', 'swap', 'deletion']  # 四种噪声类型
noise_ratios = [0.05, 0.1, 0.15, 0.2]  # 噪声强度，表示受影响token的比例

# 消融实验加速参数
ablation_epochs = 3  # 消融实验中对比学习的训练轮次（减少以加速实验）
noise_test_epochs = 3  # 噪声测试中对比学习的训练轮次
noise_classifier_epochs = 3  # 噪声测试中分类器的训练轮次

# ==================== 可视化参数 ====================
# T-SNE可视化参数
tsne_max_samples = 20  # T-SNE可视化使用的最大样本数
tsne_perplexity = 30   # T-SNE的perplexity参数，影响聚类效果
tsne_random_state = 42 # T-SNE的随机种子

# 图像保存参数
figure_dpi = 300  # 图像分辨率
figure_format = 'png'  # 图像格式

# ==================== 模型架构参数 ====================
# 嵌入维度
embedding_dim = 768  # 预训练模型的嵌入维度，通常为768

# 分类器隐藏层参数
classifier_hidden_dims = {
    'mlp_hidden1': 256,      # MLP第一隐藏层维度
    'mlp_hidden2': 64,       # MLP第二隐藏层维度
    'rnn_hidden': 256,       # RNN类模型隐藏层维度
    'cnn_channels': 64,      # CNN卷积核数量
    'bilstm_hidden': 128,    # BiLSTM隐藏层维度（单向）
    'resnet_hidden1': 256,   # ResNet隐藏层1维度
    'resnet_hidden2': 64,    # ResNet隐藏层2维度
}

# ConbaLayer参数
conba_layer_params = {
    'use_bias': False,           # 线性层是否使用偏置
    'activation': 'swish',       # 激活函数类型: 'swish', 'relu', 'gelu'
    'init_weight_std': 1.0,      # 参数初始化标准差
}

# ==================== 数据路径配置 ====================
# 语言组合数据集路径
langs_path_dict = {
    'java-python': '../../../Datasets/code_pairs_java_python.jsonl',
    'cpp-python': '../../../Datasets/code_pairs_cpp_python.jsonl',
    'java-cpp': '../../../Datasets/code_pairs_java_c.jsonl',
    'java-cs': '../../../Datasets/code_pairs_java_cs.jsonl',
}

# 编码器模型路径
model_path_dict = {
    'codebert-base': '../../../model/codebert-base/',
    'codeExecutor': '../../../model/codeExecutor/',
    'CodeReviewer': '../../../model/CodeReviewer/',
    'graphCodeBert': '../../../model/graphCodeBert/',
    'longcoder-base': '../../../model/longcoder-base/',
    'unixcoder': '../../../model/unixcoder/',
}

# ==================== 实验控制参数 ====================
# 随机种子
random_seed = 42  # 保证实验可重复性
torch.manual_seed(random_seed)
np.random.seed(random_seed)
random.seed(random_seed)

# 实验阶段控制
run_baseline_comparison = True   # 是否运行基准对比实验
run_ablation_study = True        # 是否运行消融实验
run_noise_robustness = True      # 是否运行噪声鲁棒性测试
run_temperature_analysis = True  # 是否运行温度系数分析
generate_visualizations = True   # 是否生成可视化图表

# 性能评估指标
primary_metric = 'f1_score'  # 主要评估指标，用于选择最佳模型
evaluation_metrics = ['accuracy', 'f1_score', 'recall', 'precision', 'auc']

# ==================== 输出路径配置 ====================
# 结果保存路径
output_dir = './'  # 实验结果输出目录
results_files = {
    'baseline_comparison': 'encoder_classifier_comparison_results.json',
    'best_config': 'best_baseline_config.json', 
    'complete_results': 'complete_experiment_results.json',
    'experiment_report': 'experiment_report.txt',
}

# 可视化文件名
visualization_files = {
    'encoder_comparison': 'encoder_classifier_performance_comparison.png',
    'ablation_results': 'ablation_study_results.png',
    'noise_robustness': 'noise_robustness_results.png',
    'temperature_roc': 'temperature_roc_curves.png',
    'ablation_roc': 'ablation_roc_curves.png',
    'original_tsne': 'original_embeddings_tsne.png',
    'full_conba_tsne': 'full_conba_embeddings_tsne.png',
    'no_conba_tsne': 'no_conba_embeddings_tsne.png',
}

# ==================== 计算资源参数 ====================
# GPU内存管理
torch.cuda.empty_cache() if torch.cuda.is_available() else None
max_gpu_memory = 0.9  # 最大GPU内存使用比例

# 并行处理
num_workers = 4  # DataLoader的工作进程数
pin_memory = True if torch.cuda.is_available() else False  # 是否将数据固定在内存中

# ==================== 实验验证参数 ====================
# 交叉验证
k_fold = 5  # K折交叉验证的K值（如果需要的话）

# 统计显著性检验
confidence_level = 0.95  # 置信水平

# ==================== 日志和调试参数 ====================
# 日志级别
verbose = True  # 是否详细输出训练过程
log_interval = 10  # 训练过程中打印日志的间隔

# 调试模式
debug_mode = False  # 调试模式下使用更少的数据和更少的训练轮次
if debug_mode:
    Trainsize = 10
    projectionHeadEpoch = 5
    classifyEpoch = 5
    ablation_epochs = 5
    noise_test_epochs = 5

print("=" * 80)
print("ConbaLayer 代码克隆检测实验")
print("=" * 80)
print(f"设备: {device}")
print(f"训练数据大小: {Trainsize}")
print(f"对比学习训练轮次: {projectionHeadEpoch}")
print(f"分类器训练轮次: {classifyEpoch}")
print(f"温度系数: {Temperature}")
print(f"对比学习学习率: {lr1}")
print(f"分类器学习率: {lr2}")
print(f"批次大小: 对比学习={ProjectionBatchSize}, 分类器={BatchSize}")
print(f"主要评估指标: {primary_metric}")
print(f"编码器数量: {len(model_path_dict)}")
print(f"语言对数量: {len(langs_path_dict)}")
print(f"预计总组合数: {len(model_path_dict) * len(langs_path_dict) * 6}")  # 改为6个分类器
print("=" * 80)

# 接下来是原来的代码，从ConbaLayer定义开始...

# ConbaLayer定义
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

# SimCLR 模型
class SimCLRWithLayer(nn.Module):
    def __init__(self, layer_class, input_dim=768):
        super(SimCLRWithLayer, self).__init__()
        self.layer = layer_class(input_dim)

    def forward(self, x, previous_state=None):
        return self.layer(x, previous_state)

# 数据加载函数
def load_samples_from_jsonl(file_path, max_samples=Trainsize):
    samples = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            record = json.loads(line.strip())
            samples.append(record)
            if len(samples) >= max_samples:
                break
    return samples

# 编码样本函数
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

# 训练对比学习模型
def train_contrastive_model(encoded_samples, val_encoded_samples, contrastive_model, 
                            epochs=projectionHeadEpoch, temperature=Temperature, patience=3, exp_name=""):
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
    
    # 记录训练过程
    train_losses = []
    val_losses = []
    
    best_val_loss = float('inf')
    no_improvement_count = 0

    for epoch in range(epochs):
        # 训练阶段
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
        train_losses.append(avg_train_loss)

        # 验证阶段
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

    # 绘制训练曲线 - 只绘制损失曲线
    if epochs > 1 and len(train_losses) > 1:
        plt.figure(figsize=(10, 4))
        
        # 损失曲线
        plt.subplot(1, 1, 1)
        epochs_range = range(1, len(train_losses) + 1)
        plt.plot(epochs_range, train_losses, 'b-', label='Train Loss', linewidth=2, marker='o')
        plt.plot(epochs_range, val_losses, 'r-', label='Validation Loss', linewidth=2, marker='s')
        plt.title(f'Contrastive Learning Loss - {exp_name}')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        safe_exp_name = exp_name.replace('/', '_').replace('\\', '_').replace(':', '_')
        plt.savefig(f"contrastive_training_curves_{safe_exp_name}.png", dpi=300, bbox_inches='tight')
        plt.close()
        print(f"对比学习训练曲线已保存: contrastive_training_curves_{safe_exp_name}.png")

    return train_losses, val_losses, best_val_loss

# 完整的分类器模型定义 - 添加缺失的分类器
class MLPClassifier(nn.Module):
    def __init__(self, input_dim):
        super(MLPClassifier, self).__init__()
        self.fc1 = nn.Linear(input_dim * 2, 256)
        self.fc2 = nn.Linear(256, 64)
        self.fc3 = nn.Linear(64, 1)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)

    def forward(self, x1, x2):
        if x1.dim() > 2:
            x1 = x1.reshape(x1.size(0), -1)
        if x2.dim() > 2:
            x2 = x2.reshape(x2.size(0), -1)
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
        self.fc1 = nn.Linear(256 * 2, 64)
        self.fc2 = nn.Linear(64, 1)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)

    def forward(self, x1, x2):
        if x1.dim() > 2:
            x1 = x1.reshape(x1.size(0), -1)
        if x2.dim() > 2:
            x2 = x2.reshape(x2.size(0), -1)
        x1 = x1.unsqueeze(1)
        x2 = x2.unsqueeze(1)
        _, h1 = self.gru(x1)
        _, h2 = self.gru(x2)
        x = torch.cat([h1.squeeze(0), h2.squeeze(0)], dim=1)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = torch.sigmoid(self.fc2(x))
        return x

class CNNClassifier(nn.Module):
    def __init__(self, input_dim):
        super(CNNClassifier, self).__init__()
        self.conv1 = nn.Conv1d(1, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool1d(2, stride=2)
        pooled_size = input_dim // 2
        self.fc1 = nn.Linear(pooled_size * 64 * 2, 256)
        self.fc2 = nn.Linear(256, 64)
        self.fc3 = nn.Linear(64, 1)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)

    def forward(self, x1, x2):
        if x1.dim() > 2:
            x1 = x1.reshape(x1.size(0), -1)
        if x2.dim() > 2:
            x2 = x2.reshape(x2.size(0), -1)

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
        self.fc1 = nn.Linear(256 * 2, 64)
        self.fc2 = nn.Linear(64, 1)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)

    def forward(self, x1, x2):
        if x1.dim() > 2:
            x1 = x1.reshape(x1.size(0), -1)
        if x2.dim() > 2:
            x2 = x2.reshape(x2.size(0), -1)
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
        self.fc1 = nn.Linear(256 * 2, 64)  # 128*2方向*2个输入
        self.fc2 = nn.Linear(64, 1)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)

    def forward(self, x1, x2):
        if x1.dim() > 2:
            x1 = x1.reshape(x1.size(0), -1)
        if x2.dim() > 2:
            x2 = x2.reshape(x2.size(0), -1)

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
        self.fc1 = nn.Linear(256 * 2, 64)
        self.fc2 = nn.Linear(64, 1)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)

    def forward(self, x1, x2):
        if x1.dim() > 2:
            x1 = x1.reshape(x1.size(0), -1)
        if x2.dim() > 2:
            x2 = x2.reshape(x2.size(0), -1)

        x1 = x1.unsqueeze(1)
        x2 = x2.unsqueeze(1)

        _, h1 = self.rnn(x1)
        _, h2 = self.rnn(x2)

        x = torch.cat([h1.squeeze(0), h2.squeeze(0)], dim=1)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = torch.sigmoid(self.fc2(x))
        return x

# 分类器数据集
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

                for pos in sample['positive_embeddings']:
                    pos = pos.to(device)
                    pos_emb, _ = self.contrastive_model(pos)
                    self.code_pairs.append((anchor_emb.cpu(), pos_emb.cpu()))
                    self.labels.append(1)

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
def train_classifier(train_dataset, val_dataset, classifier, epochs=classifyEpoch, 
                    batch_size=32, patience=3, exp_name=""):
    print(f"\nTraining Classifier: {exp_name}")
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    criterion = nn.BCELoss()
    optimizer = optim.Adam(classifier.parameters(), lr=lr2)

    # 记录训练过程
    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []
    
    best_val_loss = float('inf')
    no_improvement_count = 0

    for epoch in range(epochs):
        # 训练阶段
        classifier.train()
        total_loss = 0
        correct = 0
        total = 0
        
        for code1, code2, labels in train_loader:
            code1, code2, labels = code1.to(device), code2.to(device), labels.float().to(device)
            optimizer.zero_grad()
            outputs = classifier(code1, code2).squeeze()
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
            # 计算准确率
            predicted = (outputs > 0.5).float()
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        avg_train_loss = total_loss / len(train_loader)
        train_acc = correct / total
        train_losses.append(avg_train_loss)
        train_accs.append(train_acc)

        # 验证阶段
        classifier.eval()
        total_val_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for code1, code2, labels in val_loader:
                code1, code2, labels = code1.to(device), code2.to(device), labels.float().to(device)
                outputs = classifier(code1, code2).squeeze()
                loss = criterion(outputs, labels)
                total_val_loss += loss.item()
                
                predicted = (outputs > 0.5).float()
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        avg_val_loss = total_val_loss / len(val_loader)
        val_acc = correct / total
        val_losses.append(avg_val_loss)
        val_accs.append(val_acc)

        print(f"Epoch {epoch + 1:03d}/{epochs} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")
        print(f"         Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            no_improvement_count = 0
        else:
            no_improvement_count += 1
            if no_improvement_count >= patience:
                print(f"\nEarly stopping triggered at epoch {epoch + 1}")
                break

    # 绘制训练曲线 - 训练loss+acc一张图，验证loss+acc一张图
    if epochs > 1 and len(train_losses) > 1:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        epochs_range = range(1, len(train_losses) + 1)
        
        # 第一张图：训练损失 + 训练准确率（双Y轴）
        ax1_twin = ax1.twinx()
        
        # 训练损失（左Y轴）
        line1 = ax1.plot(epochs_range, train_losses, 'b-', label='Train Loss', linewidth=2, marker='o')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss', color='b')
        ax1.tick_params(axis='y', labelcolor='b')
        ax1.grid(True, alpha=0.3)
        
        # 训练准确率（右Y轴）
        line2 = ax1_twin.plot(epochs_range, train_accs, 'r-', label='Train Accuracy', linewidth=2, marker='s')
        ax1_twin.set_ylabel('Accuracy', color='r')
        ax1_twin.tick_params(axis='y', labelcolor='r')
        
        # 设置标题和图例
        ax1.set_title(f'Training Metrics - {exp_name}')
        lines1 = line1 + line2
        labels1 = [l.get_label() for l in lines1]
        ax1.legend(lines1, labels1, loc='upper left')
        
        # 第二张图：验证损失 + 验证准确率（双Y轴）
        ax2_twin = ax2.twinx()
        
        # 验证损失（左Y轴）
        line3 = ax2.plot(epochs_range, val_losses, 'b-', label='Validation Loss', linewidth=2, marker='o')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Loss', color='b')
        ax2.tick_params(axis='y', labelcolor='b')
        ax2.grid(True, alpha=0.3)
        
        # 验证准确率（右Y轴）
        line4 = ax2_twin.plot(epochs_range, val_accs, 'r-', label='Validation Accuracy', linewidth=2, marker='s')
        ax2_twin.set_ylabel('Accuracy', color='r')
        ax2_twin.tick_params(axis='y', labelcolor='r')
        
        # 设置标题和图例
        ax2.set_title(f'Validation Metrics - {exp_name}')
        lines2 = line3 + line4
        labels2 = [l.get_label() for l in lines2]
        ax2.legend(lines2, labels2, loc='upper left')
        
        plt.tight_layout()
        safe_exp_name = exp_name.replace('/', '_').replace('\\', '_').replace(':', '_')
        plt.savefig(f"classifier_training_curves_{safe_exp_name}.png", dpi=300, bbox_inches='tight')
        plt.close()
        print(f"分类器训练曲线已保存: classifier_training_curves_{safe_exp_name}.png")

    return best_val_loss

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
    
    # 添加异常处理，防止precision/recall/f1计算错误
    try:
        f1 = f1_score(all_labels, all_preds, zero_division=0)
        recall = recall_score(all_labels, all_preds, zero_division=0)
        precision = precision_score(all_labels, all_preds, zero_division=0)
    except Exception as e:
        print(f"Warning: 指标计算出错 - {e}")
        f1 = 0.0
        recall = 0.0
        precision = 0.0
    
    try:
        auc_score = roc_auc_score(all_labels, all_probs)
    except Exception as e:
        print(f"Warning: AUC计算出错 - {e}")
        auc_score = 0.5  # 随机分类器的AUC

    return acc, f1, recall, precision, auc_score

# 主实验循环
all_results = {}
best_performance = 0
best_config = None

print("开始编码器、分类器和语言组合的全面比较...")

# 确保所有编码器都被处理
successful_encoders = set()

for encoder_name, encoder_path in model_path_dict.items():
    print(f"\n{'='*80}")
    print(f"Testing Encoder: {encoder_name}")
    print(f"{'='*80}")
    
    encoder_results = {}
    encoder_successful = False
    
    for lang_pair, lang_path in langs_path_dict.items():
        print(f"\n{'='*60}")
        print(f"Testing: {encoder_name} on {lang_pair}")
        print(f"{'='*60}")
        
        try:
            # 检查文件是否存在
            if not os.path.exists(lang_path):
                print(f"Warning: 数据文件不存在: {lang_path}")
                continue
                
            # 加载编码器和tokenizer
            tokenizer = AutoTokenizer.from_pretrained(encoder_path)
            model = AutoModel.from_pretrained(encoder_path).to(device)
            
            # 加载数据
            samples = load_samples_from_jsonl(lang_path)
            if len(samples) == 0:
                print(f"Warning: 没有加载到数据: {lang_path}")
                continue
                
            encoded_samples = [encode_sample(sample, tokenizer, model, device) for sample in samples]
            
            # 分割数据
            train_samples, remaining_samples = train_test_split(encoded_samples, test_size=0.4, random_state=42)
            val_samples, test_samples = train_test_split(remaining_samples, test_size=0.5, random_state=42)
            
            # 训练ConbaLayer对比学习模型
            contrastive_model = SimCLRWithLayer(ConbaLayer).to(device)
            train_losses, val_losses, best_contrastive_loss = train_contrastive_model(
                train_samples, val_samples, contrastive_model,
                exp_name=f"{encoder_name}-{lang_pair}"
            )
            
            # 创建分类器数据集
            train_classifier_dataset = CloneClassifierDataset(train_samples, contrastive_model)
            val_classifier_dataset = CloneClassifierDataset(val_samples, contrastive_model)
            test_classifier_dataset = CloneClassifierDataset(test_samples, contrastive_model)

            # 测试所有6个分类器（移除ResNet）
            classifier_models = {
                "MLP": MLPClassifier(768).to(device),
                "GRU": GRUClassifier(768).to(device),
                "CNN": CNNClassifier(768).to(device),
                "LSTM": LSTMClassifier(768).to(device),
                "BiLSTM": BiLSTMClassifier(768).to(device),
                "RNN": RNNClassifier(768).to(device)
                # 移除ResNet因为预测输出有问题
            }

            config_results = {}
            
            for classifier_name, classifier in classifier_models.items():
                print(f"\nTesting {classifier_name} classifier...")
                
                # 训练分类器
                best_val_loss = train_classifier(
                    train_classifier_dataset, val_classifier_dataset, classifier,
                    epochs=classifyEpoch, batch_size=BatchSize, patience=classifyPatience,
                    exp_name=f"{encoder_name}-{lang_pair}-{classifier_name}"
                )
                
                # 评估分类器
                acc, f1, recall, precision, auc_score = evaluate_classifier(test_classifier_dataset, classifier)
                
                config_results[classifier_name] = {
                    "accuracy": float(acc),
                    "f1_score": float(f1),
                    "recall": float(recall),
                    "precision": float(precision),
                    "auc": float(auc_score),
                    "val_loss": float(best_val_loss)
                }
                
                print(f"{classifier_name} Results: Acc={acc:.4f}, F1={f1:.4f}, Precision={precision:.4f}, Recall={recall:.4f}")
                
                # 更新最佳配置
                current_performance = f1  # 使用F1作为主要评估指标
                if current_performance > best_performance:
                    best_performance = current_performance
                    best_config = {
                        "encoder": encoder_name,
                        "language_pair": lang_pair,
                        "classifier": classifier_name,
                        "performance": config_results[classifier_name]
                    }
            
            encoder_results[lang_pair] = config_results
            encoder_successful = True
            
        except Exception as e:
            print(f"Error with {encoder_name} on {lang_pair}: {str(e)}")
            import traceback
            traceback.print_exc()
            continue
    
    if encoder_successful:
        all_results[encoder_name] = encoder_results
        successful_encoders.add(encoder_name)
        print(f"编码器 {encoder_name} 测试完成")
    else:
        print(f"编码器 {encoder_name} 测试失败")

print(f"\n成功测试的编码器: {list(successful_encoders)}")
print(f"成功测试的编码器数量: {len(successful_encoders)}")

# 保存第一阶段结果
with open("encoder_classifier_comparison_results.json", "w") as f:
    json.dump(all_results, f, indent=4)

print(f"\n{'='*80}")
print("第一阶段完成！最佳配置：")
print(f"编码器: {best_config['encoder']}")
print(f"语言对: {best_config['language_pair']}")
print(f"分类器: {best_config['classifier']}")
print(f"性能: {best_config['performance']}")
print(f"{'='*80}")

# 保存最佳配置
with open("best_baseline_config.json", "w") as f:
    json.dump(best_config, f, indent=4)

print("\n第一阶段结果已保存。准备好后请告诉我继续第二阶段。")

# ==================== 第二部分：消融实验 + 噪声测试 + 温度系数分析 ====================

# 完整的消融实验变体
class ConbaNoControl(nn.Module):
    """移除控制权重"""
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
    """移除反馈权重"""
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
    """移除状态机制"""
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

class ConbaNoSelective(nn.Module):
    """移除选择性机制"""
    def __init__(self, input_dim):
        super(ConbaNoSelective, self).__init__()
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
        adjusted_output = x * self.control_weight + state * self.feedback_weight
        output = self.C(adjusted_output)
        output = self.fc2(output)
        return output, state

class ConbaNoSwish(nn.Module):
    """移除Swish激活函数，使用ReLU"""
    def __init__(self, input_dim):
        super(ConbaNoSwish, self).__init__()
        self.selective_fc = nn.Linear(input_dim, input_dim)
        self.relu = nn.ReLU()
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
        selective_output = self.relu(self.selective_fc(x)) * x
        adjusted_output = selective_output * self.control_weight + state * self.feedback_weight
        output = self.C(adjusted_output)
        output = self.fc2(output)
        return output, state

class ConbaNoProjection(nn.Module):
    """移除最终投影层"""
    def __init__(self, input_dim):
        super(ConbaNoProjection, self).__init__()
        self.selective_fc = nn.Linear(input_dim, input_dim)
        self.swish = nn.SiLU()
        self.control_weight = nn.Parameter(torch.ones(input_dim))
        self.feedback_weight = nn.Parameter(torch.ones(input_dim))
        self.A = nn.Linear(input_dim, input_dim, bias=False)
        self.B = nn.Linear(input_dim, input_dim, bias=False)
        self.C = nn.Linear(input_dim, input_dim, bias=False)

    def forward(self, x, previous_state=None):
        if x.dim() > 2:
            x = x.view(x.size(0), -1)
        if previous_state is None:
            previous_state = torch.zeros_like(x, device=x.device)
        state = self.A(previous_state) + self.B(x)
        selective_output = self.swish(self.selective_fc(x)) * x
        adjusted_output = selective_output * self.control_weight + state * self.feedback_weight
        output = self.C(adjusted_output)
        return output, state

class NoConbaLayer(nn.Module):
    """完全移除ConbaLayer，只使用线性变换"""
    def __init__(self, input_dim):
        super(NoConbaLayer, self).__init__()
        self.fc = nn.Linear(input_dim, input_dim)

    def forward(self, x, previous_state=None):
        if x.dim() > 2:
            x = x.view(x.size(0), -1)
        output = self.fc(x)
        return output, None

# 噪声注入函数
class CodeNoiseInjector:
    def __init__(self):
        # 编程语言关键词和符号
        self.language_keywords = {
            'python': ['def', 'class', 'if', 'else', 'elif', 'for', 'while', 'import', 'from', 'return', 
                      'try', 'except', 'with', 'as', 'lambda', 'yield', 'assert', 'break', 'continue',
                      'and', 'or', 'not', 'in', 'is', 'None', 'True', 'False'],
            'java': ['public', 'private', 'protected', 'class', 'interface', 'extends', 'implements',
                    'static', 'final', 'abstract', 'synchronized', 'volatile', 'native', 'strictfp',
                    'if', 'else', 'for', 'while', 'do', 'switch', 'case', 'default', 'break', 'continue',
                    'return', 'try', 'catch', 'finally', 'throw', 'throws', 'new', 'this', 'super',
                    'null', 'true', 'false', 'void', 'int', 'double', 'float', 'long', 'short', 'byte',
                    'char', 'boolean', 'String'],
            'cpp': ['public', 'private', 'protected', 'class', 'struct', 'namespace', 'using',
                   'virtual', 'static', 'const', 'mutable', 'inline', 'explicit', 'operator',
                   'if', 'else', 'for', 'while', 'do', 'switch', 'case', 'default', 'break', 'continue',
                   'return', 'try', 'catch', 'throw', 'new', 'delete', 'this', 'nullptr', 'true', 'false',
                   'void', 'int', 'double', 'float', 'long', 'short', 'char', 'bool', 'auto'],
            'cs': ['public', 'private', 'protected', 'internal', 'class', 'interface', 'struct',
                  'abstract', 'sealed', 'static', 'virtual', 'override', 'readonly', 'const',
                  'if', 'else', 'for', 'while', 'do', 'foreach', 'switch', 'case', 'default',
                  'break', 'continue', 'return', 'try', 'catch', 'finally', 'throw', 'new', 'this',
                  'base', 'null', 'true', 'false', 'void', 'int', 'double', 'float', 'long', 'short',
                  'byte', 'char', 'bool', 'string', 'var', 'using', 'namespace']
        }
        
        # 同义词替换词典
        self.synonyms = {
            'python': {
                'function': 'method', 'method': 'function',
                'variable': 'var', 'var': 'variable',
                'parameter': 'param', 'param': 'parameter',
                'argument': 'arg', 'arg': 'argument',
                'length': 'len', 'len': 'length',
                'string': 'str', 'str': 'string',
                'integer': 'int', 'int': 'integer',
                'list': 'array', 'array': 'list',
                'dictionary': 'dict', 'dict': 'dictionary'
            },
            'java': {
                'method': 'function', 'function': 'method',
                'variable': 'var', 'var': 'variable',
                'parameter': 'param', 'param': 'parameter',
                'argument': 'arg', 'arg': 'argument',
                'length': 'size', 'size': 'length',
                'String': 'str', 'str': 'String',
                'Integer': 'int', 'int': 'Integer',
                'ArrayList': 'List', 'List': 'ArrayList',
                'HashMap': 'Map', 'Map': 'HashMap'
            },
            'cpp': {
                'function': 'method', 'method': 'function',
                'variable': 'var', 'var': 'variable',
                'parameter': 'param', 'param': 'parameter',
                'argument': 'arg', 'arg': 'argument',
                'length': 'size', 'size': 'length',
                'string': 'str', 'str': 'string',
                'integer': 'int', 'int': 'integer',
                'vector': 'array', 'array': 'vector',
                'map': 'dictionary', 'dictionary': 'map'
            },
            'cs': {
                'method': 'function', 'function': 'method',
                'variable': 'var', 'var': 'variable',
                'parameter': 'param', 'param': 'parameter',
                'argument': 'arg', 'arg': 'argument',
                'length': 'size', 'size': 'length',
                'string': 'str', 'str': 'string',
                'integer': 'int', 'int': 'integer',
                'List': 'array', 'array': 'List',
                'Dictionary': 'map', 'map': 'Dictionary'
            }
        }
    
    def detect_language(self, code):
        """检测代码语言"""
        code_lower = code.lower()
        scores = {}
        
        for lang, keywords in self.language_keywords.items():
            score = sum(1 for keyword in keywords if keyword in code_lower)
            scores[lang] = score
        
        return max(scores, key=scores.get) if scores else 'python'
    
    def synonym_replacement(self, code, ratio=0.1):
        """同义词替换"""
        language = self.detect_language(code)
        synonyms = self.synonyms.get(language, {})
        
        if not synonyms:
            return code
        
        words = code.split()
        num_replacements = max(1, int(len(words) * ratio))
        
        # 随机选择要替换的词
        replacement_indices = random.sample(range(len(words)), 
                                          min(num_replacements, len(words)))
        
        for idx in replacement_indices:
            word = words[idx]
            if word in synonyms:
                words[idx] = synonyms[word]
        
        return ' '.join(words)
    
    def random_insertion(self, code, ratio=0.1):
        """随机插入"""
        language = self.detect_language(code)
        keywords = self.language_keywords.get(language, [])
        
        if not keywords:
            return code
        
        words = code.split()
        num_insertions = max(1, int(len(words) * ratio))
        
        for _ in range(num_insertions):
            # 随机选择插入位置和插入的关键词
            insert_pos = random.randint(0, len(words))
            keyword_to_insert = random.choice(keywords)
            words.insert(insert_pos, keyword_to_insert)
        
        return ' '.join(words)
    
    def random_swap(self, code, ratio=0.1):
        """随机交换"""
        words = code.split()
        if len(words) < 2:
            return code
        
        num_swaps = max(1, int(len(words) * ratio))
        
        for _ in range(num_swaps):
            # 随机选择两个位置进行交换
            idx1, idx2 = random.sample(range(len(words)), 2)
            words[idx1], words[idx2] = words[idx2], words[idx1]
        
        return ' '.join(words)
    
    def random_deletion(self, code, ratio=0.1):
        """随机删除"""
        words = code.split()
        if len(words) <= 1:
            return code
        
        num_deletions = max(1, int(len(words) * ratio))
        num_deletions = min(num_deletions, len(words) - 1)  # 至少保留一个词
        
        # 随机选择要删除的词的索引
        deletion_indices = random.sample(range(len(words)), num_deletions)
        deletion_indices.sort(reverse=True)  # 从后往前删除
        
        for idx in deletion_indices:
            del words[idx]
        
        return ' '.join(words)
    
    def inject_noise(self, code, noise_type, noise_ratio=0.1):
        """注入指定类型的噪声"""
        if noise_type == 'synonym':
            return self.synonym_replacement(code, noise_ratio)
        elif noise_type == 'insertion':
            return self.random_insertion(code, noise_ratio)
        elif noise_type == 'swap':
            return self.random_swap(code, noise_ratio)
        elif noise_type == 'deletion':
            return self.random_deletion(code, noise_ratio)
        else:
            return code

# 创建噪声数据集
def create_noisy_dataset(samples, noise_injector, noise_type, noise_ratio=0.1):
    """创建包含噪声的数据集"""
    noisy_samples = []
    
    for sample in samples:
        # 对原始代码添加噪声
        noisy_code = noise_injector.inject_noise(sample['code'], noise_type, noise_ratio)
        
        # 对正样本添加噪声
        noisy_positives = []
        for pos_sample in sample['positive']['code_positive']:
            noisy_pos_code = noise_injector.inject_noise(pos_sample['code'], noise_type, noise_ratio)
            noisy_positives.append({'code': noisy_pos_code})
        
        # 对负样本添加噪声
        noisy_negatives = []
        for neg_sample in sample['negative']['code_negative']:
            noisy_neg_code = noise_injector.inject_noise(neg_sample['code'], noise_type, noise_ratio)
            noisy_negatives.append({'code': noisy_neg_code})
        
        noisy_sample = {
            'code': noisy_code,
            'positive': {'code_positive': noisy_positives},
            'negative': {'code_negative': noisy_negatives}
        }
        noisy_samples.append(noisy_sample)
    
    return noisy_samples

# 温度系数ROC分析
def temperature_roc_analysis(train_samples, val_samples, test_samples, best_encoder_path, 
                           best_classifier_class, temperature_range=[0.1, 0.2, 0.3, 0.5, 0.7, 1.0]):
    """分析不同温度系数对ROC的影响"""
    print(f"\n{'='*60}")
    print("开始温度系数ROC分析...")
    print(f"{'='*60}")
    
    # 加载最佳编码器
    tokenizer = AutoTokenizer.from_pretrained(best_encoder_path)
    model = AutoModel.from_pretrained(best_encoder_path).to(device)
    
    temperature_results = {}
    roc_data = {}
    
    for temp in temperature_range:
        print(f"\n测试温度系数: {temp}")
        
        # 训练对比学习模型 - 移除test_samples参数
        contrastive_model = SimCLRWithLayer(ConbaLayer).to(device)
        losses, val_losses, best_loss = train_contrastive_model(
            train_samples, val_samples, contrastive_model,
            temperature=temp, exp_name=f"Temperature-{temp}"
        )
        
        # 创建分类器数据集
        train_dataset = CloneClassifierDataset(train_samples, contrastive_model)
        val_dataset = CloneClassifierDataset(val_samples, contrastive_model)
        test_dataset = CloneClassifierDataset(test_samples, contrastive_model)
        
        # 训练分类器 - 移除test_dataset参数
        classifier = best_classifier_class(768).to(device)
        train_classifier(train_dataset, val_dataset, classifier, 
                        epochs=10, exp_name=f"Temp-{temp}")
        
        # 评估并获取ROC数据
        acc, f1, recall, precision, auc_score = evaluate_classifier(test_dataset, classifier)
        
        # 获取详细的ROC数据
        test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
        classifier.eval()
        all_labels = []
        all_probs = []
        
        with torch.no_grad():
            for code1, code2, labels in test_loader:
                code1, code2 = code1.to(device), code2.to(device)
                outputs = classifier(code1, code2).squeeze()
                probs = outputs.cpu().numpy()
                all_labels.extend(labels.numpy())
                all_probs.extend(probs)
        
        fpr, tpr, _ = roc_curve(all_labels, all_probs)
        
        temperature_results[temp] = {
        "accuracy": float(acc),
        "f1_score": float(f1),
        "recall": float(recall),
        "precision": float(precision),
        "auc": float(auc_score)
    }

        roc_data[temp] = (fpr, tpr, auc_score)
        
        print(f"温度系数 {temp} 结果: Acc={acc:.4f}, F1={f1:.4f}, AUC={auc_score:.4f}")
    
    return temperature_results, roc_data

# 运行第二阶段实验
print("\n" + "="*80)
print("开始第二阶段实验...")
print("="*80)

# 加载最佳配置
try:
    with open("best_baseline_config.json", "r") as f:
        best_config = json.load(f)
    
    best_encoder = best_config["encoder"]
    best_language_pair = best_config["language_pair"]
    best_classifier_name = best_config["classifier"]
    
    print(f"使用最佳配置:")
    print(f"编码器: {best_encoder}")
    print(f"语言对: {best_language_pair}")
    print(f"分类器: {best_classifier_name}")
    
except FileNotFoundError:
    print("未找到最佳配置文件，使用默认配置...")
    best_encoder = "unixcoder"
    best_language_pair = "java-python"
    best_classifier_name = "MLP"

# 准备最佳配置的数据
best_encoder_path = model_path_dict[best_encoder]
best_lang_path = langs_path_dict[best_language_pair]

# 更新分类器映射
classifier_mapping = {
    "MLP": MLPClassifier,
    "GRU": GRUClassifier,
    "CNN": CNNClassifier,
    "LSTM": LSTMClassifier,
    "BiLSTM": BiLSTMClassifier,
    "RNN": RNNClassifier
    # 移除ResNet
}

# 加载数据
tokenizer = AutoTokenizer.from_pretrained(best_encoder_path)
model = AutoModel.from_pretrained(best_encoder_path).to(device)
samples = load_samples_from_jsonl(best_lang_path)
encoded_samples = [encode_sample(sample, tokenizer, model, device) for sample in samples]

from sklearn.model_selection import train_test_split
train_samples, remaining_samples = train_test_split(encoded_samples, test_size=0.4, random_state=42)
val_samples, test_samples = train_test_split(remaining_samples, test_size=0.5, random_state=42)

# 1. 完整消融实验
print(f"\n{'='*60}")
print("开始完整消融实验...")
print(f"{'='*60}")

ablation_variants = {
    "Full ConbaLayer": ConbaLayer,
    "No Control Weight": ConbaNoControl,
    "No Feedback Weight": ConbaNoFeedback,
    "No State Mechanism": ConbaNoState,
    "No Selective Mechanism": ConbaNoSelective,
    "No Swish Activation": ConbaNoSwish,
    "No Final Projection": ConbaNoProjection,
    "No ConbaLayer": NoConbaLayer
}

ablation_results = {}
ablation_roc_data = {}

for variant_name, variant_class in ablation_variants.items():
    print(f"\n测试消融变体: {variant_name}")
    
    # 训练对比学习模型 - 移除test_samples参数
    contrastive_model = SimCLRWithLayer(variant_class).to(device)
    losses, val_losses, best_loss = train_contrastive_model(
        train_samples, val_samples, contrastive_model,
        exp_name=variant_name
    )

    # 创建分类器数据集
    train_dataset = CloneClassifierDataset(train_samples, contrastive_model)
    val_dataset = CloneClassifierDataset(val_samples, contrastive_model)
    test_dataset = CloneClassifierDataset(test_samples, contrastive_model)
    
    # 训练分类器 - 移除test_dataset参数
    classifier = classifier_mapping[best_classifier_name](768).to(device)
    train_classifier(train_dataset, val_dataset, classifier,
                    epochs=classifyEpoch, exp_name=variant_name)
    
    # 评估
    acc, f1, recall, precision, auc_score = evaluate_classifier(test_dataset, classifier)
    
    ablation_results[variant_name] = {
        "accuracy": float(acc),
        "f1_score": float(f1),
        "recall": float(recall),
        "precision": float(precision),
        "auc": float(auc_score)
    }
    
    # 获取ROC数据
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
    classifier.eval()
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for code1, code2, labels in test_loader:
            code1, code2 = code1.to(device), code2.to(device)
            outputs = classifier(code1, code2).squeeze()
            probs = outputs.cpu().numpy()
            all_labels.extend(labels.numpy())
            all_probs.extend(probs)
    
    fpr, tpr, _ = roc_curve(all_labels, all_probs)
    ablation_roc_data[variant_name] = (fpr, tpr, auc_score)
    
    print(f"{variant_name} 结果: Acc={acc:.4f}, F1={f1:.4f}, AUC={auc_score:.4f}")

print("\n第二阶段实验完成！准备好后请告诉我继续第三部分（噪声测试和可视化）。")

# ==================== 第三部分：噪声测试实验 + 温度系数ROC + 可视化 ====================

# 2. 噪声鲁棒性测试
print(f"\n{'='*60}")
print("开始噪声鲁棒性测试...")
print(f"{'='*60}")

# 初始化噪声注入器
noise_injector = CodeNoiseInjector()
noise_types = ['synonym', 'insertion', 'swap', 'deletion']
noise_ratios = [0.05, 0.1, 0.15, 0.2]

# 噪声测试结果存储
noise_test_results = {}

for noise_type in noise_types:
    print(f"\n{'='*40}")
    print(f"测试噪声类型: {noise_type.upper()}")
    print(f"{'='*40}")
    
    noise_test_results[noise_type] = {}
    
    for noise_ratio in noise_ratios:
        print(f"\n测试噪声比例: {noise_ratio}")
        
        # 创建噪声数据集
        noisy_original_samples = create_noisy_dataset(samples, noise_injector, noise_type, noise_ratio)
        noisy_encoded_samples = [encode_sample(sample, tokenizer, model, device) for sample in noisy_original_samples]
        
        # 使用相同的数据分割保持一致性
        noisy_train_samples, noisy_remaining_samples = train_test_split(noisy_encoded_samples, test_size=0.4, random_state=42)
        noisy_val_samples, noisy_test_samples = train_test_split(noisy_remaining_samples, test_size=0.5, random_state=42)
        
        # 测试有ConbaLayer vs 无ConbaLayer
        models_to_test = {
            "With ConbaLayer": ConbaLayer,
            "Without ConbaLayer": NoConbaLayer
        }
        
        noise_test_results[noise_type][noise_ratio] = {}
        
        for model_name, model_class in models_to_test.items():
            print(f"  测试模型: {model_name}")
            
            # 训练对比学习模型 - 移除test_samples参数
            contrastive_model = SimCLRWithLayer(model_class).to(device)
            losses, val_losses, best_loss = train_contrastive_model(
                noisy_train_samples, noisy_val_samples, contrastive_model,
                epochs=20,  # 减少epoch加速实验
                exp_name=f"{noise_type}-{noise_ratio}-{model_name}"
            )
            
            # 创建分类器数据集
            train_dataset = CloneClassifierDataset(noisy_train_samples, contrastive_model)
            val_dataset = CloneClassifierDataset(noisy_val_samples, contrastive_model)
            test_dataset = CloneClassifierDataset(noisy_test_samples, contrastive_model)
            
            # 训练分类器 - 移除test_dataset参数
            classifier = classifier_mapping[best_classifier_name](768).to(device)
            train_classifier(train_dataset, val_dataset, classifier,
                           epochs=15, exp_name=f"{noise_type}-{noise_ratio}-{model_name}")
            
            # 评估
            acc, f1, recall, precision, auc_score = evaluate_classifier(test_dataset, classifier)
            
            noise_test_results[noise_type][noise_ratio][model_name] = {
        "accuracy": float(acc),
        "f1_score": float(f1),
        "recall": float(recall),
        "precision": float(precision),
        "auc": float(auc_score)
    }

            print(f"    {model_name} 结果: Acc={acc:.4f}, F1={f1:.4f}, AUC={auc_score:.4f}")

# 3. 温度系数ROC分析
print(f"\n{'='*60}")
print("开始温度系数ROC分析...")
print(f"{'='*60}")

temperature_results, temperature_roc_data = temperature_roc_analysis(
    train_samples, val_samples, test_samples, 
    best_encoder_path, classifier_mapping[best_classifier_name]
)

# 4. T-SNE可视化函数
def plot_tsne_visualization(samples, contrastive_model, title, filename, max_samples=20):
    """绘制T-SNE可视化"""
    embeddings = []
    labels = []
    
    # 收集嵌入向量
    sample_count = 0
    for sample in samples:
        if sample_count >= max_samples:
            break
        
        # 原始代码嵌入
        anchor = sample['code_embedding'].to(device)
        if contrastive_model:
            contrastive_model.eval()
            with torch.no_grad():
                anchor_emb, _ = contrastive_model(anchor)
                anchor_emb = anchor_emb.cpu().numpy().flatten()
        else:
            anchor_emb = anchor.cpu().numpy().flatten()
        
        embeddings.append(anchor_emb)
        labels.append('anchor')
        
        # 正样本嵌入
        for pos in sample['positive_embeddings'][:2]:  # 限制数量
            pos = pos.to(device)
            if contrastive_model:
                with torch.no_grad():
                    pos_emb, _ = contrastive_model(pos)
                    pos_emb = pos_emb.cpu().numpy().flatten()
            else:
                pos_emb = pos.cpu().numpy().flatten()
            embeddings.append(pos_emb)
            labels.append('positive')
        
        # 负样本嵌入
        for neg in sample['negative_embeddings'][:2]:  # 限制数量
            neg = neg.to(device)
            if contrastive_model:
                with torch.no_grad():
                    neg_emb, _ = contrastive_model(neg)
                    neg_emb = neg_emb.cpu().numpy().flatten()
            else:
                neg_emb = neg.cpu().numpy().flatten()
            embeddings.append(neg_emb)
            labels.append('negative')
        
        sample_count += 1
    
    # 执行T-SNE
    embeddings = np.array(embeddings)
    tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(embeddings)-1))
    tsne_results = tsne.fit_transform(embeddings)
    
    # 绘图
    plt.figure(figsize=(10, 8))
    colors = {'anchor': 'red', 'positive': 'green', 'negative': 'blue'}
    
    for label in ['anchor', 'positive', 'negative']:
        idx = [i for i, lbl in enumerate(labels) if lbl == label]
        if idx:  # 确保有数据点
            plt.scatter(tsne_results[idx, 0], tsne_results[idx, 1], 
                       c=colors[label], label=label, alpha=0.7, s=50)
    
    plt.legend()
    plt.title(title)
    plt.xlabel('t-SNE 1')
    plt.ylabel('t-SNE 2')
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"T-SNE图已保存: {filename}")

# 5. 绘制所有可视化结果
print(f"\n{'='*60}")
print("生成可视化结果...")
print(f"{'='*60}")

# 5.1 绘制编码器-分类器性能对比
def plot_encoder_classifier_comparison():
    """绘制编码器-分类器性能对比"""
    metrics = ["precision", "recall", "f1_score"]  # 只显示这三个指标
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    for i, metric in enumerate(metrics):
        ax = axes[i]
        
        # 收集数据 - 修改数据收集逻辑
        encoder_data = {}
        
        for encoder_name, lang_results in all_results.items():
            if encoder_name not in encoder_data:
                encoder_data[encoder_name] = []
            
            # 收集该编码器在所有语言对上的最佳分类器结果
            for lang_pair, classifier_results in lang_results.items():
                if classifier_results:  # 确保有结果
                    # 取该语言对上所有分类器的最佳结果
                    best_result = max(classifier_results.values(), key=lambda x: x.get(metric, 0))
                    encoder_data[encoder_name].append(best_result[metric])
        
        # 过滤掉没有数据的编码器
        encoder_data = {k: v for k, v in encoder_data.items() if v}
        
        if not encoder_data:
            print(f"Warning: 没有找到 {metric} 的数据")
            continue
        
        # 计算平均值和标准差
        encoders = list(encoder_data.keys())
        avg_scores = [np.mean(encoder_data[enc]) for enc in encoders]
        std_scores = [np.std(encoder_data[enc]) if len(encoder_data[enc]) > 1 else 0 for enc in encoders]
        
        # 绘制柱状图
        bars = ax.bar(encoders, avg_scores, yerr=std_scores, capsize=5, 
                     color='skyblue', alpha=0.7, edgecolor='navy')
        ax.set_title(f'{metric.replace("_", " ").title()}', fontsize=14, fontweight='bold')
        ax.set_ylabel('Score', fontsize=12)
        ax.tick_params(axis='x', rotation=45, labelsize=10)
        ax.grid(True, alpha=0.3)
        
        # 添加数值标签
        for bar, score, std in zip(bars, avg_scores, std_scores):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + std + 0.005,
                   f'{score:.3f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig("encoder_classifier_performance_comparison.png", dpi=300, bbox_inches='tight')
    plt.close()
    print(f"编码器-分类器性能对比图已保存，显示{len(encoder_data)}个编码器")

# 5.2 绘制消融实验结果
def plot_ablation_results():
    """绘制消融实验结果"""
    metrics = ["precision", "recall", "f1_score"]  # 只显示这三个指标
    
    fig, ax = plt.subplots(figsize=(14, 8))
    
    x = np.arange(len(metrics))
    width = 0.1
    
    variant_names = list(ablation_results.keys())
    colors = plt.cm.Set3(np.linspace(0, 1, len(variant_names)))
    
    for i, (variant_name, color) in enumerate(zip(variant_names, colors)):
        scores = [ablation_results[variant_name][metric] for metric in metrics]
        offset = (i - len(variant_names)/2) * width
        bars = ax.bar(x + offset, scores, width, label=variant_name, color=color, alpha=0.8)
        
        # 添加数值标签
        for bar, score in zip(bars, scores):
            if score > 0.1:  # 只为较高的值添加标签
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.005,
                       f'{score:.3f}', ha='center', va='bottom', fontsize=8, rotation=90)
    
    ax.set_xlabel('Metrics', fontsize=12)
    ax.set_ylabel('Scores', fontsize=12)
    ax.set_title('Ablation Study Results', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([m.replace('_', ' ').title() for m in metrics])
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig("ablation_study_results.png", dpi=300, bbox_inches='tight')
    plt.close()
    print("消融实验结果图已保存（仅显示 Precision、Recall、F1）")

# 5.3 绘制噪声鲁棒性测试结果
def plot_noise_robustness_results():
    """绘制噪声鲁棒性测试结果"""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    for i, noise_type in enumerate(noise_types):
        row = i // 2
        col = i % 2
        ax = axes[row, col]
        
        ratios = noise_ratios
        with_conba_f1 = []
        without_conba_f1 = []
        
        for ratio in ratios:
            with_score = noise_test_results[noise_type][ratio]["With ConbaLayer"]["f1_score"]
            without_score = noise_test_results[noise_type][ratio]["Without ConbaLayer"]["f1_score"]
            with_conba_f1.append(with_score)
            without_conba_f1.append(without_score)
        
        x = np.arange(len(ratios))
        width = 0.35
        
        bars1 = ax.bar(x - width/2, with_conba_f1, width, label='With ConbaLayer', color='skyblue', alpha=0.8)
        bars2 = ax.bar(x + width/2, without_conba_f1, width, label='Without ConbaLayer', color='lightcoral', alpha=0.8)
        
        # 添加数值标签
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.005,
                       f'{height:.3f}', ha='center', va='bottom', fontsize=9)
        
        ax.set_title(f'Noise Type: {noise_type.title()}')
        ax.set_xlabel('Noise Ratio')
        ax.set_ylabel('F1 Score')
        ax.set_xticks(x)
        ax.set_xticklabels([f'{r:.2f}' for r in ratios])
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig("noise_robustness_results.png", dpi=300, bbox_inches='tight')
    plt.close()
    print("噪声鲁棒性测试结果图已保存")

# 5.4 绘制温度系数ROC曲线
def plot_temperature_roc_curves():
    """绘制不同温度系数的ROC曲线"""
    plt.figure(figsize=(12, 8))
    
    colors = plt.cm.viridis(np.linspace(0, 1, len(temperature_roc_data)))
    
    for (temp, (fpr, tpr, auc_score)), color in zip(temperature_roc_data.items(), colors):
        plt.plot(fpr, tpr, color=color, linewidth=2, 
                label=f'Temperature {temp} (AUC = {auc_score:.3f})')
    
    plt.plot([0, 1], [0, 1], 'k--', linewidth=1)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves for Different Temperature Values')
    plt.legend(loc='lower right')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("temperature_roc_curves.png", dpi=300, bbox_inches='tight')
    plt.close()
    print("温度系数ROC曲线图已保存")

# 5.5 绘制消融实验ROC曲线
def plot_ablation_roc_curves():
    """绘制消融实验ROC曲线"""
    plt.figure(figsize=(12, 8))
    
    colors = plt.cm.Set1(np.linspace(0, 1, len(ablation_roc_data)))
    
    for (variant_name, (fpr, tpr, auc_score)), color in zip(ablation_roc_data.items(), colors):
        plt.plot(fpr, tpr, color=color, linewidth=2, 
                label=f'{variant_name} (AUC = {auc_score:.3f})')
    
    plt.plot([0, 1], [0, 1], 'k--', linewidth=1)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves for Ablation Study')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("ablation_roc_curves.png", dpi=300, bbox_inches='tight')
    plt.close()
    print("消融实验ROC曲线图已保存")

# 生成所有可视化
plot_encoder_classifier_comparison()
plot_ablation_results()
plot_noise_robustness_results()
plot_temperature_roc_curves()
plot_ablation_roc_curves()

# 5.6 生成T-SNE可视化
print("\n生成T-SNE可视化...")

# 原始嵌入的T-SNE
plot_tsne_visualization(test_samples, None, "Original Embeddings (No ConbaLayer)", "original_embeddings_tsne.png")

# 完整ConbaLayer的T-SNE
full_conba_model = SimCLRWithLayer(ConbaLayer).to(device)
train_contrastive_model(train_samples, val_samples, full_conba_model, epochs=10, exp_name="TSNE-Full-ConbaLayer")
plot_tsne_visualization(test_samples, full_conba_model, "Full ConbaLayer Embeddings", "full_conba_embeddings_tsne.png")

# 无ConbaLayer的T-SNE
no_conba_model = SimCLRWithLayer(NoConbaLayer).to(device)
train_contrastive_model(train_samples, val_samples, no_conba_model, epochs=10, exp_name="TSNE-No-ConbaLayer")
plot_tsne_visualization(test_samples, no_conba_model, "No ConbaLayer Embeddings", "no_conba_embeddings_tsne.png")

# 6. 保存所有实验结果
print(f"\n{'='*60}")
print("保存实验结果...")
print(f"{'='*60}")

# 汇总所有结果
final_results = {
    "best_baseline_config": best_config,
    "encoder_classifier_comparison": all_results,
    "ablation_study": ablation_results,
    "noise_robustness_test": noise_test_results,
    "temperature_analysis": temperature_results,
    "experiment_summary": {
        "total_encoders_tested": len(model_path_dict),
        "total_language_pairs_tested": len(langs_path_dict),
        "total_classifiers_tested": len(classifier_mapping),
        "noise_types_tested": noise_types,
        "noise_ratios_tested": noise_ratios,
        "temperature_values_tested": list(temperature_results.keys())
    }
}

# 保存完整结果
with open("complete_experiment_results.json", "w") as f:
    json.dump(final_results, f, indent=4)

# 生成实验报告
def generate_experiment_report():
    """生成实验报告"""
    report = []
    report.append("=" * 80)
    report.append("ConbaLayer 代码克隆检测实验完整报告")
    report.append("=" * 80)
    
    # 最佳基准配置
    report.append(f"\n1. 最佳基准配置:")
    report.append(f"   编码器: {best_config['encoder']}")
    report.append(f"   语言对: {best_config['language_pair']}")
    report.append(f"   分类器: {best_config['classifier']}")
    report.append(f"   性能指标:")
    for metric, value in best_config['performance'].items():
        report.append(f"     {metric}: {value:.4f}")
    
    # 消融实验结果
    report.append(f"\n2. 消融实验结果 (按F1分数排序):")
    sorted_ablation = sorted(ablation_results.items(), key=lambda x: x[1]['f1_score'], reverse=True)
    for variant_name, results in sorted_ablation:
        report.append(f"   {variant_name}:")
        report.append(f"     F1: {results['f1_score']:.4f}, AUC: {results['auc']:.4f}")
    
    # 噪声鲁棒性结果
    report.append(f"\n3. 噪声鲁棒性测试 (ConbaLayer vs No ConbaLayer):")
    for noise_type in noise_types:
        report.append(f"   {noise_type.title()} 噪声:")
        for ratio in noise_ratios:
            with_conba = noise_test_results[noise_type][ratio]["With ConbaLayer"]["f1_score"]
            without_conba = noise_test_results[noise_type][ratio]["Without ConbaLayer"]["f1_score"]
            improvement = with_conba - without_conba
            report.append(f"     比例 {ratio:.2f}: With={with_conba:.4f}, Without={without_conba:.4f}, 提升={improvement:.4f}")
    
    # 温度系数分析
    report.append(f"\n4. 温度系数分析结果:")
    sorted_temp = sorted(temperature_results.items(), key=lambda x: x[1]['f1_score'], reverse=True)
    for temp, results in sorted_temp:
        report.append(f"   温度 {temp}: F1={results['f1_score']:.4f}, AUC={results['auc']:.4f}")
    
    # 关键发现
    report.append(f"\n5. 关键发现:")
    
    # 最佳消融变体
    best_ablation = max(ablation_results.items(), key=lambda x: x[1]['f1_score'])
    worst_ablation = min(ablation_results.items(), key=lambda x: x[1]['f1_score'])
    report.append(f"   - 最佳配置: {best_ablation[0]} (F1: {best_ablation[1]['f1_score']:.4f})")
    report.append(f"   - 最差配置: {worst_ablation[0]} (F1: {worst_ablation[1]['f1_score']:.4f})")
    
    # 噪声鲁棒性平均提升
    total_improvements = []
    for noise_type in noise_types:
        for ratio in noise_ratios:
            with_conba = noise_test_results[noise_type][ratio]["With ConbaLayer"]["f1_score"]
            without_conba = noise_test_results[noise_type][ratio]["Without ConbaLayer"]["f1_score"]
            total_improvements.append(with_conba - without_conba)
    
    avg_improvement = np.mean(total_improvements)
    report.append(f"   - ConbaLayer平均噪声鲁棒性提升: {avg_improvement:.4f}")
    
    # 最佳温度系数
    best_temp = max(temperature_results.items(), key=lambda x: x[1]['f1_score'])
    report.append(f"   - 最佳温度系数: {best_temp[0]} (F1: {best_temp[1]['f1_score']:.4f})")
    
    report.append(f"\n6. 实验统计:")
    report.append(f"   - 总共测试编码器数量: {len(model_path_dict)}")
    report.append(f"   - 总共测试语言对数量: {len(langs_path_dict)}")
    report.append(f"   - 总共测试分类器数量: 6")  # 更新为6个
    report.append(f"   - 消融实验变体数量: {len(ablation_variants)}")
    report.append(f"   - 噪声类型数量: {len(noise_types)}")
    report.append(f"   - 温度系数测试数量: {len(temperature_results)}")
    
    report.append("\n" + "=" * 80)
    
    return "\n".join(report)

# 生成并保存报告
experiment_report = generate_experiment_report()
with open("experiment_report.txt", "w", encoding='utf-8') as f:
    f.write(experiment_report)

print("\n" + "=" * 80)
print("实验完成！")
print("=" * 80)
print("\n生成的文件:")
print("- complete_experiment_results.json: 完整实验结果")
print("- experiment_report.txt: 实验报告")
print("- encoder_classifier_performance_comparison.png: 编码器-分类器性能对比")
print("- ablation_study_results.png: 消融实验结果")
print("- noise_robustness_results.png: 噪声鲁棒性测试结果")
print("- temperature_roc_curves.png: 温度系数ROC曲线")
print("- ablation_roc_curves.png: 消融实验ROC曲线")
print("- original_embeddings_tsne.png: 原始嵌入T-SNE可视化")
print("- full_conba_embeddings_tsne.png: 完整ConbaLayer嵌入T-SNE可视化")
print("- no_conba_embeddings_tsne.png: 无ConbaLayer嵌入T-SNE可视化")

print(f"\n实验报告预览:")
print("=" * 50)
print(experiment_report)
print("=" * 50)

print(f"\n所有实验已完成！共生成了:")
print(f"- {len(model_path_dict) * len(langs_path_dict) * len(classifier_mapping)} 个编码器-语言对-分类器组合测试")
print(f"- {len(ablation_variants)} 个消融实验变体测试")
print(f"- {len(noise_types) * len(noise_ratios) * 2} 个噪声鲁棒性测试")
print(f"- {len(temperature_results)} 个温度系数测试")
print(f"- 10+ 个可视化图表和报告文件")