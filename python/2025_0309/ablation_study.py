import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score
import json
from datetime import datetime
from torch.utils.data import DataLoader
from sklearn.manifold import TSNE
from config import Config
from utils import save_model

# 使用Config中的设备配置
device = Config.DEVICE

class ConbaLayer(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, input_dim)
        self.selective_fc = nn.Linear(input_dim, input_dim)
        self.swish = nn.SiLU()
        
        # 控制参数
        self.control_weight = nn.Parameter(torch.ones(input_dim))
        self.feedback_weight = nn.Parameter(torch.ones(input_dim))
        
        # 状态空间参数
        self.A = nn.Linear(input_dim, input_dim, bias=False)
        self.B = nn.Linear(input_dim, input_dim, bias=False)
        self.C = nn.Linear(input_dim, input_dim, bias=False)
        self.fc2 = nn.Linear(input_dim, input_dim)
        
    def forward(self, x, previous_state=None, use_feedback=True, use_control=True):
        if previous_state is None:
            previous_state = torch.zeros_like(x)
        
        # 状态更新
        state = self.A(previous_state) + self.B(x)
        
        # 选择性激活和反馈控制
        selective_output = self.swish(self.selective_fc(x)) * x
        if use_control and use_feedback:
            adjusted_output = selective_output * self.control_weight + state * self.feedback_weight
        elif use_control:
            adjusted_output = selective_output * self.control_weight
        elif use_feedback:
            adjusted_output = selective_output + state * self.feedback_weight
        else:
            adjusted_output = selective_output
        
        # 输出层
        output = self.C(adjusted_output)
        output = self.fc2(output)
        return output, state

# 创建实验结果目录
def create_directories():
    dirs = ['results', 'results/figures', 'results/metrics', 'classifiers', 'models', 'logs']
    for dir in dirs:
        os.makedirs(f'2025_0309/{dir}', exist_ok=True)

# 消融实验配置
ablation_configs = {
    'full_model': {
        'use_conba': True,
        'use_feedback': True,
        'use_control': True
    },
    'no_conba': {
        'use_conba': False,
        'use_feedback': True,
        'use_control': True
    },
    'no_feedback': {
        'use_conba': True,
        'use_feedback': False,
        'use_control': True
    },
    'no_control': {
        'use_conba': True,
        'use_feedback': True,
        'use_control': False
    }
}

class AblationModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.input_dim = Config.INPUT_DIM
        
        if config['use_conba']:
            self.conba = ConbaLayer(self.input_dim)
        self.projection = nn.Linear(self.input_dim, self.input_dim)
        
    def forward(self, x1, x2):
        # 处理两个输入
        if self.config['use_conba']:
            # 保存ConbaLayer前的特征
            features_before = x1.clone()
            
            # 通过ConbaLayer
            x1_out, state1 = self.conba(x1, 
                                      use_feedback=self.config['use_feedback'],
                                      use_control=self.config['use_control'])
            x2_out, state2 = self.conba(x2,
                                      use_feedback=self.config['use_feedback'],
                                      use_control=self.config['use_control'])
            
            # 计算相似度
            output = torch.cosine_similarity(x1_out, x2_out, dim=1)
            output = torch.stack([1 - output, output], dim=1)  # [batch_size, 2]
            
            return output, x1_out, features_before
        else:
            # 不使用ConbaLayer时
            x1_out = self.projection(x1)
            x2_out = self.projection(x2)
            output = torch.cosine_similarity(x1_out, x2_out, dim=1)
            output = torch.stack([1 - output, output], dim=1)
            
            return output, x1_out, None

def run_ablation_study(train_samples, val_samples, test_samples):
    results = {}
    
    for name, config in ablation_configs.items():
        print(f"\nRunning ablation experiment: {name}")
        model = AblationModel(config).to(device)
        
        # Training
        train_metrics = train_model(model, train_samples, val_samples, name)
        
        # Testing
        test_results = evaluate_model(model, test_samples)
        
        results[name] = {
            'training_metrics': train_metrics,
            'test_metrics': test_results['metrics'],  # 只取 metrics 部分
            'features': test_results['features'],
            'features_before_conba': test_results['features_before_conba'],
            'labels': test_results['labels']
        }
    
    # Save results
    save_results(results)
    plot_ablation_results(results)
    plot_training_curves(results)
    save_detailed_results(results)
    
    return results

def plot_ablation_results(results):
    plt.style.use('seaborn')
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot training curves
    for name, data in results.items():
        ax1.plot(data['training_metrics']['train_loss'], label=f'{name} (train)')
        ax1.plot(data['training_metrics']['val_loss'], '--', label=f'{name} (val)')
    
    ax1.set_title('Training and Validation Loss')
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True)
    
    # Plot accuracy comparison
    names = list(results.keys())
    accuracies = [results[name]['test_metrics']['accuracy'] for name in names]
    losses = [results[name]['test_metrics']['test_loss'] for name in names]
    
    x = np.arange(len(names))
    width = 0.35
    
    ax2.bar(x - width/2, accuracies, width, label='Accuracy')
    ax2.bar(x + width/2, losses, width, label='Test Loss')
    
    ax2.set_title('Test Performance Comparison')
    ax2.set_xticks(x)
    ax2.set_xticklabels(names, rotation=45)
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig('2025_0309/results/figures/ablation_results.png')
    plt.close()

def save_results(results):
    """保存实验结果，将numpy数组转换为列表"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 创建可序列化的结果副本
    serializable_results = {}
    for name, data in results.items():
        serializable_results[name] = {
            'training_metrics': {
                'train_loss': [float(x) for x in data['training_metrics']['train_loss']],
                'val_loss': [float(x) for x in data['training_metrics']['val_loss']],
                'train_acc': [float(x) for x in data['training_metrics']['train_acc']],
                'val_acc': [float(x) for x in data['training_metrics']['val_acc']]
            },
            'test_metrics': {
                metric: float(value)
                for metric, value in data['test_metrics'].items()
            }
        }
        
        # 如果需要保存特征，将其转换为列表
        if 'features' in data:
            serializable_results[name]['features'] = data['features'].tolist()
        if 'features_before_conba' in data and data['features_before_conba'] is not None:
            serializable_results[name]['features_before_conba'] = data['features_before_conba'].tolist()
        if 'labels' in data:
            serializable_results[name]['labels'] = data['labels'].tolist()
    
    with open(f'2025_0309/results/metrics/ablation_results_{timestamp}.json', 'w') as f:
        json.dump(serializable_results, f, indent=4)

def train_model(model, train_samples, val_samples, name, epochs=Config.EPOCHS, batch_size=Config.BATCH_SIZE, patience=Config.PATIENCE):
    optimizer = torch.optim.Adam(model.parameters(), lr=Config.LEARNING_RATE)
    criterion = nn.CrossEntropyLoss()
    best_val_loss = float('inf')
    no_improve = 0
    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []
    
    train_loader = DataLoader(train_samples, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_samples, batch_size=batch_size)
    
    print(f"\n{'='*20} Training {name} {'='*20}")
    for epoch in range(epochs):
        # Training
        model.train()
        epoch_loss = 0
        correct = 0
        total = 0
        for x1, x2, x3 in train_loader:
            optimizer.zero_grad()
            
            # 正样本对
            pos_output, _, _ = model(x1.to(device), x2.to(device))
            pos_labels = torch.ones(x2.size(0)).long().to(device)
            loss_pos = criterion(pos_output, pos_labels)
            
            # 负样本对
            neg_output, _, _ = model(x1.to(device), x3.to(device))
            neg_labels = torch.zeros(x3.size(0)).long().to(device)
            loss_neg = criterion(neg_output, neg_labels)
            
            loss = (loss_pos + loss_neg) / 2
            loss.backward()
            optimizer.step()
            
            # 计算准确率
            pred_pos = torch.argmax(pos_output, dim=1)
            pred_neg = torch.argmax(neg_output, dim=1)
            correct += (pred_pos == pos_labels).sum().item()
            correct += (pred_neg == neg_labels).sum().item()
            total += pos_labels.size(0) + neg_labels.size(0)
            
            epoch_loss += loss.item()
            
        train_loss = epoch_loss / len(train_loader)
        train_acc = correct / total
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        
        # Validation
        model.eval()
        val_loss = 0
        correct = 0
        total = 0
        with torch.no_grad():
            for x1, x2, x3 in val_loader:
                pos_output, _, _ = model(x1.to(device), x2.to(device))
                pos_labels = torch.ones(x2.size(0)).long().to(device)
                loss_pos = criterion(pos_output, pos_labels)
                
                neg_output, _, _ = model(x1.to(device), x3.to(device))
                neg_labels = torch.zeros(x3.size(0)).long().to(device)
                loss_neg = criterion(neg_output, neg_labels)
                
                # 计算准确率
                pred_pos = torch.argmax(pos_output, dim=1)
                pred_neg = torch.argmax(neg_output, dim=1)
                correct += (pred_pos == pos_labels).sum().item()
                correct += (pred_neg == neg_labels).sum().item()
                total += pos_labels.size(0) + neg_labels.size(0)
                
                val_loss += (loss_pos + loss_neg).item() / 2
                
        val_loss /= len(val_loader)
        val_acc = correct / total
        val_losses.append(val_loss)
        val_accs.append(val_acc)
        
        # 打印训练信息
        print(f"Epoch [{epoch+1}/{epochs}] "
              f"Train Loss: {train_loss:.4f} "
              f"Train Acc: {train_acc:.4f} "
              f"Val Loss: {val_loss:.4f} "
              f"Val Acc: {val_acc:.4f}")
        
        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            no_improve = 0
            if Config.SAVE_BEST_MODEL:
                save_model(model, Config.MODEL_SAVE_PATH, f'best_model_{name}')
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"\nEarly stopping at epoch {epoch+1}")
                break
    
    return {
        'train_loss': train_losses,
        'val_loss': val_losses,
        'train_acc': train_accs,
        'val_acc': val_accs
    }

def evaluate_model(model, test_samples):
    """评估模型性能"""
    model.eval()
    test_loader = DataLoader(test_samples, batch_size=32)
    
    all_preds = []
    all_labels = []
    all_features = []
    all_features_before = []
    
    with torch.no_grad():
        for x1, x2, x3 in test_loader:
            # 正样本对
            pos_output, pos_features, pos_features_before = model(x1.to(Config.DEVICE), x2.to(Config.DEVICE))
            pos_labels = torch.ones(x2.size(0)).long().to(Config.DEVICE)
            
            # 负样本对
            neg_output, neg_features, neg_features_before = model(x1.to(Config.DEVICE), x3.to(Config.DEVICE))
            neg_labels = torch.zeros(x3.size(0)).long().to(Config.DEVICE)
            
            # 收集预测和标签
            pred_pos = torch.argmax(pos_output, dim=1)
            pred_neg = torch.argmax(neg_output, dim=1)
            
            all_preds.extend(pred_pos.cpu().numpy())
            all_preds.extend(pred_neg.cpu().numpy())
            all_labels.extend(pos_labels.cpu().numpy())
            all_labels.extend(neg_labels.cpu().numpy())
            
            # 收集特征用于t-SNE
            all_features.extend(pos_features.cpu().numpy())
            all_features.extend(neg_features.cpu().numpy())
            if pos_features_before is not None:
                all_features_before.extend(pos_features_before.cpu().numpy())
                all_features_before.extend(neg_features_before.cpu().numpy())
    
    # 转换为numpy数组
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    
    # 计算各项指标
    metrics = {
        'accuracy': accuracy_score(all_labels, all_preds),
        'f1': f1_score(all_labels, all_preds),
        'recall': recall_score(all_labels, all_preds),
        'precision': precision_score(all_labels, all_preds)
    }
    
    return {
        'metrics': metrics,
        'features': np.array(all_features),
        'features_before_conba': np.array(all_features_before) if all_features_before else None,
        'labels': np.array(all_labels)
    }

def plot_training_curves(results):
    plt.figure(figsize=(15, 5))
    
    # Loss curves
    plt.subplot(1, 2, 1)
    for name, data in results.items():
        plt.plot(data['training_metrics']['train_loss'], 
                label=f'{name} (train)', alpha=0.7)
        plt.plot(data['training_metrics']['val_loss'], 
                label=f'{name} (val)', linestyle='--', alpha=0.7)
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    # Accuracy curves
    plt.subplot(1, 2, 2)
    for name, data in results.items():
        plt.plot(data['training_metrics']['train_acc'], 
                label=f'{name} (train)', alpha=0.7)
        plt.plot(data['training_metrics']['val_acc'], 
                label=f'{name} (val)', linestyle='--', alpha=0.7)
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('2025_0309/results/figures/ablation_training_curves.png')
    plt.close()

def plot_tsne_visualization(results):
    plt.figure(figsize=(15, 5))
    
    for i, (name, data) in enumerate(results.items(), 1):
        embeddings = data['embeddings']
        labels = data['labels']
        
        tsne = TSNE(n_components=2, random_state=42)
        embeddings_2d = tsne.fit_transform(embeddings)
        
        plt.subplot(1, len(results), i)
        scatter = plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], 
                            c=labels, cmap='viridis', alpha=0.6)
        plt.title(f'{name} Embeddings')
        plt.colorbar(scatter)
    
    plt.tight_layout()
    plt.savefig('2025_0309/results/figures/ablation_tsne.png')
    plt.close()

def save_detailed_results(results):
    """保存详细结果，将numpy数组转换为列表"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 创建可序列化的输出
    output = {
        'timestamp': timestamp,
        'experiment_config': ablation_configs,
        'summary': {
            name: {
                'final_train_loss': float(data['training_metrics']['train_loss'][-1]),
                'final_train_acc': float(data['training_metrics']['train_acc'][-1]),
                'final_val_loss': float(data['training_metrics']['val_loss'][-1]),
                'final_val_acc': float(data['training_metrics']['val_acc'][-1]),
                'test_metrics': {
                    'accuracy': float(data['test_metrics']['accuracy']),
                    'f1': float(data['test_metrics']['f1']),
                    'recall': float(data['test_metrics']['recall']),
                    'precision': float(data['test_metrics']['precision'])
                }
            }
            for name, data in results.items()
        }
    }
    
    with open(f'2025_0309/results/metrics/ablation_detailed_{timestamp}.json', 'w') as f:
        json.dump(output, f, indent=4)

if __name__ == "__main__":
    create_directories() 