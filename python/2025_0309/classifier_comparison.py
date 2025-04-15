import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, recall_score, precision_score
import numpy as np
from classifiers import *
import json
from datetime import datetime
from config import Config
import logging

def run_classifier_comparison(train_samples, val_samples, test_samples):
    """运行分类器比较实验"""
    classifiers = {
        'CNN': CNNClassifier(),
        'BiLSTM': BiLSTMClassifier()
    }
    
    results = {}
    for name, model in classifiers.items():
        logging.info(f"\nTraining {name} classifier...")
        model = model.to(Config.DEVICE)
        
        # Training
        metrics = train_model(model, train_samples, val_samples, name)
        
        # Testing
        test_metrics = evaluate_model(model, test_samples)
        results[name] = {
            'training_metrics': metrics,
            'test_metrics': test_metrics
        }
    
    return results

def train_model(model, train_samples, val_samples, name, epochs=Config.EPOCHS, batch_size=Config.BATCH_SIZE, patience=Config.PATIENCE):
    """训练分类器模型"""
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
            pos_output, _ = model(x1.to(Config.DEVICE), x2.to(Config.DEVICE))
            pos_labels = torch.ones(x2.size(0)).long().to(Config.DEVICE)
            loss_pos = criterion(pos_output, pos_labels)
            
            # 负样本对
            neg_output, _ = model(x1.to(Config.DEVICE), x3.to(Config.DEVICE))
            neg_labels = torch.zeros(x3.size(0)).long().to(Config.DEVICE)
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
                pos_output, _ = model(x1.to(Config.DEVICE), x2.to(Config.DEVICE))
                pos_labels = torch.ones(x2.size(0)).long().to(Config.DEVICE)
                loss_pos = criterion(pos_output, pos_labels)
                
                neg_output, _ = model(x1.to(Config.DEVICE), x3.to(Config.DEVICE))
                neg_labels = torch.zeros(x3.size(0)).long().to(Config.DEVICE)
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
    """评估分类器模型"""
    model.eval()
    test_loader = DataLoader(test_samples, batch_size=32)
    
    all_preds = []
    all_labels = []
    all_features = []
    
    with torch.no_grad():
        for x1, x2, x3 in test_loader:
            # 正样本对
            pos_output, pos_features = model(x1.to(Config.DEVICE), x2.to(Config.DEVICE))
            pos_labels = torch.ones(x2.size(0)).long().to(Config.DEVICE)
            
            # 负样本对
            neg_output, neg_features = model(x1.to(Config.DEVICE), x3.to(Config.DEVICE))
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
    
    # 计算各项指标
    metrics = {
        'accuracy': accuracy_score(all_labels, all_preds),
        'f1': f1_score(all_labels, all_preds),
        'recall': recall_score(all_labels, all_preds),
        'precision': precision_score(all_labels, all_preds)
    }
    
    return {
        'test_metrics': metrics,
        'features': np.array(all_features),
        'labels': np.array(all_labels)
    }

def plot_classifier_results(results):
    # Set style
    plt.style.use('seaborn-darkgrid')
    
    # Create figure with multiple subplots
    fig = plt.figure(figsize=(20, 15))
    gs = fig.add_gridspec(2, 2)
    
    # 1. Training curves
    ax1 = fig.add_subplot(gs[0, 0])
    for name, data in results.items():
        ax1.plot(data['training_metrics']['train_acc'], 
                label=f'{name}', linewidth=2)
    ax1.set_title('Training Accuracy', fontsize=12)
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Accuracy')
    ax1.legend()
    
    # 2. Test metrics comparison
    ax2 = fig.add_subplot(gs[0, 1])
    metrics = ['accuracy', 'f1', 'recall', 'precision']
    plot_metrics_comparison(ax2, results, metrics)
    
    # 3. Confusion matrices
    ax3 = fig.add_subplot(gs[1, :])
    plot_confusion_matrices(ax3, results)
    
    plt.tight_layout()
    plt.savefig('2025_0309/results/figures/classifier_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()

def plot_metrics_comparison(ax, results, metrics):
    data = []
    for name in results.keys():
        for metric in metrics:
            data.append({
                'Classifier': name,
                'Metric': metric,
                'Value': results[name]['test_metrics'][metric]
            })
    
    df = pd.DataFrame(data)
    sns.barplot(x='Metric', y='Value', hue='Classifier', data=df, ax=ax)
    ax.set_title('Test Metrics Comparison')
    ax.set_ylim(0, 1)

def plot_confusion_matrices(ax, results):
    n_classifiers = len(results)
    for i, (name, data) in enumerate(results.items()):
        cm = data['test_metrics']['confusion_matrix']
        plt.subplot(2, n_classifiers//2, i+1)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title(f'{name} Confusion Matrix')

if __name__ == "__main__":
    run_classifier_comparison(train_samples, val_samples, test_samples) 