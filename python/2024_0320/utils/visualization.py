import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
import numpy as np

def plot_training_curves(history, save_path):
    """绘制训练曲线"""
    plt.figure(figsize=(15, 5))
    
    # Loss曲线
    plt.subplot(1, 3, 1)
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Val Loss')
    plt.title('Loss Curves')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    # Metrics曲线
    plt.subplot(1, 3, 2)
    for metric in history['train_metrics'][0].keys():
        train_values = [m[metric] for m in history['train_metrics']]
        val_values = [m[metric] for m in history['val_metrics']]
        plt.plot(train_values, label=f'Train {metric}')
        plt.plot(val_values, label=f'Val {metric}')
    plt.title('Metrics')
    plt.xlabel('Epoch')
    plt.ylabel('Score')
    plt.legend()
    
    # 控制权重热力图
    if 'control_weights' in history:
        plt.subplot(1, 3, 3)
        weights = np.array(history['control_weights'])
        sns.heatmap(weights.T, cmap='viridis')
        plt.title('Control Weights Evolution')
        plt.xlabel('Epoch')
        plt.ylabel('Weight Index')
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def plot_tsne(embeddings, labels, save_path):
    """绘制T-SNE可视化"""
    tsne = TSNE(n_components=2, random_state=42)
    embeddings_2d = tsne.fit_transform(embeddings)
    
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], 
                         c=labels, cmap='viridis')
    plt.colorbar(scatter)
    plt.title('T-SNE Visualization of Embeddings')
    plt.savefig(save_path)
    plt.close()

def plot_attention_weights(weights, save_path):
    """绘制注意力权重"""
    plt.figure(figsize=(12, 4))
    sns.heatmap(weights.reshape(1, -1), cmap='viridis', 
                xticklabels=50, yticklabels=False)
    plt.title('Attention Weights Distribution')
    plt.xlabel('Feature Dimension')
    plt.savefig(save_path)
    plt.close() 