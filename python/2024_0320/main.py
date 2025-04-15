import os
import torch
import torch.nn as nn
import torch.optim as optim
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import yaml
import logging
from torch.utils.data import DataLoader

from models import *
from utils.metrics import calculate_metrics
from utils.visualization import plot_training_curves, plot_tsne, plot_attention_weights
from utils.logger import setup_logger

class ExperimentManager:
    def __init__(self, config_path='config.yaml'):
        # 加载配置
        with open(config_path) as f:
            self.config = yaml.safe_load(f)
            
        # 创建实验目录
        self.exp_dir = Path(f"experiments/{datetime.now().strftime('%Y%m%d_%H%M%S')}")
        self.exp_dir.mkdir(parents=True, exist_ok=True)
        
        # 设置日志
        self.logger = setup_logger(self.exp_dir / 'experiment.log')
        
        # 保存配置
        with open(self.exp_dir / 'config.yaml', 'w') as f:
            yaml.dump(self.config, f)
            
        # 设置设备
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 初始化模型和优化器
        self.initialize_models()
        
    def initialize_models(self):
        self.projection_head = CombinedProjectionHead(
            input_dim=self.config['model']['input_dim']
        ).to(self.device)
        
        # 初始化不同的分类器
        self.classifiers = {
            'resnet': ResNet152D(input_dim=self.config['model']['input_dim']),
            'densenet': DenseNet201(input_dim=self.config['model']['input_dim']),
            'efficientnet': EfficientNetV2(input_dim=self.config['model']['input_dim']),
            'vit': ViT(input_dim=self.config['model']['input_dim']),
            'convnext': ConvNeXt(input_dim=self.config['model']['input_dim'])
        }
        
        for name, model in self.classifiers.items():
            self.classifiers[name] = model.to(self.device)
            
    def train_projection_head(self, train_loader, val_loader):
        """训练投影头"""
        optimizer = optim.AdamW(
            self.projection_head.parameters(), 
            lr=self.config['training']['projection_head_lr']
        )
        
        history = {
            'train_loss': [],
            'val_loss': [],
            'control_weights': []
        }
        
        # Training loop...
        
        # 保存训练曲线
        plot_training_curves(
            history,
            save_path=self.exp_dir / 'projection_head_training.png'
        )
        
        return history
        
    def train_classifier(self, name, train_loader, val_loader):
        """训练分类器"""
        classifier = self.classifiers[name]
        optimizer = optim.AdamW(
            classifier.parameters(),
            lr=self.config['training']['classifier_lr']
        )
        
        history = {
            'train_loss': [],
            'val_loss': [],
            'train_metrics': [],
            'val_metrics': []
        }
        
        # Training loop...
        
        # 保存训练曲线
        plot_training_curves(
            history,
            save_path=self.exp_dir / f'{name}_training.png'
        )
        
        return history

    def run_experiments(self):
        """运行完整的实验流程"""
        # 1. 训练投影头
        proj_history = self.train_projection_head(train_loader, val_loader)
        
        # 2. 训练各个分类器
        classifier_results = {}
        for name in self.classifiers.keys():
            self.logger.info(f"Training {name}...")
            history = self.train_classifier(name, train_loader, val_loader)
            classifier_results[name] = history
            
        # 3. 进行消融实验
        ablation_results = self.run_ablation_study()
        
        # 4. 可视化结果
        self.visualize_results(classifier_results, ablation_results)
        
        # 5. 保存结果
        self.save_results(classifier_results, ablation_results)

    def visualize_results(self, classifier_results, ablation_results):
        """可视化实验结果"""
        # 绘制性能对比图
        plt.figure(figsize=(12, 8))
        sns.set_style("whitegrid")
        # ... 绘图代码 ...
        plt.savefig(self.exp_dir / 'performance_comparison.png')
        
        # 绘制T-SNE图
        plot_tsne(
            embeddings,
            labels,
            save_path=self.exp_dir / 'tsne_visualization.png'
        )
        
        # 绘制注意力权重
        plot_attention_weights(
            self.projection_head.control_weight.detach().cpu(),
            save_path=self.exp_dir / 'attention_weights.png'
        )

    def save_results(self, classifier_results, ablation_results):
        """保存实验结果"""
        results = {
            'classifier_results': classifier_results,
            'ablation_results': ablation_results
        }
        
        # 保存为YAML格式
        with open(self.exp_dir / 'results.yaml', 'w') as f:
            yaml.dump(results, f)
            
        # 生成详细报告
        self.generate_report(results)

    def generate_report(self, results):
        """生成实验报告"""
        with open(self.exp_dir / 'report.md', 'w') as f:
            f.write('# Experiment Report\n\n')
            f.write(f'Date: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}\n\n')
            # ... 写入详细结果 ...

if __name__ == '__main__':
    experiment = ExperimentManager()
    experiment.run_experiments() 