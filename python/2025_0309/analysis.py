import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from config import Config
import json
from sklearn.manifold import TSNE

def analyze_results(ablation_results, classifier_results):
    """综合分析实验结果"""
    plt.style.use('seaborn')
    
    # 1. 模型性能对比图
    plot_performance_comparison(ablation_results, classifier_results)
    
    # 2. 训练曲线对比
    plot_training_curves_comparison(ablation_results, classifier_results)
    
    # 3. 消融实验对比
    plot_ablation_comparison(ablation_results)
    
    # 4. 分类器对比
    plot_classifier_comparison(classifier_results)
    
    # 5. t-SNE可视化
    plot_tsne_visualization(ablation_results, classifier_results)
    
    # 6. 生成报告
    generate_analysis_report(ablation_results, classifier_results)

def plot_performance_comparison(ablation_results, classifier_results):
    plt.figure(figsize=(15, 8))
    
    # 准备数据
    all_models = list(ablation_results.keys()) + list(classifier_results.keys())
    accuracies = [r['test_metrics']['accuracy'] for r in ablation_results.values()]
    accuracies.extend([r['test_metrics']['accuracy'] for r in classifier_results.values()])
    
    # 绘制柱状图
    colors = plt.cm.viridis(np.linspace(0, 1, len(all_models)))
    plt.bar(range(len(all_models)), accuracies, color=colors)
    plt.xticks(range(len(all_models)), all_models, rotation=45)
    plt.title('Model Performance Comparison', fontsize=14)
    plt.ylabel('Accuracy', fontsize=12)
    
    # 添加数值标签
    for i, v in enumerate(accuracies):
        plt.text(i, v + 0.01, f'{v:.3f}', ha='center', fontsize=10)
    
    plt.tight_layout()
    plt.savefig('2025_0309/results/figures/performance_comparison.png', dpi=300)
    plt.close()

def plot_training_curves_comparison(ablation_results, classifier_results):
    plt.figure(figsize=(20, 10))
    
    # Loss curves
    plt.subplot(2, 1, 1)
    for name, data in ablation_results.items():
        plt.plot(data['training_metrics']['train_loss'], 
                label=f'{name} (train)', alpha=0.7)
    for name, data in classifier_results.items():
        plt.plot(data['training_metrics']['train_loss'], 
                label=f'{name} (train)', linestyle='--', alpha=0.7)
    plt.title('Training Loss Comparison', fontsize=14)
    plt.xlabel('Epochs', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.legend(fontsize=10)
    
    # Accuracy curves
    plt.subplot(2, 1, 2)
    for name, data in ablation_results.items():
        plt.plot(data['training_metrics']['train_acc'], 
                label=f'{name} (train)', alpha=0.7)
    for name, data in classifier_results.items():
        plt.plot(data['training_metrics']['train_acc'], 
                label=f'{name} (train)', linestyle='--', alpha=0.7)
    plt.title('Training Accuracy Comparison', fontsize=14)
    plt.xlabel('Epochs', fontsize=12)
    plt.ylabel('Accuracy', fontsize=12)
    plt.legend(fontsize=10)
    
    plt.tight_layout()
    plt.savefig('2025_0309/results/figures/training_curves_comparison.png', dpi=300)
    plt.close()

def plot_ablation_comparison(ablation_results):
    """绘制消融实验对比图"""
    plt.figure(figsize=(12, 6))
    
    # 准备数据
    names = list(ablation_results.keys())
    metrics = ['accuracy', 'f1', 'recall', 'precision']
    metric_markers = {'accuracy': 'o', 'f1': 'x', 'recall': 's', 'precision': 'd'}
    x = range(len(names))
    
    # 绘制每个指标的曲线
    for metric in metrics:
        values = [r['test_metrics'][metric] for r in ablation_results.values()]
        plt.plot(x, values, label=metric.capitalize(), 
                marker=metric_markers[metric], linewidth=2)
    
    plt.xticks(x, names, rotation=45, ha='right')
    plt.ylabel('Metrics')
    plt.title('Ablation Study - Combined Metrics', fontsize=14)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig('2025_0309/results/figures/ablation_combined_metrics.png', dpi=300)
    plt.close()

def plot_classifier_comparison(classifier_results):
    """绘制分类器对比图"""
    plt.figure(figsize=(12, 6))
    
    # 准备数据
    names = list(classifier_results.keys())
    metrics = ['accuracy', 'f1', 'recall', 'precision']
    metric_markers = {'accuracy': 'o', 'f1': 'x', 'recall': 's', 'precision': 'd'}
    x = range(len(names))
    
    # 绘制每个指标的曲线
    for metric in metrics:
        values = [r['test_metrics'][metric] for r in classifier_results.values()]
        plt.plot(x, values, label=metric.capitalize(), 
                marker=metric_markers[metric], linewidth=2)
    
    plt.xticks(x, names, rotation=45, ha='right')
    plt.ylabel('Metrics')
    plt.title('Model Comparison - Combined Metrics', fontsize=14)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig('2025_0309/results/figures/classifier_combined_metrics.png', dpi=300)
    plt.close()

def plot_tsne_visualization(ablation_results, classifier_results):
    """绘制t-SNE可视化"""
    plt.figure(figsize=(15, 6))
    
    # 1. ConbaLayer前后的嵌入分布对比
    plt.subplot(1, 2, 1)
    if 'full_model' in ablation_results:
        features = ablation_results['full_model'].get('features_before_conba', [])
        if len(features):
            tsne = TSNE(n_components=2, random_state=42)
            features_2d = tsne.fit_transform(features)
            plt.scatter(features_2d[:, 0], features_2d[:, 1], 
                       c=ablation_results['full_model'].get('labels', []),
                       cmap='viridis', alpha=0.6)
            plt.title('Before ConbaLayer', fontsize=12)
            plt.colorbar()
    
    plt.subplot(1, 2, 2)
    if 'full_model' in ablation_results:
        features = ablation_results['full_model'].get('features', [])
        if len(features):
            tsne = TSNE(n_components=2, random_state=42)
            features_2d = tsne.fit_transform(features)
            plt.scatter(features_2d[:, 0], features_2d[:, 1],
                       c=ablation_results['full_model'].get('labels', []),
                       cmap='viridis', alpha=0.6)
            plt.title('After ConbaLayer', fontsize=12)
            plt.colorbar()
    
    plt.suptitle('Feature Distribution Comparison', fontsize=14)
    plt.tight_layout()
    plt.savefig('2025_0309/results/figures/tsne_visualization.png', dpi=300)
    plt.close()

def generate_analysis_report(ablation_results, classifier_results):
    """生成分析报告"""
    all_models = {**ablation_results, **classifier_results}
    
    # 找出最佳模型
    best_model = max(all_models.items(), 
                    key=lambda x: x[1]['test_metrics']['accuracy'])
    
    report = {
        'experiment_summary': {
            'total_models': len(all_models),
            'best_model': best_model[0],
            'best_accuracy': best_model[1]['test_metrics']['accuracy'],
            'average_accuracy': np.mean([r['test_metrics']['accuracy'] 
                                      for r in all_models.values()]),
            'ablation_findings': {
                'full_model_accuracy': ablation_results['full_model']['test_metrics']['accuracy'],
                'component_impact': {
                    name: results['test_metrics']['accuracy']
                    for name, results in ablation_results.items()
                    if name != 'full_model'
                }
            },
            'classifier_findings': {
                'best_classifier': max(classifier_results.items(),
                                     key=lambda x: x[1]['test_metrics']['accuracy'])[0],
                'classifier_accuracies': {
                    name: results['test_metrics']['accuracy']
                    for name, results in classifier_results.items()
                }
            }
        }
    }
    
    # 保存报告
    with open('2025_0309/results/analysis_report.json', 'w') as f:
        json.dump(report, f, indent=4) 