import torch
import torch.nn as nn
from models import *
from utils.metrics import calculate_metrics
import yaml

class AblationStudy:
    def __init__(self, config_path='config.yaml'):
        with open(config_path) as f:
            self.config = yaml.safe_load(f)
            
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    def run_ablation(self, train_loader, val_loader, test_loader):
        """运行消融实验"""
        results = {}
        
        # 1. 仅使用原始嵌入
        results['baseline'] = self.evaluate_baseline(test_loader)
        
        # 2. 仅使用投影头
        results['projection_only'] = self.evaluate_projection(train_loader, 
                                                           val_loader, 
                                                           test_loader)
        
        # 3. 不同数量的专家
        expert_nums = [4, 8, 16]
        for num in expert_nums:
            results[f'experts_{num}'] = self.evaluate_experts(train_loader, 
                                                            val_loader,
                                                            test_loader, 
                                                            num_experts=num)
        
        # 4. 不同的温度参数
        temperatures = [0.1, 0.2, 0.5]
        for temp in temperatures:
            results[f'temperature_{temp}'] = self.evaluate_temperature(
                train_loader, 
                val_loader,
                test_loader, 
                temperature=temp
            )
            
        return results
        
    def evaluate_baseline(self, test_loader):
        """评估基线模型"""
        model = BaselineClassifier(self.config['model']['input_dim']).to(self.device)
        return self._evaluate_model(model, test_loader)
        
    def evaluate_projection(self, train_loader, val_loader, test_loader):
        """评估仅使用投影头的效果"""
        model = ProjectionOnlyModel(self.config['model']['input_dim']).to(self.device)
        self._train_model(model, train_loader, val_loader)
        return self._evaluate_model(model, test_loader)
        
    def evaluate_experts(self, train_loader, val_loader, test_loader, num_experts):
        """评估不同数量专家的效果"""
        model = ExpertModel(
            self.config['model']['input_dim'], 
            num_experts=num_experts
        ).to(self.device)
        self._train_model(model, train_loader, val_loader)
        return self._evaluate_model(model, test_loader)
        
    def evaluate_temperature(self, train_loader, val_loader, test_loader, temperature):
        """评估不同温度参数的效果"""
        model = ContrastiveModel(
            self.config['model']['input_dim'], 
            temperature=temperature
        ).to(self.device)
        self._train_model(model, train_loader, val_loader)
        return self._evaluate_model(model, test_loader) 