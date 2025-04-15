import os
import torch
from config import Config
from utils import prepare_data, set_seed, encode_samples
from ablation_study import run_ablation_study, create_directories
from classifier_comparison import run_classifier_comparison
from transformers import AutoTokenizer, AutoModel
import json
import logging
from datetime import datetime
from analysis import analyze_results

def setup_logging():
    """设置日志"""
    log_dir = '2025_0309/logs'
    os.makedirs(log_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(f'{log_dir}/experiment_{timestamp}.log'),
            logging.StreamHandler()
        ]
    )

def load_model_and_tokenizer():
    """加载预训练模型和分词器"""
    logging.info("Loading model and tokenizer...")
    modelType = ['codebert-base', 'codeExecutor', 'CodeReviewer', 
                 'graphCodeBert', 'longcoder-base', 'unixcoder']
    modelPath = '../../../model/' + modelType[Config.MODEL_IDX] + '/'
    
    tokenizer = AutoTokenizer.from_pretrained(modelPath)
    model = AutoModel.from_pretrained(modelPath).to(Config.DEVICE)
    return tokenizer, model

def load_and_process_data():
    """加载并处理数据"""
    logging.info("Loading data...")
    langType = ['code_pairs_cpp_python.jsonl', 'code_pairs_java_c.jsonl', 
                'code_pairs_java_cs.jsonl', 'code_pairs_java_python.jsonl']
    langPath = Config.DATASET_PATH + langType[Config.LANG_IDX]
    
    # 加载原始数据
    samples = []
    with open(langPath, 'r', encoding='utf-8') as f:
        for line in f:
            record = json.loads(line.strip())
            samples.append(record)
            if len(samples) >= Config.TRAIN_SIZE:
                break
    
    # 加载模型和分词器
    tokenizer, model = load_model_and_tokenizer()
    
    # 编码样本
    logging.info("Encoding samples...")
    encoded_samples = encode_samples(samples, tokenizer, model, Config.DEVICE)
    
    return encoded_samples

def run_experiment():
    """运行实验"""
    # 设置随机种子
    set_seed(Config.SEED)
    
    # 设置日志
    setup_logging()
    create_directories()
    
    # 加载并处理数据
    logging.info("Loading and preparing data...")
    encoded_samples = load_and_process_data()
    train_dataset, val_dataset, test_dataset = prepare_data(encoded_samples)
    
    # 运行消融实验
    logging.info("Starting ablation study...")
    ablation_results = run_ablation_study(train_dataset, val_dataset, test_dataset)
    
    # 运行分类器比较实验
    logging.info("Starting classifier comparison...")
    classifier_results = run_classifier_comparison(train_dataset, val_dataset, test_dataset)
    
    # 综合分析结果
    logging.info("Analyzing results...")
    analyze_results(ablation_results, classifier_results)
    
    logging.info("Experiment completed!")
    return ablation_results, classifier_results

if __name__ == "__main__":
    results = run_experiment() 