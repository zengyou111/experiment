import torch

class Config:
    # 设备配置
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 数据配置
    TRAIN_SIZE = 10
    BATCH_SIZE = 32
    MAX_LENGTH = 512
    
    # 模型配置
    INPUT_DIM = 768
    HIDDEN_DIM = 256
    
    # 训练配置
    EPOCHS = 100
    LEARNING_RATE = 1e-4
    PATIENCE = 5
    TEMPERATURE = 0.2
    
    # 消融实验配置
    ABLATION_CONFIGS = {
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
    
    # 可视化配置
    FIGURE_SIZE = (15, 5)
    DPI = 300
    
    # 数据集配置
    LANG_IDX = 3  # 选择数据集
    DATASET_PATH = '../../../Datasets/'
    
    # 模型保存配置
    MODEL_SAVE_PATH = '2025_0309/models'
    SAVE_BEST_MODEL = True
    
    # 实验配置
    SEED = 42
    NUM_WORKERS = 4
    
    # 评估指标配置
    METRICS = ['accuracy', 'f1', 'recall', 'precision']
    
    # 模型选择配置
    MODEL_IDX = 0  # 选择使用的预训练模型 