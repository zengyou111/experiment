config = {
    'model': {
        'input_dim': 768,
        'hidden_dim': 512,
        'num_experts': 8,
    },
    'training': {
        'projection_head_epochs': 1000,
        'classifier_epochs': 1000,
        'projection_head_patience': 5,
        'classifier_patience': 5,
        'projection_head_lr': 3e-6,
        'classifier_lr': 1e-6,
        'batch_size': 64,
        'temperature': 0.2,
    },
    'data': {
        'train_size': 3000,
        'val_size': 500,
        'test_size': 500,
    },
    'logging': {
        'log_interval': 100,
        'save_interval': 10,
    }
} 