import torch
from utils import DotDict

GNN_config = DotDict({
    'dropout': 0.1,
    'norm': True,
    'activation': torch.nn.GELU,
    'layers': 5,
    'hidden': 32,
})


def get_default_config(model_type: str):
    if model_type == 'GNN':
        return GNN_config
    
    