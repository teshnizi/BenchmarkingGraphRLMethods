import torch
from utils import DotDict

GNN_config = DotDict({
    'dropout': 0.1,
    'norm': True,
    'activation': torch.nn.GELU,
    'layers': 5,
    'hidden': 32,
})

Transformer_config = DotDict({
    'dropout': 0.1,
    'norm': True,
    'activation': torch.nn.GELU,
    'layers': 5,
    'hidden': 32,
    'att_heads': 2,
    'max_seq_len': 100,
})


def get_default_config(model_type: str):
    if model_type.startswith('GNN'):
        return GNN_config
    if model_type == 'Transformer':
        return Transformer_config
    
    