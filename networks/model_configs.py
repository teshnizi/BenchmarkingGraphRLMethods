import torch
from utils import DotDict


GNN_config = DotDict({
    'dropout': 0.1,
    'norm': True,
    'activation': torch.nn.GELU,
    'layers': 12,
    'hidden': 64,
})


Transformer_config = DotDict({
    'dropout': 0.1,
    'norm': True,
    'activation': torch.nn.GELU,
    'layers': 12,
    'hidden': 64,
    'att_heads': 12,
    'max_seq_len': 100,
})


Graphormer_config = DotDict({
    # 'num_classes': 
    # 'num_atoms': 10, 
    # 'num_edges': 10,
    # 'num_in_degree': 1, 
    # 'num_out_degree':1,
    # 'num_edge_dis': 10,
    'layers': 12,
    'num_attention_heads': 4,
    'embedding_dim': 64,
    'dropout': 0.1,
    'attention_dropout': 0.1,
})


def get_default_config(model_type: str):
    if model_type.startswith('GNN'):
        return GNN_config
    elif model_type == 'Transformer':
        return Transformer_config
    elif model_type == 'Graphormer':
        return Graphormer_config
    else:
        raise NotImplementedError(f'Unknown model type: {model_type}')
