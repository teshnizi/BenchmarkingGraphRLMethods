from networks import gnn
import torch
import graph_envs
import torch.nn as nn 
import torch_geometric as pyg

class DotDict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__
    


def get_model(model_type, model_config, env_id, env_args):
    
    if model_type == "GNN":
        if env_id == "ShortestPath-v0":
            return gnn.GNN(node_f=2, edge_f=1, action_type="node", config=model_config)
            # return Agent()
        elif env_id == "SteinerTree-v0":
            return gnn.GNN(node_f=1, edge_f=1, action_type="edge", config=model_config)


def forward_pass(model: torch.nn.Module, 
                 model_type: str,
                 x: torch.Tensor,
                 edge_features: torch.Tensor,
                 edge_index: torch.Tensor,
                 has_mask: bool,
                 masks: torch.Tensor=None,
                 actions: torch.Tensor=None):
    
    if model_type == 'GNN':
        model_input = graph_envs.utils.to_pyg_graph(x, edge_features, edge_index)
        
    
    # model_input = torch.cat([x.reshape(-1, 10), edge_index.reshape(-1, 2*40)/10.], dim=-1)
    # model_input = torch.cat([x.reshape(-1, 10), torch.rand(size=(x.shape[0], 2*90)).to(x.device)], dim=-1)
    # model_input = x.reshape(-1, 10)
    
    if has_mask:    
        action, logprob, entropy, value = model(model_input, masks, actions)
    else:
        action, logprob, entropy, value = model(model_input, actions)
        
    return action, logprob, entropy, value
        
