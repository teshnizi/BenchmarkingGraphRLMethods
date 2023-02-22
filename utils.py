from networks import gnn


class DotDict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__
    
    
def get_model(model_type, model_config, env_id, env_args):
    
    if model_type == "GNN":
        if env_id == "ShortestPath-v0":
            return gnn.GNN(node_f=1, edge_f=1, action_type="node", config=model_config)
        