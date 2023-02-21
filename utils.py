from networks import gnn

def get_model(model_type, env_id, env_args):
    
    if model_type == "GNN":
        if env_id == "ShortestPath-v0":
            return gnn.GNN(node_f=1, edge_f=1, action_type="node", hidden=32)
        