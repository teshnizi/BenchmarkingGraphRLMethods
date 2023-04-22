from networks import gnn, transformer
import torch
import graph_envs
import torch.nn as nn 
import torch_geometric as pyg
import networkx as nx 

import matplotlib.pyplot as plt

class DotDict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__
    


def get_model(model_type, model_config, env_id):
    
    node_f, edge_f, action_type = graph_envs.utils.get_env_info(env_id)
        
    if model_type.startswith("GNN"):
        return gnn.GNN(node_f=node_f, edge_f=edge_f, action_type=action_type, config=model_config)
        
    elif model_type == "Transformer":
        return transformer.Transformer(node_f=node_f, edge_f=edge_f, action_type=action_type, config=model_config)
    else:
        raise ValueError("Model type not supported")


def forward_pass(model: torch.nn.Module, 
                 model_type: str,
                 x: torch.Tensor,
                 edge_features: torch.Tensor,
                 edge_index: torch.Tensor,
                 has_mask: bool,
                 masks: torch.Tensor=None,
                 actions: torch.Tensor=None,
                 pick_max: bool=False,):
    
    if model_type == 'GNN':
        model_input = graph_envs.utils.to_pyg_graph(x, edge_features, edge_index)
    
    elif model_type == 'GNN_full':
        # TODO: fix this
        model_input = graph_envs.utils.to_pyg_graph(x, edge_features, edge_index)
        model_input.original_edges = edge_index
        
        x, edge_features, edge_index, batch = model_input.x, model_input.edge_attr, model_input.edge_index, model_input.batch
        adj = pyg.utils.to_dense_adj(edge_index, batch=batch, edge_attr=edge_features)
        adj[adj != 0] += 1
        
        edge_index, edge_features = pyg.utils.dense_to_sparse(adj[:,:,:,0])
        
        edge_index = (adj[:,:,:,0] + 1).nonzero().t()
        
        edge_features = adj[edge_index[0], edge_index[1], edge_index[2], :]
        row = edge_index[1] + adj[:,:,:,0].size(-2) * edge_index[0]
        col = edge_index[2] + adj[:,:,:,0].size(-1) * edge_index[0]
        edge_index = torch.stack([row, col], dim=0)
        
        model_input.x = x
        model_input.edge_attr = edge_features
        model_input.edge_index = edge_index
        
        
    elif model_type == 'Transformer':
        model_input = graph_envs.utils.to_pyg_graph(x, edge_features, edge_index)
        # model_input = (x, edge_features, edge_index.transpose(-1, -2))
        
    else:
        raise ValueError("Model type not supported")
    
    # model_input = torch.cat([x.reshape(-1, 10), edge_index.reshape(-1, 2*40)/10.], dim=-1)
    # model_input = torch.cat([x.reshape(-1, 10), torch.rand(size=(x.shape[0], 2*90)).to(x.device)], dim=-1)
    # model_input = x.reshape(-1, 10)
    
    if has_mask:    
        action, logprob, entropy, value = model(model_input, masks, actions, pick_max=pick_max)
    else:
        action, logprob, entropy, value = model(model_input, actions, pick_max=pick_max)
        
    return action, logprob, entropy, value
        



def draw_graph(G, file_name, edges_taken=[]):
    
    node_color = ['red' if G.nodes[n]['x'][1] == 1 else 'grey' for n in G.nodes]
    
    if len(edges_taken) == 0:
        edge_color = ['orange' if \
            (G.edges[(u,v)]['edge_attr'][1] == 1 or G.edges[(v,u)]['edge_attr'][1] == 1)\
            else 'black' for (u,v) in G.edges]
        node_color[0] = 'green'
    else:
        edge_color = ['orange' if ((u,v) in edges_taken) or ((v,u) in edges_taken) else 'black' for (u,v) in G.edges]
        node_color[edges_taken[0][0]] = 'green'
        
    
    for e in G.edges:
        # G.edges[e]['edge_attr'] = round(G.edges[e]['edge_attr'][0] * 10)/10
        G.edges[e]['edge_attr'] = round(G.edges[e]['edge_attr'] * 10)/10
    
    # save image of graph to file:
    plt.figure(figsize=(20,20))
    layout = nx.kamada_kawai_layout(G)
    nx.draw(G, pos=layout, with_labels=True, edge_color=edge_color, node_color=node_color)
    nx.draw_networkx_edge_labels(G, pos=layout, edge_labels=nx.get_edge_attributes(G, 'edge_attr'), font_color='red')
    plt.savefig(file_name)
    