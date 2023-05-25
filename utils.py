from networks import gnn, transformer, algo_gcn, algo_gat, algo_gtn, graphormer
import torch
import graph_envs
import torch.nn as nn 
import torch_geometric as pyg
import networkx as nx 
import time

from transformers.models.graphormer.collating_graphormer import preprocess_item, GraphormerDataCollator

import matplotlib.pyplot as plt

class DotDict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__
    


def get_model(model_type, model_config, env_id):
    
    node_f, edge_f, action_type = graph_envs.utils.get_env_info(env_id)
        
    if model_type == "GNN":
        return gnn.GNN(node_f=node_f, edge_f=edge_f, action_type=action_type, config=model_config)
    elif model_type == "Transformer":
        return transformer.Transformer(node_f=node_f, edge_f=edge_f, action_type=action_type, config=model_config)
    elif model_type == "GNN_GCN":
        return algo_gcn.AlgoGCN(node_f=node_f, edge_f=edge_f, action_type=action_type, config=model_config)
    elif model_type == "GNN_GAT":
        return algo_gat.AlgoGAT(node_f=node_f, edge_f=edge_f, action_type=action_type, config=model_config)
    elif model_type == "GNN_GTN":
        return algo_gtn.AlgoGraphTransformer(node_f=node_f, edge_f=edge_f, action_type=action_type, config=model_config)
    elif model_type == "Graphormer":
        return graphormer.GraphormerAgent(node_f=node_f, edge_f=edge_f, action_type=action_type, config=model_config)
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
                 pick_max: bool=False):
    
    if model_type.startswith('GNN'):
        model_input = graph_envs.utils.to_pyg_graph(x, edge_features, edge_index)
        
    elif model_type == 'Transformer':
        model_input = graph_envs.utils.to_pyg_graph(x, edge_features, edge_index)
        
    elif model_type == 'Graphormer':
        
        # print(x.shape)
        # print(edge_features.shape)
        # print(edge_index.shape)
        # print(x.device)
        # time this step:
        
        st = time.time()
        
        device = x.device
        
        model_input = {}
        
        for i in range(x.shape[0]):
            
            dct = {'num_nodes': x.shape[1],
                                 'node_feat': x[i,:,:].cpu(),
                                 'edge_attr': edge_features[i,:,:].cpu(),
                                 'edge_index': edge_index[i,:,:].cpu().transpose(-1, -2),
                                 'y': [0]}

            
            dp = preprocess_item(dct)
            dp['input_nodes'] = torch.Tensor(dp['input_nodes']).unsqueeze(0).long()
            dp['attn_bias'] = torch.Tensor(dp['attn_bias']).unsqueeze(0)
            dp['spatial_pos'] = torch.Tensor(dp['spatial_pos']).unsqueeze(0).long()
            dp['input_edges'] = torch.Tensor(dp['input_edges']).unsqueeze(0).long()
            dp['in_degree'] = torch.Tensor(dp['in_degree']).unsqueeze(0).long()
            dp['out_degree'] = torch.Tensor(dp['out_degree']).unsqueeze(0).long()
            dp['labels'] = torch.Tensor(dp['labels']).unsqueeze(0).long()
            dp['edge_index'] = torch.Tensor(dp['edge_index']).unsqueeze(0).long()
            dp['edge_attr'] = torch.Tensor(dp['edge_attr']).unsqueeze(0)
            dp['y'] = torch.Tensor(dp['y']).unsqueeze(0).long()
            dp['num_nodes'] = torch.Tensor(dp['num_nodes']).unsqueeze(0).long()
            dp['node_feat'] = torch.Tensor(dp['node_feat']).unsqueeze(0)
            dp['attn_edge_type'] = torch.Tensor(dp['attn_edge_type']).unsqueeze(0).long()
            
            for k in dp:
                if not k in model_input:
                    model_input[k] = []
                model_input[k].append(dp[k])

        for k in model_input:
            model_input[k] = torch.cat(model_input[k], dim=0).to(device)
        
        # print("Time to preprocess: ", time.time() - st)
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
    plt.figure(figsize=(7,7))
    layout = nx.kamada_kawai_layout(G)
    nx.draw(G, pos=layout, with_labels=True, edge_color=edge_color, node_color=node_color)
    nx.draw_networkx_edge_labels(G, pos=layout, edge_labels=nx.get_edge_attributes(G, 'edge_attr'), font_color='red')
    plt.savefig(file_name)
    