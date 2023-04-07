import gymnasium as gym
import torch 
import numpy as np 
import torch_geometric as pyg 
import networkx as nx 
import matplotlib.pyplot as plt

import graph_envs.utils

import networks.model_configs
import utils 
import learning.eval


env_args = {
    'n_nodes': 20,
    'n_edges': 30,
    'n_dests': 3,
    'weighted': True,
    # 'target_count': 10,
    'parenting': 'Advanced',
}

if env_args['n_edges'] == -1:
    env_args['n_edges'] = int((env_args['n_nodes'] * (env_args['n_nodes'] - 1) // 2) * 0.30)
    

device = torch.device('cpu')

# env_id = 'ShortestPath-v0'
# env_id = 'SteinerTree-v0'
# env_id = 'MaxIndependentSet-v0'
# env_id = 'TSP-v0'
# env_id = 'DistributionCenter-v0'
env_id = 'MulticastRouting-v0'


if env_id == 'ShortestPath-v0':
    has_mask = True
    mask_shape = (env_args['n_nodes'],)
elif env_id == 'SteinerTree-v0':
    has_mask = True
    mask_shape = (2*env_args['n_edges'],)
    env_args['n_dests'] = 5
elif env_id == 'MaxIndependentSet-v0':
    has_mask = True
    mask_shape = (env_args['n_nodes'],)
elif env_id == 'TSP-v0':
    has_mask = True
    mask_shape = (env_args['n_nodes'],)
elif env_id == 'DistributionCenter-v0':
    has_mask = True
    mask_shape = (env_args['n_nodes'],)
elif env_id == 'MulticastRouting-v0':
    has_mask = True
    mask_shape = (2*env_args['n_edges'],)


model_type = 'GNN'
# model_type = 'Transformer'


if __name__ == '__main__':
    
    params_path = "models/" +\
        "run_731603.0_MulticastRouting-v0_GNN_N10_E17_up30.pth"
    
    model_config = networks.model_configs.get_default_config(model_type)
    model = utils.get_model(model_type, model_config, env_id).to(device)
    model.load_state_dict(torch.load(params_path, map_location=device))
    
    
    env_args['is_eval_env'] = True
    env = gym.vector.AsyncVectorEnv([
        lambda:
            gym.wrappers.RecordEpisodeStatistics(
                gym.make(env_id, **env_args)   
            )
        for _ in range(1)])
    
    
    next_obs, info = env.reset(seed=4124)
    next_obs = torch.Tensor(next_obs).to(device)
    
    if has_mask:
        next_mask = torch.BoolTensor(np.stack(info['mask'], axis=0)).to(device)
    next_done = False
    
    with torch.no_grad():
        x, edge_features, edge_index = graph_envs.utils.devectorize_graph(next_obs, env_id, **env_args)
        action, _, _, _ = utils.forward_pass(model, model_type, x, edge_features, edge_index, has_mask, next_mask, actions=None)
    
    if has_mask:
        next_mask = torch.BoolTensor(np.stack(info['mask'], axis=0)).to(device)
        
    print(f'Investigating {params_path} on {env_id} with {model_type} model')
    for step in range(100):
        
        with torch.no_grad():
            x, edge_features, edge_index = graph_envs.utils.devectorize_graph(next_obs, env_id, **env_args)
            action, _, _, _ = utils.forward_pass(model, model_type, x, edge_features, edge_index, has_mask, next_mask, actions=None, pick_max=True)
        
        obs, reward, done, _, info = env.step(action.cpu().numpy())
        next_obs = torch.Tensor(obs).to(device)
        
        if has_mask:
            next_mask = torch.BoolTensor(np.stack(info['mask'], axis=0)).to(device)
            
        
        next_done = torch.Tensor(done).to(device)
        
        print(f'{step:3d}. Action: {action[0].item()}, Reward: {reward[0]:.2f}, Done: {done[0]}', flush=True)
        
        if done[0]:
            break
        
    total_reward = info['final_info'][0]['episode']['r']
    print(f'Total Reward: {total_reward[0]}')
    print(f'Heuristic Solution: {info["final_info"][0]["heuristic_solution"]}, \nSolution Cost: {info["final_info"][0]["solution_cost"]}')
    print(f'Solved: {info["final_info"][0]["solved"]}')
    # visualize the graph with links "edge_index" and weights "edge_features":
    
    
    data = pyg.data.Data(x=x[0], edge_index=edge_index[0].T, edge_attr=edge_features[0])
    G = pyg.utils.to_networkx(data, edge_attrs=['edge_attr'])
    

    edge_color = ['orange' if \
        (G.edges[(u,v)]['edge_attr'][1] == 1 or G.edges[(v,u)]['edge_attr'][1] == 1)\
         else 'black' for (u,v) in G.edges]
    node_color = ['red' if x[0, n, 1] == 1 else 'grey' for n in G.nodes]
    node_color[0] = 'orange'
    
    
    for e in G.edges:
        # print(G.edges[e], G.edges[e]['edge_attr'])
        G.edges[e]['edge_attr'] = round(G.edges[e]['edge_attr'][0] * 10)/10
    
    # save image of graph to file:
    layout = nx.kamada_kawai_layout(G)
    nx.draw(G, pos=layout, with_labels=True, edge_color=edge_color, node_color=node_color)
    nx.draw_networkx_edge_labels(G, pos=layout, edge_labels=nx.get_edge_attributes(G, 'edge_attr'), font_color='red')
    plt.savefig("graph.png")
    
    n = env_args['n_nodes']
    
    
    # for m in range(n, (n*n)//2, 10):
    #     # env_args['n_edges'] = m    
            
    #     env = gym.vector.AsyncVectorEnv([
    #     lambda:
    #         gym.wrappers.RecordEpisodeStatistics(
    #             gym.make(env_id, **env_args)   
    #         )
    #     for _ in range(8)])
        
    #     print('---------')
    #     print(f'Number of nodes: {env_args["n_nodes"]}')
    #     print(f'Number of edges: {env_args["n_edges"]}')     
    #     learning.eval.eval_model(model, model_type, env_id, env_args, env, n_steps=env_args["n_nodes"]*2*16, has_mask=has_mask, device=device, seed=None, writer=None, global_step=0, pick_max=False, verbose=False)
    #     break
        