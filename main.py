
import gymnasium as gym
import torch
import numpy as np 
import time


import graph_envs
import utils
import graph_envs.utils
import networks
import networks.model_configs

import learning.train

torch.set_printoptions(sci_mode=False, precision=2, threshold=1000)
np.set_printoptions(suppress=True, precision=2, threshold=1000)

seed = None 

# ===========================
# ======= Env Config ========
# ===========================

device = torch.device('cuda:0')

env_args = {
    'n_nodes': 30,
    'n_edges': -1,
    # 'n_choices': 29,
    # 'num_products': 2,
    # 'n_dests': 5,
    'weighted': True,
    # 'spatial': False,
    'parenting': -1,
}


if env_args['n_edges'] == -1:
    env_args['n_edges'] = int((env_args['n_nodes'] * (env_args['n_nodes'] - 1) // 2) * 0.30)
    
env_id = 'ShortestPath-v0'
# env_id = 'SteinerTree-v0'
# env_id = 'MaxIndependentSet-v0'
# env_id = 'TSP-v0'
# env_id = 'DistributionCenter-v0'
# env_id = 'MulticastRouting-v0'
# env_id = 'LongestPath-v0'
# env_id = 'DensestSubgraph-v0'
# env_id = 'PerishableProductDelivery-v0'


# ===========================
# ======= PPO Config ========
# ===========================

train_config = {
'n_envs': 8,
'n_steps':128,
# 'total_steps': 5000000,
'mini_batch_size': 128,
'n_epochs': 4,
'n_eval_envs': 4,
'eval_freq': 10,
'eval_steps': 100,
'anneal_lr': True,
# 'learning_rate' : 2.5e-4,
'learning_rate': 5e-4,
'gamma': 0.995,
'gae': False, # False seems to work better
'gae_lambda': 0.95,
'adv_norm': False, # False seems to work really better!
'clip_value_loss': True,
'clip_coef': 0.2,
'entropy_loss_coef': 0.01,
'value_loss_coef': 0.5,
'max_grad_norm': 0.5}

train_config = utils.DotDict(train_config)

train_config.batch_size = (train_config.n_envs * train_config.n_steps)
# train_config.num_updates = (train_config.total_steps // train_config.batch_size)
train_config.num_updates = 1000

train_config.n_mini_batches = (train_config.batch_size // train_config.mini_batch_size)
train_config.seed = seed

# ===========================
# ====== Model Config =======
# ===========================

model_type = 'GNN'
# model_type = 'Transformer'
# model_type = 'GNN_GCN'
# model_type = 'GNN_GAT'
# model_type = 'GNN_GTN'

model_config = networks.model_configs.get_default_config(model_type)

# ===========================
# ===========================

import graph_envs.steiner_tree

if __name__ == '__main__':
    
    # Initializing the tensorboard writer
    run_name = f"run_{int(time.time())%1e7}_{env_id}_{model_type}_N{env_args['n_nodes']}_E{env_args['n_edges']}_Parenting{env_args['parenting'] if 'parenting' in env_args else 'NA'}"\
        + f"{'__Spatial' + str(env_args['spatial']) if 'spatial' in env_args else ''}"
    
    # Initializing the model
    model = utils.get_model(model_type, model_config, env_id).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=train_config.learning_rate, eps=1e-5)
    
    print(f'Number of Model Params: {networks.utils.get_n_params(model)}')
    
    for i in range(env_args['n_nodes'], env_args['n_nodes']+1, 5):
        
        # env_args['n_nodes'] = i
        # env_args['n_edges'] = i * 2
        
        print('*-'*10)
        print('Nodes: ', env_args['n_nodes'], 'Edges: ', env_args['n_edges'])
        print('Env: ', env_id, ' || Model: ', model_type, ' || Parenting: ', env_args["parenting"] if 'parenting' in env_args else 'NA')
        print('*-'*10)
        
        # Initializing the environments
        
        env_args['is_eval_env'] = False
        envs = gym.vector.AsyncVectorEnv([
            lambda: 
                gym.wrappers.RecordEpisodeStatistics(
                    gym.make(env_id, **env_args)
                )
            for _ in range(train_config.n_envs)])
        

        env_args['is_eval_env'] = True
        eval_envs = gym.vector.AsyncVectorEnv([
            lambda:
                gym.wrappers.RecordEpisodeStatistics(
                    gym.make(env_id, **env_args)   
                )
            for _ in range(train_config.n_eval_envs)])
        
            
        learning.train.train_ppo(model, optimizer, envs, eval_envs, run_name, train_config, model_type, env_id, env_args, device)
        break
    