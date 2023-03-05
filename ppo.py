
import gymnasium as gym
import torch
import numpy as np 
import time 
from torch.utils.tensorboard import SummaryWriter


import graph_envs
import utils
import graph_envs.utils
import networks
import networks.model_configs

torch.set_printoptions(sci_mode=False, precision=2, threshold=1000)
np.set_printoptions(suppress=True, precision=2, threshold=1000)
torch.autograd.set_detect_anomaly(True)
seed = None 


# ===========================
# ======= Env Config ========
# ===========================

device = torch.device('cpu')

env_args = {
    'n_nodes': 10,
    'n_edges': 14,
    'weighted': True,
}


# env_id = 'ShortestPath-v0'
env_id = 'SteinerTree-v0'


if env_id == 'ShortestPath-v0':
    has_mask = True
    mask_shape = (env_args['n_nodes'],)
elif env_id == 'SteinerTree-v0':
    has_mask = True
    mask_shape = (2*env_args['n_edges'],)
    env_args['n_dests'] = 3


# ===========================
# ======= PPO Config ========
# ===========================

n_envs=8
n_eval_envs=4
n_steps=64
batch_size = n_envs * n_steps
total_steps = 5000000
num_updates = total_steps // batch_size
mini_batch_size = 32
n_mini_batches = batch_size // mini_batch_size

n_epochs = 4
eval_freq = 100
eval_steps = 128

anneal_lr = True
# learning_rate = 2.5e-4
learning_rate = 1e-3

gamma = 0.995
gae = False # False seems to work better
gae_lambda = 0.95
adv_norm = False # False seems to work really better!
clip_value_loss = True

clip_coef = 0.2
entropy_loss_coef = 0.01
value_loss_coef = 0.5
max_grad_norm = 0.5

# ===========================
# ====== Model Config =======
# ===========================

# model_type = 'GNN'
# model_type = 'GNN_full' #TODO: Implement this
model_type = 'Transformer'
model_config = networks.model_configs.get_default_config(model_type)

# ===========================
# ===========================

import graph_envs.steiner_tree

if __name__ == '__main__':
    
    # Initializing the environments
    
    envs = gym.vector.AsyncVectorEnv([
        lambda: 
            gym.wrappers.RecordEpisodeStatistics(
                gym.make(env_id, **env_args)
            )
        for _ in range(n_envs)])
    
    # tmp_env = gym.make(env_id, **env_args)
    # tmp_env.reset()
    # mask = tmp_env.get_mask()
    # for i in range(30):
    #     p = mask / mask.sum()
    #     action = np.random.choice(np.arange(mask.shape[0]), p=p)
    #     obs, reward, done, _, info= tmp_env.step(action)
    #     print(action, reward, done)
    #     mask = info['mask']
    #     if done:
    #         tmp_env.reset()
    #         mask = tmp_env.get_mask()
    # 1/0
            
    # env_args['is_eval_env'] = True
    eval_envs = gym.vector.AsyncVectorEnv([
        lambda:
            gym.wrappers.RecordEpisodeStatistics(
                gym.make(env_id, **env_args)   
            )
        for _ in range(n_eval_envs)])
    

    # Initializing the tensorboard writer
    run_name = f"run_{int(time.time())%1e7}_N{env_args['n_nodes']}_E{env_args['n_edges']}"
    writer = SummaryWriter(f'runs/{run_name}')
    
    # Initializing the model
    model = utils.get_model(model_type, model_config, env_id, env_args).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, eps=1e-5)
    
    print(f'Number of Model Params: {networks.utils.get_n_params(model)}')
    
    # Initializing the stacks
    obs_stack = torch.zeros((n_steps, n_envs) + envs.single_observation_space.shape).to(device)
    if has_mask:
        masks_stack = torch.zeros((n_steps, n_envs) + mask_shape).to(device) < 1
        
    action_stack = torch.zeros((n_steps, n_envs) + envs.single_action_space.shape).to(device)
    logprobs_stack = torch.zeros((n_steps, n_envs)).to(device)
    rewards_stack = torch.zeros((n_steps, n_envs)).to(device)
    dones_stack = torch.zeros((n_steps, n_envs)).to(device)
    values_stack = torch.zeros((n_steps, n_envs)).to(device)
    
    
    # Resetting the environments
    next_obs, info = envs.reset(seed=seed)
    next_obs = torch.Tensor(next_obs).to(device)
    
    if has_mask:
        next_mask = torch.BoolTensor(np.stack(info['mask'], axis=0)).to(device)
    next_done = torch.zeros(n_envs).to(device)
    
    global_step = 0
    
    for update in range(num_updates):
        
        if update % eval_freq == 0:
            # torch.save(model.state_dict(), f'./models/{run_name}.pth')
            # eval_model(model, eval_envs, eval_steps,
            #            has_mask=has_mask, device=device, seed=seed,
            #            writer=writer, global_step=global_step,
            #            pick_max=True, print_sol=False)
            print('Implement Evaluation!')
            # eval_model(model, eval_envs, 128*2, has_mask=has_mask, device=device, seed=seed, pick_max=False, print_sol=True)
            
        t_start = time.time()
        
        # Annealing the learning rate
        if anneal_lr:
            frac = 1.0 - (update / num_updates)
            lr_now = frac * learning_rate
            optimizer.param_groups[0]['lr'] = lr_now
            
        # Collecting the trajectories
        ep_rews = []
        ep_lens = []
        opt_sols = []
        sols_found = []

        for step in range(n_steps):
            global_step += n_envs
            obs_stack[step] = next_obs
            dones_stack[step] = next_done
            if has_mask:
                masks_stack[step] = next_mask
            
            with torch.no_grad():
                x, edge_features, edge_index = graph_envs.utils.devectorize_graph(next_obs, env_id, **env_args)
                action, logprob, entropy, value = utils.forward_pass(model, model_type, x, edge_features, edge_index, has_mask, next_mask, actions=None)
            
            
            action_stack[step] = action
            logprobs_stack[step] = logprob
            values_stack[step] = value.flatten()

            
            obs, reward, done, _, info = envs.step(action.cpu().numpy())
            rewards_stack[step] = torch.Tensor(reward).to(device)
            # print('act! ', action, 'reward= ', reward)
            
            next_obs = torch.Tensor(obs).to(device)
            if has_mask:
                next_mask = torch.BoolTensor(np.stack(info['mask'], axis=0)).to(device)
                
            next_done = torch.Tensor(done).to(device)

            
            if 'final_info' in info:
                for e in range(n_envs):
                    if info['final_info'][e] != None:
                        # print('optimal solution: ', info['final_info'][e]['optimal_solution'])
                        
                        # print('---------')
                        # print('Nodes: ', obs[e, 0:10])
                        # gg = graph_envs.utils.devectorize_graph(torch.from_numpy(obs), env_id, **env_args)
                        # print('Edges: \n', gg[2][0].T.cpu().numpy())
                        
                        solved = info['final_info'][e]['solved']
                        
                        ep_rew = info['final_info'][e]['episode']['r']
                        ep_len = info['final_info'][e]['episode']['l']
                        opt = info['final_info'][e]['heuristic_solution']
                        
                        if solved:
                            opt_sols.append(opt)
                            ep_lens.append(ep_len)
                            ep_rews.append(ep_rew)
                        
                        sols_found.append(solved)
                        
        writer.add_scalar('charts/ep_rew', np.mean(ep_rews), global_step)
        writer.add_scalar('charts/ep_len', np.mean(ep_lens), global_step)
        
        print('----------------------------------')
        print(f'Update: {update}, Mean Ep Rew: {np.mean(ep_rews)}, Mean Ep Len: {np.mean(ep_lens)}, Sample_size: {len(ep_rews)}')
        print(f'Solved: {np.mean(sols_found)}, Mean opt: {np.mean(opt_sols)}')
        
        
        # Computing the returns
        with torch.no_grad():
            # Getting the last value
        
            
            x, edge_features, edge_index = graph_envs.utils.devectorize_graph(next_obs, env_id, **env_args)
            # print(x.T, '\n', edge_features.T, '\n', edge_index)
            _, _, _, next_value = utils.forward_pass(model, model_type, x, edge_features, edge_index, has_mask, next_mask, actions=None)
            
            
                
            next_value = next_value.reshape(1, -1)
            
            if gae:
                advantages_stack = torch.zeros_like(rewards_stack).to(device)
                lastgaelam = 0
                for t in reversed(range(n_steps)):
                    if t == n_steps - 1:
                        nextnonterminal = 1.0 - next_done
                        nextvalues = next_value
                    else:
                        nextnonterminal = 1.0 - dones_stack[t + 1]
                        nextvalues = values_stack[t + 1]
                        
                    delta = rewards_stack[t] + gamma * nextvalues * nextnonterminal - values_stack[t]
                    advantages_stack[t] = lastgaelam = delta + gamma * gae_lambda * nextnonterminal * lastgaelam
                returns_stack = advantages_stack + values_stack 
            else:
                returns_stack = torch.zeros_like(rewards_stack).to(device)
                for t in reversed(range(n_steps)):
                    if t == n_steps - 1:
                        nextnonterminal = 1.0 - next_done
                        nextvalues = next_value
                    else:
                        nextnonterminal = 1.0 - dones_stack[t + 1]
                        nextvalues = values_stack[t + 1]
                    returns_stack[t] = rewards_stack[t] + gamma * nextvalues * nextnonterminal
                advantages_stack = returns_stack - values_stack
        # print(rewards_stack)
        # print(returns_stack)
        # print(advantages_stack)
        
        # Batches of flattened trajectories
        b_obs = obs_stack.reshape((-1,) + envs.single_observation_space.shape)
        b_actions = action_stack.reshape((-1,) + envs.single_action_space.shape)
        if has_mask:
            b_masks = masks_stack.reshape((-1,) + mask_shape)
        b_rewards = rewards_stack.reshape(-1)
        b_logprobs = logprobs_stack.reshape(-1)
        b_advantages = advantages_stack.reshape(-1)
        b_returns = returns_stack.reshape(-1)
        b_values = values_stack.reshape(-1)
        
        
        b_inds = np.arange(batch_size)
        clipfracs = []
        
        for epoch in range(n_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, batch_size, mini_batch_size):
                end = start + mini_batch_size
                mb_inds = b_inds[start:end]
                
                x, edge_features, edge_index = graph_envs.utils.devectorize_graph(b_obs[mb_inds], env_id, **env_args)
                _, new_logprobs, entropy, new_values = utils.forward_pass(model, model_type, x, edge_features, edge_index, has_mask, b_masks[mb_inds], b_actions[mb_inds])
                
                
                # embedding = model.encode(b_obs[mb_inds])
                # if has_mask:
                #     _, new_logprobs, entropy, new_values = model.decode(x=embedding, mask=b_masks[mb_inds], action=b_actions[mb_inds])
                # else:
                #     _, new_logprobs, entropy, new_values = model.decode(x=embedding, action=b_actions[mb_inds])
                
                logratio = new_logprobs - b_logprobs[mb_inds]
                ratio = logratio.exp()
                with torch.no_grad():
                    old_approx_kl = logratio.mean()
                    approx_kl = ((ratio-1) - logratio).mean()
                    clipfracs += [((ratio-1).abs() > clip_coef).float().mean().item()]
                    
                
                # print(ratio)
                mb_advantages = b_advantages[mb_inds]
                if adv_norm:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * ratio.clamp(1 - clip_coef, 1 + clip_coef)
                # print(pg_loss1, pg_loss2)
                pg_loss = torch.max(pg_loss1, pg_loss2)
                # print(pg_loss)
                
                
                new_values = new_values.view(-1)
                if clip_value_loss:
                    # Clipped value loss
                    value_pred_clipped = b_values[mb_inds] + (new_values - b_values[mb_inds]).clamp(-clip_coef, clip_coef)
                    value_losses_clipped = (value_pred_clipped - b_returns[mb_inds]).pow(2)
                    
                    # Unclipped value loss
                    value_losses = (new_values - b_returns[mb_inds]).pow(2)
                    
                    value_loss = 0.5 * torch.max(value_losses, value_losses_clipped).mean()
                else:
                    value_loss = 0.5 * (new_values - b_returns[mb_inds]).pow(2).mean()
                
                entropy_loss = entropy.mean()
                
                loss = pg_loss.mean() + value_loss_coef * value_loss - entropy_loss_coef * entropy_loss
                
                optimizer.zero_grad()

                
                loss.backward()
                
                # for p in model.parameters():
                #     p.grad.data.clamp_(-max_grad_norm, max_grad_norm)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                optimizer.step()
        
        y_pred, y_true = b_values.cpu().detach().numpy(), b_returns.cpu().detach().numpy()
        var_y = np.var(y_true)
        explained_var = 1 - np.var(y_true - y_pred) / var_y
        
        writer.add_scalar('losses/explained_var', explained_var, global_step)
        writer.add_scalar('losses/loss', loss.item(), global_step)
        writer.add_scalar('losses/pg_loss', pg_loss.mean().item(), global_step)
        writer.add_scalar('losses/value_loss', value_loss.item(), global_step)
        writer.add_scalar('losses/entropy_loss', entropy_loss.item(), global_step)
        writer.add_scalar('losses/kl', approx_kl.item(), global_step)
        writer.add_scalar('losses/clipfrac', np.mean(clipfracs), global_step)
        
        writer.add_scalar('charts/learning_rate', optimizer.param_groups[0]['lr'], global_step)
        writer.add_scalar('charts/sps', n_steps * n_envs / (time.time() - t_start), global_step)
    
    envs.close()
    writer.close()