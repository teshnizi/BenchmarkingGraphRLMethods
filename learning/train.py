
import torch
from torch.utils.tensorboard import SummaryWriter
import numpy as np 
import time 
import graph_envs
import utils 

def train_ppo(model, optimizer, envs, eval_envs, run_name, train_config, model_type, env_id, env_args, device):
    
    writer = SummaryWriter(f'runs/{run_name}')
    
    # Initializing the stacks
    obs_stack = torch.zeros((train_config.n_steps, train_config.n_envs) + envs.single_observation_space.shape).to(device)
    if train_config.has_mask:
        masks_stack = torch.zeros((train_config.n_steps, train_config.n_envs) + train_config.mask_shape).to(device) < 1
        
    action_stack = torch.zeros((train_config.n_steps, train_config.n_envs) + envs.single_action_space.shape).to(device)
    logprobs_stack = torch.zeros((train_config.n_steps, train_config.n_envs)).to(device)
    rewards_stack = torch.zeros((train_config.n_steps, train_config.n_envs)).to(device)
    dones_stack = torch.zeros((train_config.n_steps, train_config.n_envs)).to(device)
    values_stack = torch.zeros((train_config.n_steps, train_config.n_envs)).to(device)
    
    
    # Resetting the environments
    next_obs, info = envs.reset(seed=train_config.seed)
    next_obs = torch.Tensor(next_obs).to(device)
    
    if train_config.has_mask:
        next_mask = torch.BoolTensor(np.stack(info['mask'], axis=0)).to(device)
    next_done = torch.zeros(train_config.n_envs).to(device)
    
    global_step = 0
    
    for update in range(train_config.num_updates):
        
        if update % train_config.eval_freq == 0:
            # torch.save(model.state_dict(), f'./models/{run_name}.pth')
            # eval_model(model, eval_envs, train_config.eval_steps,
            #            has_mask=train_config.has_mask, device=device, seed=seed,
            #            writer=writer, global_step=global_step,
            #            pick_max=True, print_sol=False)
            print('Implement Evaluation!')
            
            
        t_start = time.time()
        
        # Annealing the learning rate
        if train_config.anneal_lr:
            frac = 1.0 - (update / train_config.num_updates)
            lr_now = frac * train_config.learning_rate
            optimizer.param_groups[0]['lr'] = lr_now
            
        # Collecting the trajectories
        ep_rews = []
        ep_lens = []
        opt_sols = []
        sol_costs = []
        sols_found = []

        for step in range(train_config.n_steps):
            global_step += train_config.n_envs
            obs_stack[step] = next_obs
            dones_stack[step] = next_done
            if train_config.has_mask:
                masks_stack[step] = next_mask
            
            with torch.no_grad():
                x, edge_features, edge_index = graph_envs.utils.devectorize_graph(next_obs, env_id, **env_args)
                action, logprob, entropy, value = utils.forward_pass(model, model_type, x, edge_features, edge_index, train_config.has_mask, next_mask, actions=None)
            
            
            action_stack[step] = action
            logprobs_stack[step] = logprob
            values_stack[step] = value.flatten()

            
            obs, reward, done, _, info = envs.step(action.cpu().numpy())
            rewards_stack[step] = torch.Tensor(reward).to(device)
            # print('act! ', action, 'reward= ', reward)
            
            next_obs = torch.Tensor(obs).to(device)
            if train_config.has_mask:
                next_mask = torch.BoolTensor(np.stack(info['mask'], axis=0)).to(device)
                
            next_done = torch.Tensor(done).to(device)

            
            if 'final_info' in info:
                for e in range(train_config.n_envs):
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
                        sol_cost = info['final_info'][e]['solution_cost']
                        
                        if solved:
                            opt_sols.append(opt)
                            ep_lens.append(ep_len)
                            ep_rews.append(ep_rew)
                            sol_costs.append(sol_cost)
                        
                        
                        sols_found.append(solved)
                        
        writer.add_scalar('charts/ep_rew', np.mean(ep_rews), global_step)
        writer.add_scalar('charts/ep_len', np.mean(ep_lens), global_step)
        writer.add_scalar('charts/solution_cost', np.mean(sol_costs), global_step)
        writer.add_scalar('charts/solved', np.mean(sols_found), global_step)
        
        
        print('----------------------------------')
        print(f'Update: {update}, Mean Ep Rew: {np.mean(ep_rews)}, Mean Ep Len: {np.mean(ep_lens)}, Sample_size: {len(ep_rews)}')
        print(f'Solved: {np.mean(sols_found)}, Mean Sol Cost: {np.mean(sol_costs)}, Mean Opt Sol: {np.mean(opt_sols)}')
        
        
        # Computing the returns
        with torch.no_grad():
            # Getting the last value
        
            
            x, edge_features, edge_index = graph_envs.utils.devectorize_graph(next_obs, env_id, **env_args)
            # print(x.T, '\n', edge_features.T, '\n', edge_index)
            _, _, _, next_value = utils.forward_pass(model, model_type, x, edge_features, edge_index, train_config.has_mask, next_mask, actions=None)
            
            
                
            next_value = next_value.reshape(1, -1)
            
            if train_config.gae:
                advantages_stack = torch.zeros_like(rewards_stack).to(device)
                lastgaelam = 0
                for t in reversed(range(train_config.n_steps)):
                    if t == train_config.n_steps - 1:
                        nextnonterminal = 1.0 - next_done
                        nextvalues = next_value
                    else:
                        nextnonterminal = 1.0 - dones_stack[t + 1]
                        nextvalues = values_stack[t + 1]
                        
                    delta = rewards_stack[t] + train_config.gamma * nextvalues * nextnonterminal - values_stack[t]
                    advantages_stack[t] = lastgaelam = delta + train_config.gamma * train_config.gae_lambda * nextnonterminal * lastgaelam
                returns_stack = advantages_stack + values_stack 
            else:
                returns_stack = torch.zeros_like(rewards_stack).to(device)
                for t in reversed(range(train_config.n_steps)):
                    if t == train_config.n_steps - 1:
                        nextnonterminal = 1.0 - next_done
                        nextvalues = next_value
                    else:
                        nextnonterminal = 1.0 - dones_stack[t + 1]
                        nextvalues = values_stack[t + 1]
                    returns_stack[t] = rewards_stack[t] + train_config.gamma * nextvalues * nextnonterminal
                advantages_stack = returns_stack - values_stack
        # print(rewards_stack)
        # print(returns_stack)
        # print(advantages_stack)
        
        # Batches of flattened trajectories
        b_obs = obs_stack.reshape((-1,) + envs.single_observation_space.shape)
        b_actions = action_stack.reshape((-1,) + envs.single_action_space.shape)
        if train_config.has_mask:
            b_masks = masks_stack.reshape((-1,) + train_config.mask_shape)
        b_rewards = rewards_stack.reshape(-1)
        b_logprobs = logprobs_stack.reshape(-1)
        b_advantages = advantages_stack.reshape(-1)
        b_returns = returns_stack.reshape(-1)
        b_values = values_stack.reshape(-1)
        
        
        b_inds = np.arange(train_config.batch_size)
        clipfracs = []
        
        for epoch in range(train_config.n_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, train_config.batch_size, train_config.mini_batch_size):
                end = start + train_config.mini_batch_size
                mb_inds = b_inds[start:end]
                
                x, edge_features, edge_index = graph_envs.utils.devectorize_graph(b_obs[mb_inds], env_id, **env_args)
                _, new_logprobs, entropy, new_values = utils.forward_pass(model, model_type, x, edge_features, edge_index, train_config.has_mask, b_masks[mb_inds], b_actions[mb_inds])
                
                
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
                    clipfracs += [((ratio-1).abs() > train_config.clip_coef).float().mean().item()]
                    
                
                # print(ratio)
                mb_advantages = b_advantages[mb_inds]
                if train_config.adv_norm:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * ratio.clamp(1 - train_config.clip_coef, 1 + train_config.clip_coef)
                # print(pg_loss1, pg_loss2)
                pg_loss = torch.max(pg_loss1, pg_loss2)
                # print(pg_loss)
                
                
                new_values = new_values.view(-1)
                if train_config.clip_value_loss:
                    # Clipped value loss
                    value_pred_clipped = b_values[mb_inds] + (new_values - b_values[mb_inds]).clamp(-train_config.clip_coef, train_config.clip_coef)
                    value_losses_clipped = (value_pred_clipped - b_returns[mb_inds]).pow(2)
                    
                    # Unclipped value loss
                    value_losses = (new_values - b_returns[mb_inds]).pow(2)
                    
                    value_loss = 0.5 * torch.max(value_losses, value_losses_clipped).mean()
                else:
                    value_loss = 0.5 * (new_values - b_returns[mb_inds]).pow(2).mean()
                
                entropy_loss = entropy.mean()
                
                loss = pg_loss.mean() + train_config.value_loss_coef * value_loss - train_config.entropy_loss_coef * entropy_loss
                
                optimizer.zero_grad()

                
                loss.backward()
                
                # for p in model.parameters():
                #     p.grad.data.clamp_(-max_grad_norm, max_grad_norm)
                torch.nn.utils.clip_grad_norm_(model.parameters(), train_config.max_grad_norm)
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
        writer.add_scalar('charts/sps', train_config.n_steps * train_config.n_envs / (time.time() - t_start), global_step)
    
    envs.close()
    writer.close()