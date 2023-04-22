

import torch 
import graph_envs
import numpy as np

import utils 


def eval_model(model, model_type, env_id, env_args, eval_envs, n_steps, has_mask, device, seed, writer, global_step, pick_max=True, verbose=False):
    
    model.eval()
    
    next_obs, info = eval_envs.reset(seed=seed)
    next_obs = torch.Tensor(next_obs).to(device)
    
    if has_mask:
        next_mask = torch.BoolTensor(np.stack(info['mask'], axis=0)).to(device)
    next_done = torch.zeros(eval_envs.num_envs).to(device)
    
    with torch.no_grad():
        x, edge_features, edge_index = graph_envs.utils.devectorize_graph(next_obs, env_id, **env_args)
        action, _, _, _ = utils.forward_pass(model, model_type, x, edge_features, edge_index, has_mask, next_mask, actions=None)
        
        
    ep_rews = []
    ep_lens = []
    opt_sols = []
    sol_costs = []
    sols_found = []
    
    
    while True:
        
        with torch.no_grad():
            x, edge_features, edge_index = graph_envs.utils.devectorize_graph(next_obs, env_id, **env_args)
            action, _, _, _ = utils.forward_pass(model, model_type, x, edge_features, edge_index, has_mask, next_mask, actions=None, pick_max=pick_max,)
        
        
        obs, reward, done, _, info = eval_envs.step(action.cpu().numpy())
        next_obs = torch.Tensor(obs).to(device)
        
        if has_mask:
            next_mask = torch.BoolTensor(np.stack(info['mask'], axis=0)).to(device)
            
        next_done = torch.Tensor(done).to(device)
        
        if verbose:
            print(f'{step:3d}. Action: {action}, Reward: {reward}, Done: {done}')
            
        if 'final_info' in info:
            for e in range(eval_envs.num_envs):
                if info['final_info'][e] != None:
                    
                    solved = info['final_info'][e]['solved']
                    
                    # ep_rew = info['final_info'][e]['episode']['r']
                    # ep_len = info['final_info'][e]['episode']['l']
                    opt = info['final_info'][e]['heuristic_solution']
                    sol_cost = info['final_info'][e]['solution_cost']
                    
                    if solved:
                        opt_sols.append(opt)
                        # ep_lens.append(ep_len)
                        # ep_rews.append(ep_rew)
                        sol_costs.append(sol_cost)
                    
                    sols_found.append(solved)
        
        if len(sols_found) >= n_steps:
            break
        
    sols_found = np.array(sols_found)
    opt_sols = np.array(opt_sols)
    sol_costs = np.array(sol_costs)
    
    
    print('=============================')
    # print(f'Eval!, Mean Ep Rew: {np.mean(ep_rews)}, Mean Ep Len: {np.mean(ep_lens)}, Sample_size: {len(ep_rews)}')
    print(f'Solved: {np.mean(sols_found)}, Mean Solution Cost: {np.mean(sol_costs)}, Mean Opt Cost: {np.mean(opt_sols)}, Sample_size: {len(sols_found)}')
    if len(opt_sols) > 0:
        print(f'Overall Relative Gap: {(np.mean(sol_costs - opt_sols)) / np.mean(opt_sols)}')
        print(f'Average Relative Gap: {np.mean((sol_costs - opt_sols) / opt_sols)}')
        print(f'Minimum Sol Cost: {np.min(sol_costs)}')
        print(f'Minimum Relative Gap: {np.min((sol_costs - opt_sols) / opt_sols)}')
        print(f'Std ratio: {np.std(sol_costs)/np.std(opt_sols)}')
        print(f'Std: {np.std((sol_costs - opt_sols) / (opt_sols))}', flush=True)
    
    # k=0
    # print('-----------------------------')
    # for i in range(len(sols_found)):
    #     if sols_found[i]:
    #         cost = sol_costs[i-k]
    #     else:
    #         cost = -1
    #         k += 1
    #     print(i, sols_found[i], cost)
    # print(k)
    # print('-----------------------------')
    
    if writer != None:
    
        writer.add_scalar('Eval/mean_solved', np.mean(sols_found), global_step)
        writer.add_scalar('Eval/mean_opt_cost', np.mean(opt_sols), global_step)
        writer.add_scalar('Eval/mean_sol_cost', np.mean(sol_costs), global_step)
        writer.add_scalar('Eval/overall_rel_gap', (np.mean(sol_costs - opt_sols)) / np.mean(opt_sols), global_step)
        writer.add_scalar('Eval/average_rel_gap', np.mean((sol_costs - opt_sols) / opt_sols), global_step)
        writer.add_scalar('Eval/std_ratio', np.std(sol_costs)/np.std(opt_sols), global_step)
        
    return np.mean(sols_found), np.mean(sol_costs), np.mean(opt_sols), 
