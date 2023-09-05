import gymnasium as gym
import torch
import numpy as np
import time


# import graph_envs
import multicast_env_onr
import utils
import networks
import networks.model_configs

import learning.train

torch.set_printoptions(sci_mode=False, precision=2, threshold=1000)
np.set_printoptions(suppress=True, precision=2, threshold=1000)

seed = None

# ===========================
# ======= Env Config ========
# ===========================

device = torch.device("cuda:0")


class NetgenConfig:
    def __init__(self):
        self.seed = -1
        self.num_nodes = 11
        self.num_arcs = 30
        self.num_provisioned_bundles = 0
        self.num_all_bundles = 16
        self.num_priorities = 10
        self.bandwidth_lb = 15_000
        self.bandwidth_ub = 250_000
        self.latency_lb = 295_000
        self.latency_ub = 600_000
        self.rate_lb = 128
        self.rate2_lb = 128
        self.rate_ub = 65_536
        self.rate2_ub = 65_536
        self.delay_lb = 1_080_000
        self.delay2_lb = 1_080_000
        self.delay_ub = 9_072_000
        self.delay2_ub = 9_072_000
        self.num_unicast_reqs_per_bundle_lb = 0
        self.num_unicast_reqs_per_bundle_ub = 1
        self.bundle_size_lb = 1
        self.bundle_size_ub = 1
        self.request_size_lb = 2
        self.request_size_ub = 5

        # Extra parameters
        self.parenting = 0


config = NetgenConfig()

env_args = {
    "config": config,
}

env_id = "MulticastONREnv-v0"

# ===========================
# ======= PPO Config ========
# ===========================

train_config = {
    "n_envs": 8,
    "n_steps": 128,
    # 'total_steps': 5000000,
    "mini_batch_size": 128,
    "n_epochs": 4,
    "n_eval_envs": 4,
    "eval_freq": 10,
    "eval_steps": 100,
    "anneal_lr": True,
    # 'learning_rate' : 2.5e-4,
    "learning_rate": 5e-4,
    "gamma": 0.995,
    "gae": False,  # False seems to work better
    "gae_lambda": 0.95,
    "adv_norm": False,  # False seems to work really better!
    "clip_value_loss": True,
    "clip_coef": 0.2,
    "entropy_loss_coef": 0.01,
    "value_loss_coef": 0.5,
    "max_grad_norm": 0.5,
}

train_config = utils.DotDict(train_config)

train_config.batch_size = train_config.n_envs * train_config.n_steps
# train_config.num_updates = (train_config.total_steps // train_config.batch_size)
train_config.num_updates = 1000

train_config.n_mini_batches = train_config.batch_size // train_config.mini_batch_size
train_config.seed = seed

# ===========================
# ====== Model Config =======
# ===========================

model_type = "GNN"
# model_type = 'Transformer'
# model_type = 'GNN_GCN'
# model_type = 'GNN_GAT'
# model_type = 'GNN_GTN'

model_config = networks.model_configs.get_default_config(model_type)

# ===========================
# ===========================


if __name__ == "__main__":
    # Initializing the tensorboard writer
    run_name = f"run_{int(time.time())%1e7}_{env_id}_{model_type}_N{config.num_nodes}_E{config.num_arcs}_Parenting{config.parenting}"

    # Initializing the model
    model = utils.get_model(model_type, model_config, env_id).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=train_config.learning_rate, eps=1e-5
    )

    print(f"Number of Model Params: {networks.utils.get_n_params(model)}")

    print("*-" * 10)
    print("Nodes: ", config.num_nodes, "Edges: ", config.num_arcs)
    print(
        "Env: ",
        env_id,
        " || Model: ",
        model_type,
    )
    print("*-" * 10)

    # Initializing the environments

    env_args["is_eval_env"] = False
    envs = gym.vector.AsyncVectorEnv(
        [
            lambda: gym.wrappers.RecordEpisodeStatistics(gym.make(env_id, **env_args))
            for _ in range(train_config.n_envs)
        ]
    )

    env_args["is_eval_env"] = True
    eval_envs = gym.vector.AsyncVectorEnv(
        [
            lambda: gym.wrappers.RecordEpisodeStatistics(gym.make(env_id, **env_args))
            for _ in range(train_config.n_eval_envs)
        ]
    )

    learning.train.train_ppo(
        model,
        optimizer,
        envs,
        eval_envs,
        run_name,
        train_config,
        model_type,
        env_id,
        env_args,
        device,
    )
