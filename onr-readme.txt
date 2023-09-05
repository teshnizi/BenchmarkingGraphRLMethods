These are the main two files:


main_onr.py: used for configuring and running the train loop
multicast_env_onr.py: used for defining the environment


These are the places where you should change based on the environment development: 

----- graph_envs.utils, lines 69 to 72 -----

    elif env_id == 'MulticastONREnv-v0':
        node_f = 4
        edge_f = 2
        action_type = "node"

you need to choose the right number of node and edge features based on the data you want to include in each instance.


----- learning.train, lines 53 to 55 -----

    elif env_id == "MulticastONREnv-v0":
        has_mask = True
        mask_shape = (env_args["config"].num_nodes,)


you need to choose the right shape for the mask based on what the actions are (node, edge, messages, or a combination of them).

----- multicast_env_onr.py, all lines -----

The environemnt is defined here

----- main_onr.py, lines 27 to 52 -----

You can modify hyperparameters and configurations here.

