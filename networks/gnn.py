import torch 
import torch_geometric as pyg
import numpy as np

from networks.utils import MyGNNLayer
        
class GNN(torch.nn.Module):
    def __init__(self, node_f: int, edge_f: int, action_type: str, config) -> None:
        
        super().__init__() 
        self.node_f = node_f
        self.edge_f = edge_f
        self.action_type = action_type
        self.config = config
        
        self.node_proc = torch.nn.Sequential(
            torch.nn.Linear(self.node_f, self.config.hidden),
            config.activation(),
        )
        
        self.edge_proc = torch.nn.Sequential(
            torch.nn.Linear(self.edge_f, self.config.hidden),
            config.activation(),
        )

        self.init_u = torch.nn.Parameter(torch.zeros(1, self.config.hidden))
        
        self.gnn_layers = torch.nn.ModuleList([
            MyGNNLayer(self.config.hidden, config.activation, 2, config.norm, config.dropout) for _ in range(config.layers)])
        
        if self.action_type == 'node':
            self.action_net = torch.nn.Linear(self.config.hidden, 1)
            
        self.critic_net = torch.nn.Linear(self.config.hidden, 1)
    
    def encode(self, batch_graph: pyg.data.Batch):
        node_features = self.node_proc(batch_graph.x)
        edge_features = self.edge_proc(batch_graph.edge_attr)
        global_features = self.init_u.repeat(batch_graph.num_graphs, 1)
        
        for gnn_layer in self.gnn_layers:                
            node_features, edge_features, global_features = gnn_layer(node_features, batch_graph.edge_index, edge_features, global_features, batch_graph.batch)
        
        return node_features, edge_features, global_features
    
    
    def forward(self, batch_graph: pyg.data.Batch, mask: torch.Tensor, action: torch.Tensor=None) -> torch.Tensor:
        
        node_features, edge_features, global_features = self.encode(batch_graph)
    
        
        if self.action_type == 'node':
            node_probs = torch.sigmoid(self.action_net(node_features))
            node_probs = node_probs.reshape(batch_graph.num_graphs, -1)

            node_probs[~mask] = -torch.inf
            dist = torch.distributions.Categorical(logits=node_probs)
            actions = dist.sample()
            logprobs = dist.log_prob(actions)
            entropy = dist.entropy()
            
            value = self.critic_net(global_features)
            
            return actions, logprobs, entropy, value
        pass
