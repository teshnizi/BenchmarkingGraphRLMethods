import torch 
import torch_geometric as pyg
import numpy as np

from networks.utils import ResidualMLP
        
class AlgoGAT(torch.nn.Module):
    def __init__(self, node_f: int, edge_f: int, action_type: str, config) -> None:
        
        super().__init__() 
        
        # config.hidden = int(config.hidden * 1.4) # Parameter count adjustment across models
        
        self.node_f = node_f
        self.edge_f = edge_f
        self.action_type = action_type
        self.config = config
        
        
        self.node_proc = torch.nn.Sequential(
            torch.nn.Linear(self.node_f, self.config.hidden),
            config.activation(),
            torch.nn.Dropout(config.dropout),
        )
        
        self.edge_proc = torch.nn.Sequential(
            torch.nn.Linear(self.edge_f, self.config.hidden),
            config.activation(),
            torch.nn.Dropout(config.dropout),
        )

        self.init_u = torch.nn.Parameter(torch.zeros(1, self.config.hidden))
        torch.nn.init.xavier_uniform_(self.init_u)
        
        self.gnn_layers = torch.nn.ModuleList([
            pyg.nn.GATv2Conv(in_channels=self.config.hidden, 
                           out_channels=self.config.hidden//2,
                           heads=2, 
                           dropout=config.dropout,
                           edge_dim=self.config.hidden,
                           )
            for _ in range(config.layers)])

    
        self.norm_layers = torch.nn.ModuleList([torch.nn.LayerNorm(self.config.hidden) for _ in range(config.layers)])
        self.dropout = torch.nn.Dropout(config.dropout)
        
        self.action_net = torch.nn.Sequential(
            torch.nn.Linear(self.config.hidden, self.config.hidden),
            self.config.activation(),
            torch.nn.Dropout(config.dropout),
            torch.nn.Linear(self.config.hidden, 1)
        )
        
        self.critic_net = torch.nn.Sequential(
            torch.nn.Linear(self.config.hidden, self.config.hidden),
            self.config.activation(),
            torch.nn.Dropout(config.dropout),
            torch.nn.Linear(self.config.hidden, 1)
        )
        
        self.global_net = torch.nn.Sequential(
            torch.nn.Linear(3 * self.config.hidden, self.config.hidden),
            self.config.activation(),
            torch.nn.LayerNorm(self.config.hidden),
            torch.nn.Dropout(config.dropout),
        )
        self.edge_net = torch.nn.Sequential(
            torch.nn.Linear(3 * self.config.hidden, self.config.hidden),
            self.config.activation(),
            torch.nn.LayerNorm(self.config.hidden),
            torch.nn.Dropout(config.dropout),
        )
        
        print('Initiated! AlgoGATv2')
    
    
    def encode(self, batch_graph: pyg.data.Batch):

        x, edge_features, edge_index, batch = batch_graph.x, batch_graph.edge_attr, batch_graph.edge_index, batch_graph.batch

        x = self.node_proc(x)
        edge_features = self.edge_proc(edge_features)

        for i in range(self.config.layers):
            
            x = x + self.gnn_layers[i](x, edge_index, edge_attr=edge_features)
            
            x = self.norm_layers[i](x)
            x = self.dropout(x)            
            
            
        global_features = torch.cat([
            x.reshape(batch_graph.num_graphs, -1, self.config.hidden).mean(dim=1),
            x.reshape(batch_graph.num_graphs, -1, self.config.hidden).max(dim=1).values,
            x.reshape(batch_graph.num_graphs, -1, self.config.hidden).min(dim=1).values,
        ], dim=-1)
        global_features = self.global_net(global_features)

        edge_features = self.edge_net(torch.cat([x[edge_index[0]], edge_features, x[edge_index[1]]], dim=-1))
    
        return x, edge_features, global_features
    
    
    def forward(self, batch_graph: pyg.data.Batch, mask: torch.Tensor, actions: torch.Tensor=None, pick_max: bool=False) -> torch.Tensor:
        
        
        x, edge_features, global_features = self.encode(batch_graph)
        
        
        if self.action_type == 'node':    
            logits = self.action_net(x)
        elif self.action_type == 'edge':
            logits = self.action_net(edge_features)
        
        logits = logits.reshape(batch_graph.num_graphs, -1)
        logits[~mask] = -torch.inf
        
        dist = torch.distributions.Categorical(logits=logits)
        
        if actions is None:
            
            if pick_max:
                actions = torch.argmax(logits, dim=-1)
            else:
                actions = dist.sample()
        
        logprobs = dist.log_prob(actions)
        entropy = dist.entropy()
        
        value = self.critic_net(global_features)
    
        return actions, logprobs, entropy, value
        