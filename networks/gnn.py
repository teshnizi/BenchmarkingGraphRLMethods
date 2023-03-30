import torch 
import torch_geometric as pyg
import numpy as np

from networks.utils import MyGNNLayer, ResidualMLP
        
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
            torch.nn.Dropout(config.dropout),
        )
        
        self.edge_proc = torch.nn.Sequential(
            torch.nn.Linear(self.edge_f, self.config.hidden),
            config.activation(),
            torch.nn.Dropout(config.dropout),
        )

        self.init_u = torch.nn.Parameter(torch.zeros(1, self.config.hidden))
        torch.nn.init.xavier_uniform_(self.init_u)
        
        # self.gnn_layers = torch.nn.ModuleList([
        #     # pyg.nn.GATConv(in_channels=self.config.hidden, out_channels=self.config.hidden, dropout=config.dropout) for _ in range(config.layers)])
        #     MyGNNLayer(self.config.hidden, config.activation, 2, config.norm, config.dropout) for _ in range(config.layers)])
        
        self.norm = torch.nn.LayerNorm(self.config.hidden)
        
        self.gnn_layer = MyGNNLayer(self.config.hidden, config.activation, 2, config.norm, config.dropout)
        # self.gat_layer = pyg.nn.GATv2Conv(in_channels=self.config.hidden, out_channels=self.config.hidden//2, dropout=config.dropout, heads=2, edge_dim=self.config.hidden)
        
        # if self.action_type == 'node':
        # self.action_net = ResidualMLP(10, self.config.hidden, 10, config.activation, 2, norm=False, dropout=config.dropout)
            
        self.action_net = torch.nn.Sequential(
            torch.nn.Linear(self.config.hidden, self.config.hidden),
            self.config.activation(),
            torch.nn.Dropout(config.dropout),
            torch.nn.Linear(self.config.hidden, 1)
        )
        
        # elif self.action_type == 'edge':
        #     # self.action_netct =
        #     pass
        
        self.critic_net = torch.nn.Sequential(
            torch.nn.Linear(self.config.hidden, self.config.hidden),
            self.config.activation(),
            torch.nn.Dropout(config.dropout),
            torch.nn.Linear(self.config.hidden, 1)
        )
    
    
    def encode(self, batch_graph: pyg.data.Batch):

        x, edge_features, edge_index, batch = batch_graph.x, batch_graph.edge_attr, batch_graph.edge_index, batch_graph.batch

        x = self.node_proc(x)
        edge_features = self.edge_proc(edge_features)
        global_features = torch.repeat_interleave(self.init_u, batch_graph.num_graphs, dim=0)
        
        
        for i in range(self.config.layers):
            
            x, edge_features, global_features = self.gnn_layer(x, edge_index, edge_features, global_features, batch)
            # x = x + self.gat_layer(x, edge_index, edge_features)
            
            # x = self.norm(x)
            
            
        
        # global_features = x.reshape(batch_graph.num_graphs, -1, self.config.hidden).mean(dim=1)
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
        