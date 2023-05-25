from transformers.models.graphormer.collating_graphormer import preprocess_item, GraphormerDataCollator
from transformers import GraphormerConfig, GraphormerModel
import torch

from typing import Dict

from networks.utils import MyGNNLayer, ResidualMLP
import torch_geometric as pyg


class GraphormerAgent(torch.nn.Module):
    def __init__(self, node_f: int, edge_f: int, action_type: str, config):
        super().__init__()
        
        self.action_type = action_type
        self.config = config
        
        self.encoder = GraphormerModel(GraphormerConfig(**config))
        
        if self.action_type == 'node':
            action_hidden = self.config.embedding_dim
        else:
            action_hidden = self.config.embedding_dim * 2
        
        self.action_net = torch.nn.Sequential(
            torch.nn.Linear(action_hidden, self.config.embedding_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(self.config.embedding_dim, 1)
        )
        
        self.critic_net = torch.nn.Linear(self.config.embedding_dim, 1)
        
    def encode(self, batch_graph):
            
        encoder_outputs = self.encoder(**batch_graph)
        outputs, hidden_states = encoder_outputs["last_hidden_state"], encoder_outputs["hidden_states"]
        
        node_features = outputs[:, 1:, :]
        global_features = outputs[:, 0, :]
            
        return node_features, global_features

    
    def forward(self, batch_graph: Dict[str, torch.Tensor], mask: torch.Tensor, actions: torch.Tensor=None, pick_max: bool=False) -> torch.Tensor:
        x, global_features = self.encode(batch_graph)
        
        edge_index = batch_graph['edge_index']
        
        if self.action_type == 'node':
            logits = self.action_net(x)
            
        elif self.action_type == 'edge':
            # Reshape edge_index for advanced indexing
            bs, _, edge_count = edge_index.shape
            edge_index = edge_index.transpose(-1, -2)

            edge_features = []
            for b in range(bs):
                edge_attr_start = x[b, edge_index[b, :, 0]]
                edge_attr_end = x[b, edge_index[b, :, 1]]
                edge_attr = torch.cat([edge_attr_start, edge_attr_end], dim=-1)
                edge_features.append(edge_attr)
                
            edge_features = torch.stack(edge_features, dim=0)
            
            logits = self.action_net(edge_features)
                      
        logits = logits.squeeze(-1)
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
        