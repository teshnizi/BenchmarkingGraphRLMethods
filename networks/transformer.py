import torch
import torch.nn as nn
import numpy as np  
import math
import torch_geometric as pyg


class Attention(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        
        self.config = config
        assert self.config.hidden % self.config.att_heads == 0, f"Hidden size {self.config.hidden} must be divisible by number of attention heads {self.config.att_heads}"
        
        self.config = config
        
        self.q = nn.Linear(self.config.hidden, self.config.hidden)
        self.k = nn.Linear(self.config.hidden, self.config.hidden)
        self.v = nn.Linear(self.config.hidden, self.config.hidden)
        self.out_lin = nn.Linear(self.config.hidden, self.config.hidden)
        self.dropout = nn.Dropout(p=self.config.dropout)
        self.sa_layer_norm = nn.LayerNorm([self.config.hidden])
        self.out_layer_norm = nn.LayerNorm([self.config.hidden])
        
        self.e = nn.Linear(self.config.hidden, self.config.hidden)
        
        self.edge_coeff_lin = nn.Sequential(
            nn.Linear(self.config.hidden, self.config.hidden),
            self.config.activation(),
            nn.Dropout(p=self.config.dropout),
            nn.Linear(self.config.hidden, self.config.hidden),
            self.config.activation(),
            nn.Dropout(p=self.config.dropout),
            nn.Linear(self.config.hidden, self.config.att_heads),
        )
        
        self.post_process = nn.Sequential(
            nn.Linear(self.config.hidden, 2*self.config.hidden),
            nn.GELU(),
            nn.Linear(2*self.config.hidden, self.config.hidden),
            nn.Dropout(p=self.config.dropout),
        )
        
        self.dim_per_head = (self.config.hidden // self.config.att_heads)

        self.alpha = torch.nn.Parameter(torch.ones(1))
        self.beta = torch.nn.Parameter(torch.ones(1))
        
    
    def forward(self, x: torch.Tensor, edge_attr: torch.Tensor, edge_index: torch.Tensor):
        
        # print(x.shape)
        bs, seq_len, _ = x.shape
           
        def shape(x: torch.Tensor) -> torch.Tensor:
            """separate heads"""
            return x.view(bs, x.shape[1], self.config.att_heads, -1).transpose(1, 2)

        def unshape(x: torch.Tensor) -> torch.Tensor:
            """group heads"""
            return x.transpose(1, 2).contiguous().view(bs, seq_len, self.config.att_heads * self.dim_per_head)
        
        q_vals = shape(self.q(x))
        k_vals = shape(self.k(x))
        v_vals = shape(self.v(x))

        tmp = self.q(x)
        tmp2 = unshape(shape(tmp))
        assert (tmp-tmp2).abs().max() < 1e-5
        # print('sanity: ', (tmp-tmp2).abs().max())
        
        q_vals = q_vals / math.sqrt(self.dim_per_head) # (bs, n_heads, seq_len, dim_per_head)
        scores = torch.matmul(q_vals, k_vals.transpose(2, 3)) # (bs, n_heads, seq_len, seq_len)
        
        coeffs = self.edge_coeff_lin(edge_attr)
        
        coeffs = coeffs.reshape(bs, seq_len * seq_len, self.config.att_heads, 1)
        coeffs = shape(coeffs)
        coeffs = coeffs.reshape(bs, self.config.att_heads, seq_len, seq_len)

        
        scores = self.alpha * scores + self.beta * coeffs

        weights = torch.softmax(scores, dim=-1)  # (bs, n_heads, seq_length, seq_length)
        weights = self.dropout(weights)  # (bs, n_heads, seq_length, seq_length)
        res = torch.matmul(weights, v_vals) # (bs, n_heads, seq_length, dim_per_head)
        res = unshape(res) # (bs, seq_length, hidden)
        res = self.out_lin(res) # (bs, seq_length, hidden)
        
        res = self.sa_layer_norm(res + x)
        res = self.out_layer_norm(self.post_process(res) + res)
        return res



class Transformer(torch.nn.Module):
    def __init__(self, node_f: int, edge_f: int, action_type: str, config: dict) -> None:
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
        
        self.positional_encoding = torch.nn.Embedding(self.config.max_seq_len, self.config.hidden)

        self.attention_layers = torch.nn.ModuleList([Attention(self.config) for _ in range(self.config.layers)])
        
        
        if self.action_type == 'node':
            action_hidden = self.config.hidden
        else:
            action_hidden = self.config.hidden * 2
        
        self.action_net = torch.nn.Sequential(
            torch.nn.Linear(action_hidden, self.config.hidden),
            torch.nn.ReLU(),
            torch.nn.Linear(self.config.hidden, 1)
        )
        
        self.critic_net = torch.nn.Linear(self.config.hidden, 1)
                
    
    def encode(self, batch_graph: pyg.data.Batch):
        
        x, edge_features, edge_index, batch = batch_graph.x, batch_graph.edge_attr, batch_graph.edge_index, batch_graph.batch
        
        x = x.reshape(batch_graph.num_graphs, -1, self.node_f)
        x = self.node_proc(x)
        
        x += self.positional_encoding(torch.arange(x.shape[1], device=x.device))
        
        edge_features = pyg.utils.to_dense_adj(edge_index, batch=batch, edge_attr=edge_features)
        edge_features = edge_features / 100000.0 + 1.0
        edge_features = self.edge_proc(edge_features)
        
        for att_layer in self.attention_layers:
            x = att_layer(x, edge_features, edge_index)
        
        global_features = x.mean(dim=1)
        
        return x, edge_features, global_features
    
    
    def forward(self, batch_graph, mask: torch.Tensor, actions: torch.Tensor=None) -> torch.Tensor:
        
        x, edge_features, global_features = self.encode(batch_graph)
        edge_index = batch_graph.edge_index
        
        # print(x.shape, edge_features.shape, edge_index.shape)
        
        if self.action_type == 'node':    
            logits = self.action_net(x)
        elif self.action_type == 'edge':
            x = x.reshape(-1, self.config.hidden)
            edge_features = torch.cat([x[edge_index[0]], x[edge_index[1]]], dim=-1)
            logits = self.action_net(edge_features)
            # print(edge_features.shape)
        
        logits = logits.reshape(batch_graph.num_graphs, -1)
        logits[~mask] = -torch.inf
        
        dist = torch.distributions.Categorical(logits=logits)
        
        if actions is None:
            actions = dist.sample()
        
        logprobs = dist.log_prob(actions)
        entropy = dist.entropy()
        
        value = self.critic_net(global_features)
        
        # 1/0    
        return actions, logprobs, entropy, value
        