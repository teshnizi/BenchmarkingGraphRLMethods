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

        self.gnn_layers = torch.nn.ModuleList([
            MyGNNLayer(self.config.hidden, config.activation, 2, config.norm, config.dropout) for _ in range(config.layers)])
        
        
    def forward(self, batch_graph: pyg.data.Batch, mask: torch.Tensor, action: torch.Tensor=None) -> torch.Tensor:
        
        print(batch_graph, mask.shape)
        1/0
        pass
