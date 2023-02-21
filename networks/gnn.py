import torch 
import torch_geometric as pyg
import numpy as np

        
        
class GNN(torch.nn.Module):
    def __init__(self, node_f: int, edge_f: int, action_type: str, hidden: int=32) -> None:
        super().__init__() 
        self.node_f = node_f
        self.edge_f = edge_f
        self.action_type = action_type

        self.node_proc = torch.nn.Sequential(
            torch.nn.Linear(node_f, hidden),
            torch.nn.GELU(),
        )
        
        self.edge_proc = torch.nn.Sequential(
            torch.nn.Linear(edge_f, hidden),
            torch.nn.GELU(),
        )
        
            
    def forward(self, x, edge_index, edge_attr):
        pass