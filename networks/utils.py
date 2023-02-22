
import torch 
from torch_scatter import scatter_mean
from torch_geometric.nn import MetaLayer
        
class ResidualMLP(torch.nn.Module):
    def __init__(self, 
                 input_dim: int,
                 hidden_f: int,
                 output_dim: int,
                 activation: torch.nn.Module = torch.nn.GELU,
                 num_layers: int = 3,
                 norm: bool = True,
                 dropout: float = 0.1) -> None:
        super().__init__()
        
        self.hidden_f = hidden_f
        self.activation = activation
        self.norm = norm
        
        self.layers = []
        
        for i in range(num_layers-1):
            if i == 0:
                self.layers.append(torch.nn.Linear(input_dim, hidden_f))
            else :
                self.layers.append(torch.nn.Linear(hidden_f, hidden_f))
            self.layers.append(activation())
            self.layers.append(torch.nn.Dropout(dropout))
        
        self.layers.append(torch.nn.Linear(hidden_f, output_dim))
        self.layers.append(activation())
        self.layers.append(torch.nn.Dropout(dropout))
        
        self.layers = torch.nn.ModuleList(self.layers)
        
        if self.norm:
            self.norm_layer = torch.nn.LayerNorm(hidden_f)
            
    def forward(self, x):
        init_x = x
        
        for layer in self.layers:
            x = layer(x)
        if self.norm:
            if init_x.shape != x.shape:
                x = self.norm_layer(x)
            else:
                x = self.norm_layer(x + init_x)
        return x


class MyGNNLayer(torch.nn.Module):
    def __init__(self, hidden_f: int=32, activation: torch.nn.Module = torch.nn.GELU, num_layers: int = 2, norm: bool = True, dropout: float = 0.1):
        super(MyGNNLayer, self).__init__()

        self.edge_mlp = torch.nn.Sequential(
            torch.nn.Linear(2*hidden_f + hidden_f + hidden_f, hidden_f),
            activation(),
            torch.nn.Dropout(dropout)
        )
        self.edge_norm = torch.nn.LayerNorm(hidden_f)
        
        self.node_mlp_1 = torch.nn.Sequential(
            torch.nn.Linear(hidden_f + hidden_f, hidden_f),
            activation(),
            torch.nn.Dropout(dropout)
        )
        self.node_mlp_2 = torch.nn.Sequential(
            torch.nn.Linear(hidden_f + hidden_f, hidden_f),
            activation(),
            torch.nn.Dropout(dropout)
        )
        self.node_norm = torch.nn.LayerNorm(hidden_f)
        
        self.global_mlp = torch.nn.Sequential(
            torch.nn.Linear(hidden_f + hidden_f, hidden_f),
            activation(),
            torch.nn.Dropout(dropout)
        )
        self.global_norm = torch.nn.LayerNorm(hidden_f)
        
        
        def edge_model(src, dest, edge_attr, u, batch):
            
            out = torch.cat([src, dest, edge_attr, u[batch]], 1)
            out = self.edge_mlp(out)
            out = self.edge_norm(out + edge_attr)
            return out


        def node_model(x, edge_index, edge_attr, u, batch):
            row, col = edge_index
            out = torch.cat([x[col], edge_attr], dim=1)
            out = self.node_mlp_1(out)
            out = scatter_mean(out, row, dim=0, dim_size=x.size(0))
            out = torch.cat([out, u[batch]], dim=1)
            out = self.node_mlp_2(out)
            out = self.node_norm(out + x)
            return out
            

        def global_model(x, edge_index, edge_attr, u, batch):  
            out = torch.cat([u, scatter_mean(x, batch, dim=0)], dim=1)
            out = self.global_mlp(out)
            out = self.global_norm(out + u)
            return out

        self.op = MetaLayer(edge_model, node_model, global_model)

    def forward(self, x, edge_index, edge_attr, u, batch):
        return self.op(x, edge_index, edge_attr, u, batch)