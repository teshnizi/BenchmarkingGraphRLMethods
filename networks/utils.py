
import torch 
from torch_scatter import scatter_mean
from torch_geometric.nn import MetaLayer
        
class ResidualMLP(torch.nn.Module):
    def __init__(self, 
                 hidden_f: int,
                 activation: torch.nn.Module = torch.nn.GELU,
                 num_layers: int = 3,
                 norm: bool = True,
                 dropout: float = 0.1) -> None:
        super().__init__()
        
        self.hidden_f = hidden_f
        self.activation = activation
        self.norm = norm
        
        layers = []
        for i in range(num_layers):
            layers.append(torch.nn.Linear(hidden_f, hidden_f))
            layers.append(activation())
            layers.append(torch.nn.Dropout(dropout))
        
        if self.norm:
            self.norm_layer = torch.nn.LayerNorm(hidden_f)
            
        def forward(self, x):
            init_x = x
            for layer in self.layers:
                x = layer(x)
            if self.norm:
                x = self.norm_layer(x + init_x)
            return x


class MyGNNLayer(torch.nn.Module):
    def __init__(self, hidden_f: int=32, activation: torch.nn.Module = torch.nn.GELU, num_layers: int = 2, norm: bool = True, dropout: float = 0.1):
        super(MyGNNLayer, self).__init__()

        # self.edge_mlp = Seq(Lin(..., ...), torch.nn.GELU(), Lin(..., ...))
        self.edge_mlp = ResidualMLP(hidden_f, activation, num_layers, norm, dropout)
        self.node_mlp_1 = ResidualMLP(hidden_f, activation, num_layers, norm, dropout)
        self.node_mlp_2 = ResidualMLP(hidden_f, activation, num_layers, norm, dropout)
        # self.global_mlp = Seq(Lin(..., ...), torch.nn.GELU(), Lin(..., ...))
        self.global_mlp = ResidualMLP(hidden_f, activation, num_layers, norm, dropout)

        def edge_model(src, dest, edge_attr, u, batch):
            # source, target: [E, F_x], where E is the number of edges.
            # edge_attr: [E, F_e]
            # u: [B, F_u], where B is the number of graphs.
            # batch: [E] with max entry B - 1.
            out = torch.cat([src, dest, edge_attr, u[batch]], 1)
            return self.edge_mlp(out)

        def node_model(x, edge_index, edge_attr, u, batch):
            # x: [N, F_x], where N is the number of nodes.
            # edge_index: [2, E] with max entry N - 1.
            # edge_attr: [E, F_e]
            # u: [B, F_u]
            # batch: [N] with max entry B - 1.
            row, col = edge_index
            out = torch.cat([x[col], edge_attr], dim=1)
            out = self.node_mlp(out)
            out = scatter_mean(out, row, dim=0, dim_size=x.size(0))
            out = torch.cat([out, u[batch]], dim=1)
            return self.node_mlp_2(out)

        def global_model(x, edge_index, edge_attr, u, batch):
            # x: [N, F_x], where N is the number of nodes.
            # edge_index: [2, E] with max entry N - 1.
            # edge_attr: [E, F_e]
            # u: [B, F_u]
            # batch: [N] with max entry B - 1.
            out = torch.cat([u, scatter_mean(x, batch, dim=0)], dim=1)
            return self.global_mlp(out)

        self.op = MetaLayer(edge_model, node_model, global_model)

    def forward(self, x, edge_index, edge_attr, u, batch):
        return self.op(x, edge_index, edge_attr, u, batch)