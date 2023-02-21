
import gnn 
import torch 
        
        
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
        