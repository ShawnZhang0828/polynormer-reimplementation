import torch
import torch.nn as nn
import torch.nn.functional as F

from models.global_attention import GlobalAttention
from models.local_attention import LocalAttention


class Polynormer(nn.Module):
    def __init__(
        self,
        in_dim,
        hidden_dim,
        out_dim,
        n_local_layers,
        n_global_layers,
        n_local_heads=1,
        n_global_heads=8,
        use_local_attention_network=True,
        local_dropout=0.0,
        use_relu=False,
    ):
        super().__init__()

        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim
        self.n_local_layers = n_local_layers
        self.n_global_layers = n_global_layers
        self.n_local_heads = n_local_heads
        self.n_global_heads = n_global_heads
        self.use_relu = use_relu

        self.input_projection = nn.Linear(in_dim, hidden_dim)

        self.local_layers = nn.ModuleList(
            [
                LocalAttention(
                    dim=hidden_dim,
                    use_attention_network=use_local_attention_network,
                    n_heads=n_local_heads,
                    dropout=local_dropout,
                )
                for _ in range(n_local_layers)
            ]
        )

        self.local_h = nn.ModuleList(
            [nn.Linear(hidden_dim, hidden_dim) for _ in range(n_local_layers)]
        )

        self.local_betas = nn.Parameter(torch.zeros(n_local_layers, hidden_dim))

        self.local_norms = nn.ModuleList(
            [nn.LayerNorm(hidden_dim) for _ in range(n_local_layers)]
        )

        self.global_layers = nn.ModuleList(
            [
                GlobalAttention(dim=hidden_dim, n_heads=n_global_heads)
                for _ in range(n_global_layers)
            ]
        )

        self.global_h = nn.ModuleList(
            [nn.Linear(hidden_dim, hidden_dim) for _ in range(n_global_layers)]
        )

        self.global_betas = nn.Parameter(torch.zeros(n_global_layers, hidden_dim))

        self.global_norms = nn.ModuleList(
            [nn.LayerNorm(hidden_dim) for _ in range(n_global_layers)]
        )

        self.prediction_head = nn.Linear(hidden_dim, out_dim)

        self.reset_parameters()

    def reset_parameters(self):
        self.input_projection.reset_parameters()

        for local_layer in self.local_layers:
            local_layer.reset_parameters()
        for global_layer in self.global_layers:
            global_layer.reset_parameters()

        for layer in self.local_h:
            layer.reset_parameters()
        for layer in self.global_h:
            layer.reset_parameters()

        nn.init.zeros_(self.local_betas)
        nn.init.zeros_(self.global_betas)

        for layer in self.local_norms:
            layer.reset_parameters()
        for layer in self.global_norms:
            layer.reset_parameters()

        self.prediction_head.reset_parameters()

    def forward(self, x, edge_index):
        if x.size(1) != self.in_dim:
            raise ValueError(
                f"INPUT ERROR: Expected input dimension {self.in_dim}, got {x.size(1)}"
            )
        if edge_index.dim() != 2 or edge_index.size(0) != 2:
            raise ValueError(
                f"INPUT ERROR: edge_index must be of shape [2, num_edges], got {edge_index.shape}"
            )

        x = self.input_projection(x)

        # Local modules
        local_x = torch.zeros_like(x)
        for i, local_layer in enumerate(self.local_layers):
            h = self.local_h[i](x)
            beta = torch.sigmoid(self.local_betas[i]).unsqueeze(0)
            layer_out = local_layer(x, edge_index)
            layer_norm = self.local_norms[i](
                h * layer_out
            )  # Add layer norm to stabilize training

            x = (1 - beta) * layer_norm + beta * layer_out  # Hadamard product

            if self.use_relu:
                x = F.relu(x)

            local_x = local_x + x

        x = local_x

        # Global modules
        for i, global_layer in enumerate(self.global_layers):
            h = self.global_h[i](x)
            beta = torch.sigmoid(self.global_betas[i])
            layer_out = global_layer(x)
            layer_norm = self.global_norms[i](
                h * layer_out
            )  # Add layer norm to stabilize training

            x = (1 - beta) * layer_norm + beta * layer_out  # Hadamard product

            if self.use_relu:
                x = F.relu(x)

        # Final prediction
        out = self.prediction_head(x)

        return out
