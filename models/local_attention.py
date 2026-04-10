import torch
import torch.nn as nn

try:
    from torch_geometric.nn import GATConv, GCNConv
except ImportError as e:
    raise ImportError(
        "IMPORT ERROR: torch_geometric is required for LocalAttention. Please install it first."
    ) from e


class LocalAttention(nn.Module):
    def __init__(
        self,
        dim,
        use_attention_network=True,
        n_heads=1,
        dropout=0.0,
        add_self_loops=True,
        bias=False,
    ):
        super().__init__()

        self.dim = dim
        self.n_heads = n_heads
        self.use_attention_network = use_attention_network
        self.dropout = dropout
        self.add_self_loops = add_self_loops
        self.bias = bias

        # Use GATConv for homophilic/heterophilic datasetsa and GCNConv for large-scale datasets
        if use_attention_network:
            if dim % n_heads != 0:
                raise ValueError(
                    "INITIALIZATION ERROR: Dimension must be divisible by number of heads"
                )

            self.head_dim = dim // n_heads
            self.conv = GATConv(
                in_channels=dim,
                out_channels=self.head_dim,
                heads=n_heads,
                dropout=dropout,
                add_self_loops=add_self_loops,
                bias=bias,
                concat=True,
            )
        else:
            self.conv = GCNConv(
                in_channels=dim,
                out_channels=dim,
                add_self_loops=add_self_loops,
                bias=bias,
            )

        self.reset_parameters()

    def reset_parameters(self):
        self.conv.reset_parameters()

    def forward(self, x, edge_index):
        if x.size(1) != self.dim:
            raise ValueError(
                f"INPUT ERROR: Expected input dimension {self.dim}, got {x.size(1)}"
            )
        if edge_index.dim() != 2 or edge_index.size(0) != 2:
            raise ValueError(
                f"INPUT ERROR: edge_index must be of shape [2, num_edges], got {edge_index.shape}"
            )

        out = self.conv(x, edge_index)

        if out.shape != x.shape:
            raise RuntimeError(
                f"OUTPUT ERROR: Expected output shape {x.shape}, got {out.shape}"
            )

        return out
