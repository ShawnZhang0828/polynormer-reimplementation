import torch
import torch.nn as nn
import torch.nn.functional as F


class GlobalAttention(nn.Module):
    def __init__(self, dim, dropout, n_heads=8):
        super().__init__()

        if dim % n_heads != 0:
            raise ValueError(
                "INITIALIZATION ERROR: Dimension must be divisible by number of heads"
            )

        self.dim = dim
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.dropout = dropout

        self.q = nn.Linear(dim, dim, bias=True)
        self.k = nn.Linear(dim, dim, bias=True)
        self.v = nn.Linear(dim, dim, bias=True)

        self.reset_parameters()

    def reset_parameters(self):
        self.q.reset_parameters()
        self.k.reset_parameters()
        self.v.reset_parameters()

    def reshape_heads(self, x):
        n = x.size(0)
        x = x.view(
            n, self.n_heads, self.head_dim
        )  # Reshape D into multiple heads with head_dim dimensions
        x = x.permute(
            0, 2, 1
        ).contiguous()  # Permute to (n, head_dim, n_heads) for attention computation
        return x

    def merge_heads(self, x):
        n = x.size(0)
        x = x.permute(0, 2, 1).contiguous()  # Permute back to (n, n_heads, head_dim)
        x = x.view(n, self.dim)  # Merge heads back to original dimension
        return x

    def forward(self, x):
        if x.size(1) != self.dim:
            raise ValueError(
                f"INPUT ERROR: Expected input dimension {self.dim}, got {x.size(1)}"
            )

        # Compute query, key, value projections ([N, dim_h, n_heads])
        q = self.q(x)
        k = self.k(x)
        v = self.v(x)

        # Reshape Q, K, V for multi-head attention ([N, n_heads, dim_h])
        q = self.reshape_heads(q)
        k = self.reshape_heads(k)
        v = self.reshape_heads(v)

        q = torch.sigmoid(q)  # Apply sigmoid activation to Q
        k = torch.sigmoid(k)  # Apply sigmoid activation to K

        # Compute numerator (m = dim_features_k, d = dim_features_v)
        kv = torch.einsum("nmh,ndh->mdh", k, v)  # K^T V ([dim_h, dim_h, n_heads])
        numerator = torch.einsum(
            "ndh,mdh->nmh", q, kv
        )  # Q (K^T V) ([N, dim_h, n_heads])

        # Compute denominator
        k_sum = torch.einsum("ndh->dh", k)  # sum Ki ([dim_h, n_heads])
        denominator = torch.einsum("ndh,dh->nh", q, k_sum)  # Q sum(Ki) ([N, n_heads])
        denominator = denominator.unsqueeze(1)  # Reshape to [N, 1， n_heads]

        out = numerator / (
            denominator + 1e-8
        )  # Compute attention output ([N, dim_h, n_heads])
        out = self.merge_heads(out)  # Merge heads back to original dimension ([N, dim])
        out = F.dropout(out, p=self.dropout, training=self.training)

        return out
