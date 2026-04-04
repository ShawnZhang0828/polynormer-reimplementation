import torch.nn as nn

from models.global_attention import GlobalAttention
from models.local_attention import LocalAttention


class PolyNormer(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, n_heads=8):
        super().__init__()
        self.global_attention = GlobalAttention(hidden_dim, n_heads)
        self.local_attention = LocalAttention(hidden_dim, n_heads)
        self.linear1 = nn.Linear(in_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, out_dim)

    def forward(self, x):
        x = self.linear1(x)
        global_out = self.global_attention(x)
        local_out = self.local_attention(x)
        out = global_out + local_out
        out = self.linear2(out)
        return out
