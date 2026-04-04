import torch.nn as nn


class GlobalAttention(nn.Module):
    def __init__(self, dim, n_heads):
        super().__init__()
        self.dim = dim
        self.n_heads = n_heads

    def forward(self, x):
        return self.linear(x)