import torch
import torch.nn as nn

from models.polynormer import PolyNormer


def train_one_epoch(model, data, optimizer, criterion):
    pass


def evaluate(model, data, split):
    pass


def main():
    torch.manual_seed(0)

    edge_index = torch.tensor(
        [[0, 1, 1, 2, 2, 3, 3, 4], [1, 0, 2, 1, 3, 2, 4, 3]], dtype=torch.long
    )  # Example edge index
    x = torch.randn(5, 32)

    model = PolyNormer(
        in_dim=32,
        hidden_dim=64,
        out_dim=3,
        n_local_layers=2,
        n_global_layers=2,
        n_heads=8,
        use_local_attention_network=True,
        use_relu=False,
    )
    out = model(x, edge_index)
    print("output shape:", out.shape)
    print("has nan:", torch.isnan(out).any().item())
    print("has inf:", torch.isinf(out).any().item())

    y = torch.tensor([0, 1, 2, 1, 0], dtype=torch.long)
    loss = torch.nn.functional.cross_entropy(out, y)
    loss.backward()

    print("loss:", loss.item())

    for name, p in model.named_parameters():
        if p.requires_grad and p.grad is None:
            print(name, "grad is None")


if __name__ == "__main__":
    main()
