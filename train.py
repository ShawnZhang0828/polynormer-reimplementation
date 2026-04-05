import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam

from models.polynormer import Polynormer
from utils.io import save_checkpoint, load_checkpoint
from utils.metrics import compute_metrics
from utils.seed import set_seed
from utils.data_loaders import load_planetoid_dataset
from config import get_default_config


def get_split_idx(data):
    # Get train, val, and test indices from the data object (mask or index).
    if (
        hasattr(data, "train_mask")
        and hasattr(data, "val_mask")
        and hasattr(data, "test_mask")
    ):
        return data.train_mask, data.val_mask, data.test_mask

    if hasattr(data, "split_idx"):
        split_idx = data.split_idx
        return split_idx["train"], split_idx["valid"], split_idx["test"]

    raise ValueError("DATA ERROR: No valid split indices found in the data object.")


def determine_output_and_label(outputs, y, split_idx):
    # Determine the output and label for the given split index (mask or index).
    return outputs[split_idx], y[split_idx]


def train_one_epoch(model, data, optimizer, device):
    model.train()
    optimizer.zero_grad()

    x = data.x.to(device)
    edge_index = data.edge_index.to(device)
    y = data.y.to(device)

    out = model(x, edge_index)
    # Determine output and label from the training split
    train_out, train_y = determine_output_and_label(out, y, data.train_mask)
    loss = F.cross_entropy(train_out, train_y)
    loss.backward()

    optimizer.step()

    train_accuracy = compute_metrics(train_out.detach(), train_y)
    return loss.item(), train_accuracy


@torch.no_grad()
def evaluate(model, data, device):
    model.eval()

    x = data.x.to(device)
    edge_index = data.edge_index.to(device)
    y = data.y.to(device)

    train_idx, val_idx, test_idx = get_split_idx(data)

    out = model(x, edge_index)

    # Extract output and label for each split
    train_out, train_y = determine_output_and_label(out, y, train_idx)
    val_out, val_y = determine_output_and_label(out, y, val_idx)
    test_out, test_y = determine_output_and_label(out, y, test_idx)

    return {
        "train_loss": F.cross_entropy(train_out, train_y).item(),
        "train_acc": compute_metrics(train_out, train_y),
        "val_loss": F.cross_entropy(val_out, val_y).item(),
        "val_acc": compute_metrics(val_out, val_y),
        "test_loss": F.cross_entropy(test_out, test_y).item(),
        "test_acc": compute_metrics(test_out, test_y),
    }


def main():
    # Get default configuration
    configuration = get_default_config()

    # Set random seed for reproducibility
    set_seed(configuration["seed"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # TEMP: Load dataset (Cora) and get input/output dimensions
    data, in_dim, out_dim = load_planetoid_dataset("data", "Cora")

    # Initialize model and optimizer
    model = Polynormer(
        in_dim=in_dim,
        hidden_dim=configuration["hidden_dim"],
        out_dim=out_dim,
        n_local_layers=configuration["n_local_layers"],
        n_global_layers=configuration["n_global_layers"],
        n_heads=configuration["n_heads"],
        use_relu=configuration["use_relu"],
        use_local_attention_network=configuration["use_local_attention_network"],
    ).to(device)

    optimizer = Adam(
        model.parameters(),
        lr=configuration["lr"],
        weight_decay=configuration["weight_decay"],
    )

    best_val_acc = -1
    best_test_acc = -1

    # Training loop
    for epoch in range(1, configuration["epochs"] + 1):
        train_loss, train_acc = train_one_epoch(model, data, optimizer, device)
        metrics = evaluate(model, data, device)

        if metrics["val_acc"] > best_val_acc:
            best_val_acc = metrics["val_acc"]
            best_test_acc = metrics["test_acc"]
            save_checkpoint(
                configuration["checkpoint_path"], model, optimizer, epoch, best_val_acc
            )

        print(
            f"Epoch {epoch:03d} | ",
            f"Train loss: {train_loss:.4f}, Train acc: {train_acc:.4f} | ",
            f"Val acc: {metrics['val_acc']:.4f} | ",
            f"Test acc: {metrics['test_acc']:.4f}",
        )

    print(f"Best validation accuracy: {best_val_acc}")
    print(f"Best test accuracy: {best_test_acc}")


if __name__ == "__main__":
    main()
