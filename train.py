import argparse
import torch
import torch.nn.functional as F
from torch.optim import Adam

from models.polynormer import Polynormer
from utils.io import save_checkpoint
from utils.metrics import compute_metrics
from utils.seed import set_seed
from utils.data_loaders import load_dataset
from config import get_default_config


def strToBool(value):
    # Helper function to parse boolean command-line arguments
    if isinstance(value, bool):
        return value
    value = value.lower()
    if value in ("true", "t"):
        return True
    elif value in ("false", "f"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


def parse_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--dataset",
        type=str,
        default="Cora",
        help="Name of the dataset to use (e.g., Cora, Citeseer)",
    )
    parser.add_argument(
        "--hidden_dim", type=int, default=None, help="Hidden dimension size"
    )
    parser.add_argument(
        "--n_local_layers",
        type=int,
        default=None,
        help="Numberof local attention layers",
    )
    parser.add_argument(
        "--n_global_layers",
        type=int,
        default=None,
        help="Number of global attention layers",
    )
    parser.add_argument(
        "--n_local_heads",
        type=int,
        default=None,
        help="Number of heads for local attention",
    )
    parser.add_argument(
        "--n_global_heads",
        type=int,
        default=None,
        help="Number of heads for global attention",
    )
    parser.add_argument(
        "--warm_up_epochs", type=int, default=None, help="Number of warm-up epochs"
    )
    parser.add_argument(
        "--local_to_global_epochs",
        type=int,
        default=None,
        help="Epoch to switch from local to global training",
    )
    parser.add_argument(
        "--use_relu",
        type=strToBool,
        default=None,
        help="Whether to use ReLU activation",
    )
    parser.add_argument(
        "--use_local_attention_network",
        type=strToBool,
        default=None,
        help="Whether to use attention network in local layers",
    )
    parser.add_argument(
        "--lr", type=float, default=None, help="Learning rate for the optimizer"
    )
    parser.add_argument(
        "--dropout", type=float, default=None, help="Dropout rate for attention layers"
    )

    return parser.parse_args()


def merge_args_with_config(args, config):
    args = vars(args)

    for key, value in args.items():
        if value is not None:
            config[key] = value

    return config


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


def train_one_epoch(model, data, train_idx, optimizer, device, freeze_global=False):
    model.train()
    optimizer.zero_grad()

    x = data.x.to(device)
    edge_index = data.edge_index.to(device)
    y = data.y.to(device)

    out = model(x, edge_index, freeze_global=freeze_global)
    # Determine output and label from the training split
    train_out, train_y = determine_output_and_label(out, y, train_idx)
    loss = F.cross_entropy(train_out, train_y)
    loss.backward()

    optimizer.step()

    train_accuracy = compute_metrics(train_out.detach(), train_y)
    return loss.item(), train_accuracy


@torch.no_grad()
def evaluate(model, data, device, freeze_global=False):
    model.eval()

    x = data.x.to(device)
    edge_index = data.edge_index.to(device)
    y = data.y.to(device)

    train_idx, val_idx, test_idx = get_split_idx(data)

    out = model(x, edge_index, freeze_global=freeze_global)

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
    # Get default configuration and merge with command-line arguments
    configuration = get_default_config()
    args = parse_arguments()
    configuration = merge_args_with_config(args, configuration)

    # Set random seed for reproducibility
    set_seed(configuration["seed"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load dataset and get train/val/test indices
    data, in_dim, out_dim = load_dataset(
        configuration["dataset"], root="data", seed=configuration["seed"]
    )
    train_idx, val_idx, test_idx = get_split_idx(data)
    train_idx = train_idx.to(device) if torch.is_tensor(train_idx) else None
    val_idx = val_idx.to(device) if torch.is_tensor(val_idx) else None
    test_idx = test_idx.to(device) if torch.is_tensor(test_idx) else None

    # Initialize model and optimizer
    model = Polynormer(
        in_dim=in_dim,
        hidden_dim=configuration["hidden_dim"],
        out_dim=out_dim,
        n_local_layers=configuration["n_local_layers"],
        n_global_layers=configuration["n_global_layers"],
        n_local_heads=configuration["n_local_heads"],
        n_global_heads=configuration["n_global_heads"],
        dropout=configuration["dropout"],
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

    for epoch in range(1, configuration["warm_up_epochs"] + 1):
        train_loss, train_acc = train_one_epoch(
            model, data, train_idx, optimizer, device, freeze_global=True
        )

    metrics = evaluate(model, data, device, freeze_global=True)
    print(
        f"Finish warm-up with {epoch:03d} epochs | ",
        f"Train loss: {train_loss:.4f}, Train acc: {train_acc:.4f} | ",
        f"Val acc: {metrics['val_acc']:.4f} | ",
        f"Test acc: {metrics['test_acc']:.4f}",
    )

    # Training loop
    for epoch in range(1, configuration["local_to_global_epochs"] + 1):
        train_loss, train_acc = train_one_epoch(
            model, data, train_idx, optimizer, device, freeze_global=False
        )
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
