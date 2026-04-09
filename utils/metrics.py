import torch
import numpy as np
from sklearn.metrics import roc_auc_score


def compute_accuracy(forward_out, labels):
    forward_out = forward_out.detach().cpu()
    labels = labels.detach().cpu()

    if labels.dim() == 2 and labels.size(1) == 1:
        labels = labels.squeeze(1)

    prediction = forward_out.argmax(dim=1)  # Get predicted class labels
    accuracy = (prediction == labels).float().mean().item()  # Compute accuracy

    return accuracy


def compute_roc_auc(forward_out, labels):
    forward_out = forward_out.detach().cpu()
    labels = labels.detach().cpu()

    label_dim = labels.dim()

    # Binary classification ([N] or [N, 1])
    if label_dim == 1 or (label_dim == 2 and labels.size(1) == 1):
        if label_dim == 2:
            labels = labels.squeeze(1)

        # Bindary classifier
        if forward_out.dim() == 1:              # [N]
            scores = torch.sigmoid(forward_out).numpy()
        elif forward_out.size(-1) == 1:         # [N, 1]
            forward_out = forward_out.squeeze(-1)
            scores= torch.sigmoid(forward_out).numpy()
        else:                                   # [N, C]
            scores = torch.softmax(forward_out, dim=-1)[:, 1].numpy()  # Multi-class classifier

        return float(roc_auc_score(labels.numpy(), scores))

    # Multi-label classification
    scores = torch.sigmoid(forward_out).numpy()
    labels = labels.numpy()

    aucs = []
    for col in range(labels.shape[1]):
        col_labels = labels[:, col]

        # Skip if there is only one class
        if np.unique(col_labels).size < 2:
            continue

        col_scores = scores[:, col]
        aucs.append(roc_auc_score(col_labels, col_scores))

    return float(np.mean(aucs))


def compute_metrics(forward_out, labels, metric="accuracy"):
    if metric == "accuracy":
        return compute_accuracy(forward_out, labels)
    elif metric == "roc_auc":
        return compute_roc_auc(forward_out, labels)

    raise ValueError(f"Invalid metric: {metric}")
