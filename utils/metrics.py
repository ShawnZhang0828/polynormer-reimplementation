def compute_metrics(forward_out, labels):
    prediction = forward_out.argmax(dim=1)  # Get predicted class labels
    accuracy = (prediction == labels).float().mean().item()  # Compute accuracy
    return accuracy
