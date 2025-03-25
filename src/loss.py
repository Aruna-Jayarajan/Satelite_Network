import torch
import numpy as np


def compute_mse_loss(pred_matrix, true_matrix):
    """
    Compute Mean Squared Error between predicted and true binary matrices.
    """
    pred_tensor = torch.tensor(pred_matrix, dtype=torch.float32)
    true_tensor = torch.tensor(true_matrix, dtype=torch.float32)
    return torch.mean((pred_tensor - true_tensor) ** 2)


def get_binary_predictions(preds_gnn):
    """
    Convert logits to binary predictions (top-1 one-hot encoding).
    """
    top_gateways = torch.argmax(preds_gnn, dim=1)
    binary_matrix = np.zeros((preds_gnn.shape[0], preds_gnn.shape[1]))
    for i, gw in enumerate(top_gateways.cpu().numpy()):
        binary_matrix[i, gw] = 1
    return binary_matrix


def top_k_accuracy(preds, labels, k):
    """
    Compute top-k accuracy.
    """
    topk = torch.topk(preds, k, dim=1).indices
    return (topk == labels.unsqueeze(1)).any(dim=1).float().mean().item()
