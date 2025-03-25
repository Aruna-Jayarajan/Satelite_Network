import torch
import numpy as np
import torch.nn.functional as F
import torch.nn as nn
from torch.nn import CrossEntropyLoss



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


# ===============================
# Main Classification Loss
# ===============================
def classification_loss(preds, targets, label_smoothing=0.0):
    """
    Standard cross-entropy loss with optional label smoothing.
    """
    if label_smoothing > 0:
        n_classes = preds.size(1)
        smooth_labels = (1 - label_smoothing) * F.one_hot(targets, num_classes=n_classes).float()
        smooth_labels += label_smoothing / n_classes
        log_probs = F.log_softmax(preds, dim=1)
        return -(smooth_labels * log_probs).sum(dim=1).mean()
    else:
        return CrossEntropyLoss()(preds, targets)


# ===============================
# Global Constraint Loss
# ===============================
def compute_global_gateway_loss(preds, k=1):
    """
    Computes a global loss to encourage every gateway to be assigned at least one satellite.
    """
    if k == 1:
        top_k = torch.argmax(preds, dim=1)
    else:
        top_k = torch.topk(preds, k=k, dim=1).indices

    binary_matrix = torch.zeros_like(preds)
    if k == 1:
        binary_matrix[torch.arange(preds.size(0)), top_k] = 1
    else:
        for i in range(preds.size(0)):
            binary_matrix[i, top_k[i]] = 1

    per_gateway_usage = binary_matrix.sum(dim=0)
    missing_gateways = (per_gateway_usage == 0).float()
    return missing_gateways.sum() / preds.size(1)  # Normalize by num gateways


# ===============================
# Combined Loss Function
# ===============================
def total_loss_fn(preds, targets, lambda_global=0.2, label_smoothing=0.0):
    """
    Total loss = classification loss + lambda * global loss
    """
    loss_cls = classification_loss(preds, targets, label_smoothing)
    loss_global = compute_global_gateway_loss(preds, k=1)
    return loss_cls + lambda_global * loss_global, loss_cls, loss_global
