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

import torch.nn.functional as F

def compute_global_gateway_loss(preds, k=1):
    """
    Differentiable global constraint loss:
    Encourage each gateway to receive some probability mass.
    """
    probs = F.softmax(preds, dim=1)  # [batch, gateways]

    # Sum of probabilities assigned to each gateway
    per_gateway_usage = probs.sum(dim=0)  # [gateways]

    # Penalize gateways with near-zero assignment
    loss = torch.mean(torch.clamp(1.0 - per_gateway_usage, min=0.0))

    return loss

def compute_gateway_fairness_loss(preds):
    """
    Encourage all gateways to be used approximately equally.
    """
    probs = F.softmax(preds, dim=1)
    usage = probs.sum(dim=0)  # total usage of each gateway

    fairness_target = usage.mean()
    loss = torch.mean((usage - fairness_target) ** 2)  # MSE from mean

    return loss

def compute_satellite_coverage_loss(preds):
    """
    Encourage each satellite to connect to at least one gateway.
    """
    probs = F.softmax(preds, dim=1)
    max_prob = probs.max(dim=1).values  # [num_sats]
    loss = torch.mean(torch.clamp(0.5 - max_prob, min=0.0))  # penalize weak assignments

    return loss

def compute_satellite_fairness_loss(preds):
    """
    Encourage satellites to spread assignments more evenly across gateways.
    High entropy = better fairness.
    """
    probs = F.softmax(preds, dim=1)
    entropy = -torch.sum(probs * torch.log(probs + 1e-8), dim=1)  # [num_sats]
    loss = -entropy.mean()  # we want *high* entropy → minimize negative

    return loss


# ===============================
# Combined Loss Function
# ===============================
def total_loss_fn(preds, targets=None, lambda_global=1.0, lambda_gateway_fair=0.1,
                  lambda_sat_coverage=0.1, lambda_sat_fair=0.1):
    """
    Combined loss function with:
    - Global gateway coverage
    - Gateway usage fairness
    - Satellite coverage
    - Satellite fairness
    """
    loss_global = compute_global_gateway_loss(preds)
    loss_gateway_fair = compute_gateway_fairness_loss(preds)
    loss_sat_coverage = compute_satellite_coverage_loss(preds)
    loss_sat_fair = compute_satellite_fairness_loss(preds)

    total_loss = (
        lambda_global * loss_global +
        lambda_gateway_fair * loss_gateway_fair +
        lambda_sat_coverage * loss_sat_coverage +
        lambda_sat_fair * loss_sat_fair
    )

    return total_loss, loss_global, total_loss  # you can break out individual components too
