# utils.py

import torch

def top_k_accuracy(gateway_probs, visibility_matrix, true_gateway_indices, k=3):
    """
    gateway_probs: [num_sats, num_gateways], predicted assignment probabilities
    visibility_matrix: [num_sats, num_gateways], binary visibility
    true_gateway_indices: [num_sats], ground truth gateway indices
    k: top-k value
    """
    # Mask predictions with visibility
    masked_probs = gateway_probs * visibility_matrix
    topk_preds = torch.topk(masked_probs, k=k, dim=1).indices

    # Check if true gateway is within top-k predictions
    correct = sum([
        true_gateway_indices[i] in topk_preds[i] 
        for i in range(len(true_gateway_indices))
    ])

    accuracy = correct / len(true_gateway_indices)
    return accuracy

def gateway_coverage_metric(gateway_probs, visibility_matrix):
    """
    Returns the percentage of gateways that have at least one satellite assigned.
    """
    gateway_assignments = torch.sum(gateway_probs * visibility_matrix, dim=0)
    num_covered_gateways = (gateway_assignments >= 1).sum().item()
    total_gateways = visibility_matrix.size(1)

    coverage_ratio = num_covered_gateways / total_gateways
    return coverage_ratio

def satellite_coverage_metric(gateway_probs, visibility_matrix):
    """
    Returns the percentage of satellites that have exactly one gateway assigned.
    """
    masked_probs = gateway_probs * visibility_matrix
    assignments_per_satellite = torch.sum(masked_probs, dim=1)

    covered_sats = ((assignments_per_satellite - 1).abs() < 1e-2).sum().item()
    total_sats = visibility_matrix.size(0)

    satellite_coverage_ratio = covered_sats / total_sats
    return satellite_coverage_ratio
