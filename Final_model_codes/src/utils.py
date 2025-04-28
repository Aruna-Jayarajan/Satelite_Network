# utils.py

import torch

def top_k_accuracy(gateway_probs: torch.Tensor,
                   visibility_matrix: torch.Tensor,
                   true_gateway_indices: torch.Tensor,
                   k: int = 3) -> float:
    """
    Computes top-k accuracy of satellite-to-gateway assignments.
    Args:
        gateway_probs: [num_sats, num_gateways] predicted probabilities
        visibility_matrix: [num_sats, num_gateways] binary visibility mask
        true_gateway_indices: [num_sats] ground-truth gateway indices
        k: number of top predictions to consider

    Returns:
        Top-k accuracy score (0.0 to 1.0)
    """
    masked_probs = gateway_probs * visibility_matrix
    topk_preds = torch.topk(masked_probs, k=k, dim=1).indices

    correct = sum([
        true_gateway_indices[i].item() in topk_preds[i]
        for i in range(len(true_gateway_indices))
    ])

    return correct / len(true_gateway_indices)


def gateway_coverage_metric(gateway_probs: torch.Tensor,
                            visibility_matrix: torch.Tensor) -> float:
    """
    Measures % of gateways that have at least one satellite assigned.

    Returns:
        Gateway coverage ratio (0.0 to 1.0)
    """
    gateway_assignments = torch.sum(gateway_probs * visibility_matrix, dim=0)
    num_covered_gateways = (gateway_assignments >= 1).sum().item()
    total_gateways = visibility_matrix.size(1)

    return num_covered_gateways / total_gateways


def satellite_assignment_accuracy(gateway_probs: torch.Tensor,
                                  visibility_matrix: torch.Tensor,
                                  tolerance: float = 1e-2) -> float:
    """
    Measures % of satellites that have exactly one assigned gateway.

    Returns:
        Satellite assignment accuracy (0.0 to 1.0)
    """
    masked_probs = gateway_probs * visibility_matrix
    assignments_per_satellite = torch.sum(masked_probs, dim=1)

    covered_sats = ((assignments_per_satellite - 1).abs() < tolerance).sum().item()
    total_sats = visibility_matrix.size(0)

    return covered_sats / total_sats
