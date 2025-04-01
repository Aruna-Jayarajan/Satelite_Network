import torch

def coverage_loss(pred_gateway_logits, gateway_mask=None, cell_coverage_matrix=None):
    """
    Penalize if any gateway or cell is not covered.
    
    Args:
        pred_gateway_logits: Tensor [N, G] - logits for gateway assignment
        gateway_mask: Optional Tensor [N, G] - binary (1 if visible)
        cell_coverage_matrix: Optional Tensor [C] - 1 if a cell is covered

    Returns:
        Scalar tensor loss
    """
    # Gateway coverage: ensure every gateway is selected by at least one satellite
    top_gateway = torch.argmax(pred_gateway_logits, dim=1)
    num_gateways = pred_gateway_logits.size(1)
    cover_vector = torch.zeros(num_gateways, device=pred_gateway_logits.device)
    cover_vector.scatter_add_(0, top_gateway, torch.ones_like(top_gateway, dtype=torch.float))
    missing_gateways = (cover_vector == 0).float()
    gw_loss = missing_gateways.sum() / num_gateways

    # Optional: Cell coverage loss
    if cell_coverage_matrix is not None:
        missing_cells = (cell_coverage_matrix == 0).float()
        cell_loss = missing_cells.mean()
    else:
        cell_loss = 0.0

    return gw_loss + cell_loss


def fairness_loss(sat_gateway_counts, sat_cell_counts):
    """
    Penalize unfair distribution across satellites.
    
    Args:
        sat_gateway_counts: Tensor [N] - number of gateways per satellite
        sat_cell_counts: Tensor [N] - number of cells per satellite

    Returns:
        Scalar tensor loss
    """
    gw_mean = sat_gateway_counts.mean()
    cell_mean = sat_cell_counts.mean()

    gw_var = ((sat_gateway_counts - gw_mean) ** 2).mean()
    cell_var = ((sat_cell_counts - cell_mean) ** 2).mean()

    return gw_var + cell_var


def balance_loss(gateway_cell_distribution):
    """
    Penalize gateways with highly imbalanced cell distribution across satellites.

    Args:
        gateway_cell_distribution: Tensor [G, N] - number of cells assigned via each satellite for each gateway

    Returns:
        Scalar tensor loss
    """
    per_gateway_var = torch.var(gateway_cell_distribution.float(), dim=1).mean()
    return per_gateway_var


def switch_penalty(current_gateways, prev_gateways, current_cells, prev_cells):
    """
    Penalize excessive reassignment of gateways and cells.

    Args:
        current_gateways: Tensor [N, G] - current assignment (binary)
        prev_gateways: Tensor [N, G] - previous assignment (binary)
        current_cells: Tensor [N, C] - current cell assignment (binary)
        prev_cells: Tensor [N, C] - previous cell assignment (binary)

    Returns:
        Scalar tensor loss
    """
    switch_gw = (current_gateways != prev_gateways).float().sum(dim=1).mean()
    switch_cells = (current_cells != prev_cells).float().sum(dim=1).mean()
    return switch_gw + switch_cells


def overlap_penalty(cell_assignments, neighbor_assignments, gateway_ids):
    """
    Penalize cell assignment overlaps through same gateway among neighbors.

    Args:
        cell_assignments: Dict[int, Set[int]] - {sat_id: set of (cell_id, gateway_id)}
        neighbor_assignments: Dict[int, List[Set[int]]] - {sat_id: list of neighbor cell sets}
        gateway_ids: List[int] - list of gateway IDs for each satellite

    Returns:
        Scalar float penalty
    """
    penalty = 0
    total_checks = 0

    for s_id, own_assignments in cell_assignments.items():
        gw_set = set(gateway_ids[s_id])
        for neighbor_cells in neighbor_assignments.get(s_id, []):
            overlaps = own_assignments.intersection(neighbor_cells)
            if overlaps:
                penalty += len(overlaps)
            total_checks += 1

    return torch.tensor(penalty / max(1, total_checks), dtype=torch.float)


def total_loss(pred_gateway_logits,
               cell_coverage_matrix,
               sat_gateway_counts,
               sat_cell_counts,
               gateway_cell_distribution,
               current_gateways,
               prev_gateways,
               current_cells,
               prev_cells,
               cell_assignments,
               neighbor_assignments,
               gateway_ids,
               lambda_coverage=1.0,
               lambda_fairness=1.0,
               lambda_balance=1.0,
               lambda_switch=1.0,
               lambda_overlap=1.0):
    """
    Composite loss function.

    Returns:
        Combined scalar tensor loss
    """
    l1 = coverage_loss(pred_gateway_logits, cell_coverage_matrix=cell_coverage_matrix)
    l2 = fairness_loss(sat_gateway_counts, sat_cell_counts)
    l3 = balance_loss(gateway_cell_distribution)
    l4 = switch_penalty(current_gateways, prev_gateways, current_cells, prev_cells)
    l5 = overlap_penalty(cell_assignments, neighbor_assignments, gateway_ids)

    return (
        lambda_coverage * l1 +
        lambda_fairness * l2 +
        lambda_balance * l3 +
        lambda_switch * l4 +
        lambda_overlap * l5
    )
