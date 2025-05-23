import torch
import torch.nn.functional as F

# === Individual Losses ===

def gateway_coverage_loss(gateway_probs, visibility_matrix):
    """Encourage each gateway to be connected by ≥1 satellite"""
    gw_assignments = torch.sum(gateway_probs * visibility_matrix, dim=0)
    return torch.mean(F.relu(1 - gw_assignments))

def satellite_assignment_loss(gateway_probs, visibility_matrix):
    """Encourage each satellite to connect to exactly 1 visible gateway"""
    masked_probs = gateway_probs * visibility_matrix
    sat_sums = torch.sum(masked_probs, dim=1)
    return torch.mean(torch.abs(sat_sums - 1))

def cell_coverage_loss(cell_probs, cell_visibility_matrix):
    """Encourage each cell to be covered by ≥1 satellite"""
    masked_probs = cell_probs * cell_visibility_matrix
    cell_coverage = torch.sum(masked_probs, dim=0)
    return torch.mean(F.relu(1 - cell_coverage))

def cell_demand_fairness_loss(cell_probs, cell_visibility_matrix, cell_demands):
    """Reduce variation in satellite load based on demand"""
    masked_probs = cell_probs * cell_visibility_matrix
    sat_demand = torch.matmul(masked_probs, cell_demands)
    return torch.std(sat_demand)

def satellite_cell_capacity_loss(cell_probs, cell_visibility_matrix, max_cells=25):
    """Prevent satellites from handling too many cells"""
    masked_probs = cell_probs * cell_visibility_matrix
    sat_load = torch.sum(masked_probs, dim=1)
    return torch.mean(F.relu(sat_load - max_cells))

def spatial_contiguity_loss(cell_probs, cell_visibility_matrix, cell_coords):
    """Encourage satellites to serve spatially close cells"""
    device = cell_probs.device
    masked_probs = cell_probs * cell_visibility_matrix
    total_penalty, count = 0.0, 0

    for i in range(masked_probs.size(0)):
        assigned = (masked_probs[i] > 0.5).nonzero(as_tuple=True)[0]
        if assigned.size(0) < 2:
            continue
        coords = cell_coords[assigned]
        dist = torch.cdist(coords.unsqueeze(0), coords.unsqueeze(0)).squeeze(0)
        total_penalty += dist.mean()
        count += 1

    return torch.tensor(0.0, device=device) if count == 0 else total_penalty / count


# === Total Loss Aggregator ===

def total_loss(
    gateway_logits,
    cell_logits,
    visibility_matrix,
    cell_visibility_matrix,
    cell_demands,
    cell_coordinates=None,
    alpha=6.0,     # Gateway coverage
    beta=1.0,      # Sat-gateway exclusivity
    gamma=0.3,     # Fair demand distribution
    delta=10.0,    # Cell coverage
    zeta=0.5,      # Overload penalty
    spatial=0.0    # Spatial clustering
):
    device = gateway_logits.device

    # Convert to probabilities
    gateway_probs = F.softmax(gateway_logits, dim=1)
    cell_probs = torch.sigmoid(cell_logits)


    # === Compute each loss component ===
    loss_gw_coverage = gateway_coverage_loss(gateway_probs, visibility_matrix)
    loss_gw_assignment = satellite_assignment_loss(gateway_probs, visibility_matrix)
    loss_cell_coverage = cell_coverage_loss(cell_probs, cell_visibility_matrix)
    loss_fairness = cell_demand_fairness_loss(cell_probs, cell_visibility_matrix, cell_demands)
    loss_capacity = satellite_cell_capacity_loss(cell_probs, cell_visibility_matrix)

    if spatial > 0.0 and cell_coordinates is not None:
        loss_spatial = spatial_contiguity_loss(cell_probs, cell_visibility_matrix, cell_coordinates)
    else:
        loss_spatial = torch.tensor(0.0, device=device)

    # === Weighted sum ===
    total = (
        alpha * loss_gw_coverage +
        beta * loss_gw_assignment +
        gamma * loss_fairness +
        delta * loss_cell_coverage +
        zeta * loss_capacity +
        spatial * loss_spatial
    )

    breakdown = {
        'gateway_coverage': loss_gw_coverage.item(),
        'sat_assignment': loss_gw_assignment.item(),
        'cell_coverage': loss_cell_coverage.item(),
        'demand_fairness': loss_fairness.item(),
        'capacity': loss_capacity.item(),
        'spatial_contiguity': loss_spatial.item()
    }

    return total, breakdown
