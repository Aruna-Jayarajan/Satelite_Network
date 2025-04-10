import torch
import torch.nn.functional as F

# === Individual Losses ===

def gateway_coverage_loss(gateway_probs, visibility_matrix):
    gw_assignments = torch.sum(gateway_probs * visibility_matrix, dim=0)
    return torch.mean(F.relu(1 - gw_assignments))

def satellite_assignment_loss(gateway_probs, visibility_matrix):
    masked_probs = gateway_probs * visibility_matrix
    sat_sums = torch.sum(masked_probs, dim=1)
    return torch.mean(torch.abs(sat_sums - 1))

def cell_coverage_loss(cell_probs, cell_visibility_matrix):
    masked_probs = cell_probs * cell_visibility_matrix
    cell_coverage = torch.sum(masked_probs, dim=0)
    return torch.mean(F.relu(1 - cell_coverage))

def cell_demand_fairness_loss(cell_probs, cell_visibility_matrix, cell_demands):
    masked_probs = cell_probs * cell_visibility_matrix
    sat_demand = torch.matmul(masked_probs, cell_demands)
    return torch.std(sat_demand)

def satellite_cell_capacity_loss(cell_probs, cell_visibility_matrix, max_cells=25):
    masked_probs = cell_probs * cell_visibility_matrix
    sat_load = torch.sum(masked_probs, dim=1)
    return torch.mean(F.relu(sat_load - max_cells))

def spatial_contiguity_loss(cell_probs, cell_visibility_matrix, cell_coords):
    """
    Encourage spatially coherent cell selection per satellite using soft clustering.
    """
    masked_probs = cell_probs * cell_visibility_matrix  # [num_sats, num_cells]
    sat_total_prob = masked_probs.sum(dim=1, keepdim=True) + 1e-6
    expected_pos = torch.matmul(masked_probs, cell_coords) / sat_total_prob  # [num_sats, 2]

    diff = cell_coords.unsqueeze(0) - expected_pos.unsqueeze(1)  # [num_sats, num_cells, 2]
    sq_dist = (diff ** 2).sum(dim=-1)  # [num_sats, num_cells]
    weighted_sq_dist = masked_probs * sq_dist  # soft spatial variance
    return weighted_sq_dist.mean()


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
    spatial=0.01   # Spatial clustering
):
    device = gateway_logits.device

    gateway_probs = F.softmax(gateway_logits, dim=1)
    cell_probs = torch.sigmoid(cell_logits)

    loss_gw_coverage = gateway_coverage_loss(gateway_probs, visibility_matrix)
    loss_gw_assignment = satellite_assignment_loss(gateway_probs, visibility_matrix)
    loss_cell_coverage = cell_coverage_loss(cell_probs, cell_visibility_matrix)
    loss_fairness = cell_demand_fairness_loss(cell_probs, cell_visibility_matrix, cell_demands)
    loss_capacity = satellite_cell_capacity_loss(cell_probs, cell_visibility_matrix)

    if spatial > 0.0 and cell_coordinates is not None:
        loss_spatial = spatial_contiguity_loss(cell_probs, cell_visibility_matrix, cell_coordinates)
    else:
        loss_spatial = torch.tensor(0.0, device=device)

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
