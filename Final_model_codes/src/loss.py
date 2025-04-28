import torch
import torch.nn.functional as F

import torch
import torch.nn.functional as F

# === Individual Loss Components ===

def gateway_coverage_loss(gateway_probs, sat_gateway_visibility):
    gw_assignments = torch.sum(gateway_probs * sat_gateway_visibility, dim=0)
    loss = F.relu(1.0 - gw_assignments)
    return torch.mean(loss)

def satellite_assignment_loss(gateway_probs, sat_gateway_visibility):
    masked_probs = gateway_probs * sat_gateway_visibility
    sat_sums = torch.sum(masked_probs, dim=1)
    loss = torch.mean(torch.abs(sat_sums - 1))
    return loss

def cell_coverage_loss(cell_probs, sat_cell_visibility):
    masked_probs = cell_probs * sat_cell_visibility
    cell_coverage = torch.sum(masked_probs, dim=0)
    visibility_mask = (sat_cell_visibility.sum(dim=0) > 0).float()
    uncovered = F.relu(1.0 - cell_coverage)
    loss = torch.sum(uncovered * visibility_mask) / visibility_mask.sum()
    return loss

def satellite_capacity_loss(cell_probs, sat_cell_visibility, max_cells=40):
    masked_probs = cell_probs * sat_cell_visibility
    sat_load = torch.sum(masked_probs, dim=1)
    loss = torch.mean(F.relu(sat_load - max_cells))
    return loss

def satellite_cell_exclusivity_loss(cell_probs, sat_cell_visibility):
    masked_probs = cell_probs * sat_cell_visibility
    sat_sums = torch.sum(masked_probs, dim=1)
    loss = torch.mean(torch.abs(sat_sums - 1))
    return loss

def gateway_exclusivity_loss(gateway_probs):
    total_assignments = gateway_probs.sum(dim=0)
    loss = F.relu(total_assignments - 1.0).mean()
    return loss

def cell_satellite_assignment_loss(cell_sat_probs):
    # Encourage each cell to pick one satellite strongly
    ideal_satellite = torch.argmax(cell_sat_probs, dim=1)
    loss = F.cross_entropy(cell_sat_probs, ideal_satellite)
    return loss

# === Total Loss Aggregator ===

def total_loss(
    gateway_logits,
    cell_logits,
    cell_sat_probs,                # <--- NEW input
    sat_gateway_visibility,
    sat_cell_visibility,
    max_cells=40,
    alpha=10.0,    # Gateway coverage
    beta=1.0,      # Sat-gateway assignment
    delta=100.0,   # Cell coverage
    zeta=0.5,      # Sat capacity
    eta=0.0,       # Sat-cell exclusivity
    gamma=5.0,     # Gateway exclusivity
    omega=2.0      # Cell-satellite matching
):
    device = gateway_logits.device

    gateway_probs = F.softmax(gateway_logits, dim=1)
    cell_probs = torch.sigmoid(cell_logits)

    loss_gw_coverage = gateway_coverage_loss(gateway_probs, sat_gateway_visibility)
    loss_gw_assignment = satellite_assignment_loss(gateway_probs, sat_gateway_visibility)
    loss_cell_coverage = cell_coverage_loss(cell_probs, sat_cell_visibility)
    loss_capacity = satellite_capacity_loss(cell_probs, sat_cell_visibility, max_cells)
    loss_sat_cell_excl = satellite_cell_exclusivity_loss(cell_probs, sat_cell_visibility)
    loss_gw_exclusivity = gateway_exclusivity_loss(gateway_probs)
    loss_cell_sat = cell_satellite_assignment_loss(cell_sat_probs)

    total = (
        alpha * loss_gw_coverage +
        beta * loss_gw_assignment +
        delta * loss_cell_coverage +
        zeta * loss_capacity +
        eta * loss_sat_cell_excl +
        gamma * loss_gw_exclusivity +
        omega * loss_cell_sat         # <--- NEW loss
    )

    breakdown = {
        'gateway_coverage': loss_gw_coverage.item(),
        'sat_assignment': loss_gw_assignment.item(),
        'cell_coverage': loss_cell_coverage.item(),
        'capacity': loss_capacity.item(),
        'sat_cell_exclusivity': loss_sat_cell_excl.item(),
        'gateway_exclusivity': loss_gw_exclusivity.item(),
        'cell_satellite_assignment': loss_cell_sat.item()
    }

    return total, breakdown




'''
def gateway_coverage_loss(gateway_probs, sat_gateway_visibility):
    """
    Ensure every gateway is covered by at least one satellite.
    """
    gw_assignments = torch.sum(gateway_probs * sat_gateway_visibility, dim=0)  # (num_gateways,)
    loss = F.relu(1.0 - gw_assignments)
    return torch.mean(loss)

def satellite_assignment_loss(gateway_probs, sat_gateway_visibility):
    """
    Ensure satellites focus their prediction only on visible gateways.
    Each satellite should ideally 'choose' one gateway strongly.
    """
    masked_probs = gateway_probs * sat_gateway_visibility
    sat_sums = torch.sum(masked_probs, dim=1)  # (num_sats,)
    loss = torch.mean(torch.abs(sat_sums - 1))  # Push to a soft-1-hot
    return loss

def cell_coverage_loss(cell_probs, sat_cell_visibility):
    """
    Ensure all cells are covered by at least one satellite.
    """
    masked_probs = cell_probs * sat_cell_visibility
    cell_coverage = torch.sum(masked_probs, dim=0)  # (num_cells,)

    # Optional debug
    # print("Cell coverage stats — min:", cell_coverage.min().item(), 
    #       "max:", cell_coverage.max().item(), 
    #       "mean:", cell_coverage.mean().item())

    # Only penalize visible cells
    visibility_mask = (sat_cell_visibility.sum(dim=0) > 0).float()

    uncovered = F.relu(1.0 - cell_coverage)
    loss = torch.sum(uncovered * visibility_mask) / visibility_mask.sum()
    return loss

def satellite_cell_capacity_loss(cell_probs, sat_cell_visibility, max_cells=40):
    """
    Penalize satellites for covering too many cells (capacity control).
    """
    masked_probs = cell_probs * sat_cell_visibility
    sat_load = torch.sum(masked_probs, dim=1)  # (num_sats,)
    loss = torch.mean(F.relu(sat_load - max_cells))
    return loss

def satellite_cell_exclusivity_loss(cell_probs, sat_cell_visibility):
    """
    Encourage satellites to not overload; spread out their cell assignments cleanly.
    Softly encourage each satellite's assigned probability to sum near 1.
    """
    masked_probs = cell_probs * sat_cell_visibility
    sat_sums = torch.sum(masked_probs, dim=1)
    loss = torch.mean(torch.abs(sat_sums - 1))
    return loss
def gateway_exclusivity_loss(gateway_probs):
    # Penalize assigning multiple satellites to the same gateway too heavily
    total_assignments = gateway_probs.sum(dim=0)  # [num_gateways]
    loss = F.relu(total_assignments - 1.0).mean()
    return loss

# === Total Loss Aggregator ===

def total_loss(
    gateway_logits,
    cell_logits,
    sat_gateway_visibility,
    sat_cell_visibility,
    max_cells=40,
    alpha=10.0,   # Gateway coverage weight
    beta=1.0,     # Satellite-to-gateway exclusivity
    delta=100.0,  # Cell coverage
    zeta=0.5,     # Satellite capacity control
    eta=0.0,       # Satellite-to-cell exclusivity (optional)
    gamma = 5.0,
    omega=2.0
    
):
    """
    Computes total loss for the system with tunable component weights.
    """
    device = gateway_logits.device

    gateway_probs = F.softmax(gateway_logits, dim=1)  # (num_sats, num_gateways)
    cell_probs = torch.sigmoid(cell_logits)           # (num_sats, num_cells)

    loss_gw_coverage = gateway_coverage_loss(gateway_probs, sat_gateway_visibility)
    loss_gw_assignment = satellite_assignment_loss(gateway_probs, sat_gateway_visibility)
    loss_cell_coverage = cell_coverage_loss(cell_probs, sat_cell_visibility)
    loss_capacity = satellite_cell_capacity_loss(cell_probs, sat_cell_visibility, max_cells)
    loss_sat_cell_excl = satellite_cell_exclusivity_loss(cell_probs, sat_cell_visibility)
    loss_gw_exclusivity = gateway_exclusivity_loss(gateway_probs)


    total = (
        alpha * loss_gw_coverage +
        beta * loss_gw_assignment +
        delta * loss_cell_coverage +
        zeta * loss_capacity +
        eta * loss_sat_cell_excl +
        gamma * loss_gw_exclusivity  # <--- new term
    )
    
    breakdown = {
        'gateway_coverage': loss_gw_coverage.item(),
        'sat_assignment': loss_gw_assignment.item(),
        'cell_coverage': loss_cell_coverage.item(),
        'capacity': loss_capacity.item(),
        'sat_cell_exclusivity': loss_sat_cell_excl.item(),
        'gateway_exclusivity': loss_gw_exclusivity.item()  # 🚀 Add this line
    }


    return total, breakdown
'''