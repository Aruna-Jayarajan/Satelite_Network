import torch
import torch.nn.functional as F
import importlib
import src.loss

importlib.reload(src.loss)

from src.loss import total_loss

def compute_gateway_coverage(gateway_probs, sat_gateway_visibility):
    assignments = torch.sum(gateway_probs * sat_gateway_visibility, dim=0)
    covered = (assignments >= 1).sum().item()
    return covered / sat_gateway_visibility.shape[1]

def compute_cell_coverage(cell_probs, sat_cell_visibility):
    assignments = torch.sum(cell_probs * sat_cell_visibility, dim=0)
    covered = (assignments >= 1).sum().item()
    return covered / sat_cell_visibility.shape[1]

def compute_satellite_demand(cell_probs, sat_cell_visibility, cell_demands):
    masked_probs = cell_probs * sat_cell_visibility
    return torch.matmul(masked_probs, cell_demands).squeeze(-1)

def get_bin_key(lat, lon, bin_size=1.0):
    return (round(lat / bin_size), round(lon / bin_size))

import random
import torch

def randomly_mask_satellites(data, drop_prob=0.15):
    """
    Randomly mask/drop a fraction of satellites from the graph during training.

    Args:
        data (HeteroData): The graph data.
        drop_prob (float): Fraction of satellites to randomly mask.

    Returns:
        data (HeteroData): Updated graph with masked satellites.
        mask (Tensor): 1 if satellite is active, 0 if masked.
    """
    num_sats = data['sat'].num_nodes
    device = data['sat'].x.device

    # Random mask decision
    mask = torch.rand(num_sats, device=device) > drop_prob

    # Mask satellite node features
    data['sat'].x = data['sat'].x * mask.unsqueeze(1)

    # Mask satellite outgoing edges
    for (src_type, rel, dst_type), edge_index in data.edge_index_dict.items():
        if src_type == 'sat':
            keep = mask[edge_index[0]]
            data[(src_type, rel, dst_type)].edge_index = edge_index[:, keep]
        if dst_type == 'sat':
            keep = mask[edge_index[1]]
            data[(src_type, rel, dst_type)].edge_index = edge_index[:, keep]

    return data, mask

def train_until_converged(
    model,
    data,
    optimizer,
    aux_inputs,
    loss_weights,
    epochs=30,
    patience=10,
    target_gw_coverage=1.0,
    target_cell_coverage=1.0,
    track_coverage=True,
    verbose=True,
    return_probs=False,
    memory_dict=None,
    sat_positions=None,
    bin_size=1.0
):
    """
    Training loop for SatGatewayCellGNN.
    """

    best_loss = float('inf')
    patience_counter = 0

    history = {
        'loss': [],
        'gateway_coverage': [],
        'cell_coverage': [],
        'sat_demand': [],
    }

    final_outputs = None

    # === Initialize satellite memory ===
    sat_memory_input = torch.zeros(data['sat'].x.shape[0], model.hidden_dim, device=data['sat'].x.device)
    if memory_dict is not None and sat_positions is not None:
        for i, (lat, lon) in enumerate(sat_positions):
            bin_key = get_bin_key(lat, lon, bin_size)
            if bin_key in memory_dict:
                sat_memory_input[i] = memory_dict[bin_key]

    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad()

        # Build visibility matrices properly
        visibility_matrices = {
            'sat_gateway': aux_inputs['sat_gateway'],
            'sat_cell': aux_inputs['sat_cell'],
            'cell_sat': aux_inputs['cell_sat'],
        }

        output = model(data, visibility_matrices=visibility_matrices, sat_memory=sat_memory_input)

        loss_val, _ = total_loss(
            gateway_logits=output['sat_gateway_logits'],
            cell_logits=output['sat_cell_logits'],
            sat_gateway_visibility=aux_inputs['sat_gateway'],   # 🚀 Fixed key
            sat_cell_visibility=aux_inputs['sat_cell'],         # 🚀 Fixed key
            **loss_weights
        )


        loss_val.backward()
        optimizer.step()
        history['loss'].append(loss_val.item())

        if track_coverage:
            gw_cov = compute_gateway_coverage(output['sat_gateway_probs'], aux_inputs['sat_gateway'])
            cell_cov = compute_cell_coverage(output['sat_cell_probs'], aux_inputs['sat_cell'])

            history['gateway_coverage'].append(gw_cov)
            history['cell_coverage'].append(cell_cov)

            sat_demand = compute_satellite_demand(
                output['sat_cell_probs'],
                aux_inputs['sat_cell'],
                aux_inputs['cell_demands']
            ).detach().cpu().numpy()
            history['sat_demand'].append(sat_demand)


            if verbose:
                print(f"[Epoch {epoch:03d}] Loss: {loss_val:.4f} | "
                      f"GW-Coverage: {gw_cov*100:.2f}% | Cell-Coverage: {cell_cov*100:.2f}%")

            final_outputs = output

            if gw_cov >= target_gw_coverage and cell_cov >= target_cell_coverage:
                if verbose:
                    print(f"Target coverage reached at epoch {epoch}")
                break

        # === Early stopping ===
        if loss_val.item() < best_loss - 1e-4:
            best_loss = loss_val.item()
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= patience:
            if verbose:
                print(f"Early stopping at epoch {epoch}")
            break

    # === Update Memory ===
    if memory_dict is not None and sat_positions is not None and final_outputs is not None:
        from collections import defaultdict
        bin_sums = defaultdict(lambda: torch.zeros(model.hidden_dim, device=sat_memory_input.device))
        bin_counts = defaultdict(int)

        for i, (lat, lon) in enumerate(sat_positions):
            bin_key = get_bin_key(lat, lon, bin_size)
            bin_sums[bin_key] += final_outputs['sat_memory_out'][i].detach()
            bin_counts[bin_key] += 1

        for bin_key in bin_sums:
            memory_dict[bin_key] = bin_sums[bin_key] / bin_counts[bin_key]

    if verbose and track_coverage:
        print(f"\nFinal GW Coverage: {gw_cov*100:.2f}% | Final Cell Coverage: {cell_cov*100:.2f}%")

    if return_probs:
        return model, history, final_outputs
    else:
        return model, history, None
