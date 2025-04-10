import torch
import torch.nn.functional as F
from src.loss3 import total_loss


def compute_gateway_coverage(gateway_probs, visibility_matrix):
    assignments = torch.sum(gateway_probs * visibility_matrix, dim=0)
    covered = (assignments >= 1).sum().item()
    return covered / visibility_matrix.shape[1]

def compute_cell_coverage(cell_probs, cell_visibility_matrix):
    assignments = torch.sum(cell_probs * cell_visibility_matrix, dim=0)
    covered = (assignments >= 1).sum().item()
    return covered / cell_visibility_matrix.shape[1]

def compute_satellite_demand(cell_probs, visibility_matrix, cell_demands):
    masked_probs = cell_probs * visibility_matrix
    return torch.matmul(masked_probs, cell_demands).squeeze(-1)

def train_until_converged(
    model,
    data,
    optimizer,
    aux_inputs,
    loss_weights,
    epochs=200,
    patience=20,
    target_gw_coverage=1.0,
    target_cell_coverage=1.0,
    track_coverage=True,
    verbose=True,
    return_probs=False,
):
    best_loss = float('inf')
    patience_counter = 0

    history = {
        'loss': [],
        'gateway_coverage': [],
        'cell_coverage': [],
        'sat_demand': [],
    }

    final_gw_probs, final_cell_probs = None, None

    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad()

        # === Forward ===
        gw_logits_all, cell_logits_all, _ = model(
            data,
            cell_visibility_matrix=aux_inputs['cell_visibility_matrix'],
            cell_demands=aux_inputs['cell_demands']
        )
        gateway_logits = gw_logits_all[-1]
        cell_logits = cell_logits_all[-1]

        # === Compute loss ===
        loss_val, _ = total_loss(
            gateway_logits=gateway_logits,
            cell_logits=cell_logits,
            visibility_matrix=aux_inputs['visibility_matrix'],
            cell_visibility_matrix=aux_inputs['cell_visibility_matrix'],
            cell_demands=aux_inputs['cell_demands'],
            cell_coordinates=aux_inputs.get('cell_coordinates', None),
            **loss_weights
        )

        loss_val.backward()
        optimizer.step()
        history['loss'].append(loss_val.item())

        if track_coverage:
            # === Convert logits to probabilities ===
            gateway_probs = torch.softmax(gateway_logits, dim=1)

            # === Mutually Exclusive Cell Assignment ===
            cell_probs = F.softmax(cell_logits.transpose(0, 1), dim=1).transpose(0, 1)

            gw_cov = compute_gateway_coverage(gateway_probs, aux_inputs['visibility_matrix'])
            cell_cov = compute_cell_coverage(cell_probs, aux_inputs['cell_visibility_matrix'])

            history['gateway_coverage'].append(gw_cov)
            history['cell_coverage'].append(cell_cov)

            sat_demand = compute_satellite_demand(
                cell_probs,
                aux_inputs['cell_visibility_matrix'],
                aux_inputs['cell_demands']
            ).detach().cpu().numpy()
            history['sat_demand'].append(sat_demand)

            if verbose:
                print(f"[Epoch {epoch:03d}] Loss: {loss_val:.4f} | "
                      f"GW-Coverage: {gw_cov*100:.2f}% | Cell-Coverage: {cell_cov*100:.2f}%")

            final_gw_probs = gateway_probs.detach()
            final_cell_probs = cell_probs.detach()

            if gw_cov >= target_gw_coverage and cell_cov >= target_cell_coverage:
                if verbose:
                    print(f"Full coverage reached at epoch {epoch}")
                break

        # === Early Stopping ===
        if loss_val.item() < best_loss - 1e-4:
            best_loss = loss_val.item()
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= patience:
            if verbose:
                print(f"Early stopping triggered at epoch {epoch}")
            break

    if verbose and track_coverage:
        print(f"Final GW Coverage: {gw_cov*100:.2f}% | Final Cell Coverage: {cell_cov*100:.2f}%")

    if return_probs:
        return model, history, final_gw_probs, final_cell_probs
    else:
        return model, history
