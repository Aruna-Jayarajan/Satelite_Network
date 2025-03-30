import torch
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss
from collections import defaultdict


# === 1. Classification Loss (with optional label smoothing) ===
def classification_loss(preds, targets, label_smoothing=0.0):
    if label_smoothing > 0:
        num_classes = preds.size(1)
        smooth_targets = (1 - label_smoothing) * F.one_hot(targets, num_classes=num_classes).float()
        smooth_targets += label_smoothing / num_classes
        log_probs = F.log_softmax(preds, dim=1)
        return -(smooth_targets * log_probs).sum(dim=1).mean()
    else:
        return CrossEntropyLoss()(preds, targets)


# === 2. Global Gateway Usage Loss ===
def compute_global_gateway_loss(preds, k=1):
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
    return missing_gateways.sum() / preds.size(1)


# === 3. Entropy Regularization ===
def entropy_regularization(preds, temperature=1.0):
    probs = F.softmax(preds / temperature, dim=1)
    entropy = -torch.sum(probs * torch.log(probs + 1e-8), dim=1)
    return entropy.mean()


# === 4. Greedy Cell Assignment via Gateways ===
def assign_cells_via_gateways(sat_to_gw, gw_to_cells):
    sat_to_cells = defaultdict(list)
    assigned_cells = set()

    # Reverse mapping: gateway → list of satellites
    gw_to_sats = defaultdict(list)
    for sat, gw in sat_to_gw.items():
        gw_to_sats[gw].append(sat)

    for gw, sats in gw_to_sats.items():
        cells = gw_to_cells.get(gw, [])
        if not cells or not sats:
            continue

        chunks = split_evenly(cells, len(sats))
        for sat, chunk in zip(sats, chunks):
            for cell in chunk:
                if cell not in assigned_cells:
                    sat_to_cells[sat].append(cell)
                    assigned_cells.add(cell)

    return sat_to_cells, assigned_cells


def split_evenly(items, n):
    if n == 0:
        return []
    k, m = divmod(len(items), n)
    return [items[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(n)]


# === 5. Cell Coverage Loss ===
def compute_cell_coverage_loss(sat_to_gw, gw_to_cells, total_cells):
    sat_to_cells, covered_cells = assign_cells_via_gateways(sat_to_gw, gw_to_cells)
    coverage_ratio = len(covered_cells) / total_cells
    return 1 - coverage_ratio  # Want to minimize uncovered portion


# === 6. Combined Loss Function ===
def total_loss_fn(preds, targets, sat_ids, gw_to_cells, total_cells,
                  lambda_global=0.2, lambda_entropy=0.01, lambda_coverage=1.0,
                  label_smoothing=0.1, temperature=1.0):

    loss_cls = classification_loss(preds, targets, label_smoothing)
    loss_global = compute_global_gateway_loss(preds, k=1)
    loss_entropy = entropy_regularization(preds, temperature)

    # Build satellite → predicted gateway mapping
    top1_preds = torch.argmax(preds, dim=1).cpu().numpy()
    predicted_sat_to_gw = {sat_ids[i]: int(top1_preds[i]) for i in range(len(sat_ids))}

    # Compute greedy cell assignment and coverage
    loss_coverage = compute_cell_coverage_loss(predicted_sat_to_gw, gw_to_cells, total_cells)

    total_loss = (
        loss_cls
        + lambda_global * loss_global
        + lambda_entropy * loss_entropy
        + lambda_coverage * loss_coverage
    )

    return total_loss, loss_cls, loss_global, loss_entropy, loss_coverage


# === 7. Top-k Accuracy Metric ===
def top_k_accuracy(preds, labels, k):
    topk = torch.topk(preds, k, dim=1).indices
    return (topk == labels.unsqueeze(1)).any(dim=1).float().mean().item()
