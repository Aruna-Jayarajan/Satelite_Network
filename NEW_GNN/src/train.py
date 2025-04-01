import torch
from tqdm import tqdm
from loss import total_loss
from utils import multilabel_accuracy, top_k_accuracy

def train(model, data_loader, optimizer, device, epoch, scheduler=None, lambda_dict=None):
    """
    Training loop for one epoch using custom total_loss.

    Args:
        model: GNN model
        data_loader: PyTorch DataLoader yielding PyG Data objects
        optimizer: optimizer
        device: 'cuda' or 'cpu'
        epoch: current epoch (for display)
        scheduler: optional LR scheduler
        lambda_dict: dictionary of lambda weights for each loss component
    """
    model.train()
    total_epoch_loss, total_acc, total_topk = 0, 0, 0

    # Unpack or set default lambda weights
    lambda_dict = lambda_dict or {
        'lambda_coverage': 1.0,
        'lambda_fairness': 1.0,
        'lambda_balance': 1.0,
        'lambda_switch': 1.0,
        'lambda_overlap': 1.0
    }

    for data in tqdm(data_loader, desc=f"[Epoch {epoch}]"):
        data = data.to(device)
        optimizer.zero_grad()

        # === Forward ===
        pred_gateway_logits = model(data.x, data.edge_index)

        # === Placeholder Metrics (You should extract these from data or memory) ===
        # Replace with real tracking when you integrate full cell assignment logic
        cell_coverage_matrix = torch.ones(1, device=device)                    # dummy
        sat_gateway_counts = pred_gateway_logits.argmax(dim=1).bincount(minlength=pred_gateway_logits.size(1)).float()
        sat_cell_counts = torch.ones_like(sat_gateway_counts)                 # dummy
        gateway_cell_distribution = torch.ones(pred_gateway_logits.size(1), pred_gateway_logits.size(0))  # dummy

        current_gateways = (pred_gateway_logits > 0).float()                  # thresholded logits
        prev_gateways = current_gateways.clone().detach()                    # replace with real buffer
        current_cells = torch.ones(pred_gateway_logits.size(0), 100).to(device)  # dummy: 100 cells
        prev_cells = current_cells.clone().detach()                          # replace with real buffer

        cell_assignments = {i: set() for i in range(pred_gateway_logits.size(0))}  # dummy
        neighbor_assignments = {i: [] for i in range(pred_gateway_logits.size(0))} # dummy
        gateway_ids = [list(range(pred_gateway_logits.size(1))) for _ in range(pred_gateway_logits.size(0))]  # dummy

        # === Compute Total Loss ===
        loss = total_loss(
            pred_gateway_logits,
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
            **lambda_dict
        )

        # === Backprop ===
        loss.backward()
        optimizer.step()

        # === Metrics ===
        total_epoch_loss += loss.item()
        total_acc += multilabel_accuracy(pred_gateway_logits, data.y)
        total_topk += top_k_accuracy(pred_gateway_logits, data.y, k=3)

    if scheduler:
        scheduler.step()

    n = len(data_loader)
    print(f"[Epoch {epoch}] Loss: {total_epoch_loss/n:.4f} | Acc: {total_acc/n:.4f} | Top-3 Acc: {total_topk/n:.4f}")
    return total_epoch_loss / n, total_acc / n, total_topk / n
