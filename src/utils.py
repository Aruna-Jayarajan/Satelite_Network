import matplotlib.pyplot as plt
import torch
from torch_geometric.data import Data
from collections import defaultdict
import pandas as pd

NUM_GATEWAYS = 54  # Adjust or import as needed


# === 1. GNN Input Preparation ===
def prepare_input_for_gnn(data, top3_gateway_predictions):
    updated_x = data.x.clone()
    expanded_predictions = torch.zeros(updated_x.size(0), NUM_GATEWAYS, dtype=torch.float)
    top3_gateway_predictions = top3_gateway_predictions.long()

    for i in range(updated_x.size(0)):
        for gateway_idx in top3_gateway_predictions[i]:
            expanded_predictions[i, gateway_idx] = 1

    updated_x[:, -NUM_GATEWAYS:] = expanded_predictions
    return Data(x=updated_x, edge_index=data.edge_index)


# === 2. Accuracy Metric ===
def top_k_accuracy(preds, labels, k):
    topk = torch.topk(preds, k, dim=1).indices
    return (topk == labels.unsqueeze(1)).any(dim=1).float().mean().item()


# === 3. Plotting Training Metrics ===
def plot_metrics(train_losses, val_losses,
                 train_top1_acc, train_top3_acc, train_top5_acc,
                 val_top1_acc, val_top3_acc, val_top5_acc):
    epochs = range(1, len(train_losses) + 1)

    # Plot Loss
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, train_losses, label='Train Loss')
    plt.plot(epochs, val_losses, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Loss Over Epochs')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # Plot Accuracy
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, train_top1_acc, label='Train Top-1 Accuracy', linestyle='-')
    plt.plot(epochs, train_top3_acc, label='Train Top-3 Accuracy', linestyle='-')
    plt.plot(epochs, train_top5_acc, label='Train Top-5 Accuracy', linestyle='-')
    plt.plot(epochs, val_top1_acc, label='Val Top-1 Accuracy', linestyle='--')
    plt.plot(epochs, val_top3_acc, label='Val Top-3 Accuracy', linestyle='--')
    plt.plot(epochs, val_top5_acc, label='Val Top-5 Accuracy', linestyle='--')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Top-k Accuracy Over Epochs')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# === 4. Gateway-to-Cell Mapping Builder ===

import pandas as pd
from collections import defaultdict

def build_gateway_to_cells_mapping(csv_path):
    """
    Builds a mapping from gateway ID to cell indices based on closest and second-closest gateways.

    Args:
        csv_path (str): Path to the cells_with_gateways.csv file

    Returns:
        dict: {gateway_id: [cell_idx, ...]}
    """
    df = pd.read_csv(csv_path)
    gw_to_cells = defaultdict(list)

    for idx, row in df.iterrows():
        gw_to_cells[row['closest_gw_id']].append(idx)
        gw_to_cells[row['second_closest_gw_id']].append(idx)

    return gw_to_cells


from collections import defaultdict

def assign_cells_via_gateways(predicted_sat_to_gw, gw_to_cells):
    """
    Assign cells to satellites based on predicted satellite-to-gateway assignments.
    Ensures each cell is covered by at most one satellite.
    """
    satellite_to_cells = defaultdict(list)
    assigned_cells = set()

    # Reverse: gateway → satellites
    gw_to_sats = defaultdict(list)
    for sat, gw in predicted_sat_to_gw.items():
        gw_to_sats[gw].append(sat)

    for gw, sats in gw_to_sats.items():
        candidate_cells = gw_to_cells.get(gw, [])
        sats = list(set(sats))  # just in case

        if not candidate_cells or not sats:
            continue

        cell_chunks = split_evenly(candidate_cells, len(sats))
        for sat, cells in zip(sats, cell_chunks):
            for cell in cells:
                if cell not in assigned_cells:
                    satellite_to_cells[sat].append(cell)
                    assigned_cells.add(cell)

    return satellite_to_cells


def split_evenly(items, n):
    """Split list into n approximately equal chunks."""
    if n == 0:
        return []
    k, m = divmod(len(items), n)
    return [items[i*k + min(i, m):(i+1)*k + min(i+1, m)] for i in range(n)]


# === 4. Satellite-to-Cell Mapping Builder ===
def build_sat_to_cells(file_paths):
    """
    Constructs sat_to_cells mapping and total unique cells count from training files.

    Args:
        file_paths (list of str): Paths to CSV files containing 'feed_sat' and 'cell_id'.

    Returns:
        dict: sat_to_cells mapping {sat_id: [cell_ids]}
        int: total number of unique cells
    """
    sat_to_cells = defaultdict(set)
    all_cells = set()

    for path in file_paths:
        df = pd.read_csv(path)
        for _, row in df.iterrows():
            sat = row['feed_sat']
            cell = row['cell_id']
            sat_to_cells[sat].add(cell)
            all_cells.add(cell)

    sat_to_cells = {sat: list(cells) for sat, cells in sat_to_cells.items()}
    total_cells = len(all_cells)
    return sat_to_cells, total_cells