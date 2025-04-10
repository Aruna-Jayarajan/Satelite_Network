import torch
from src.loss import compute_custom_loss

def evaluate(model, graphs, cells_per_graph, device, cell_gateway_file):
    model.eval()
    total_loss = 0.0
    total_coverage = 0.0
    total_fairness = 0.0
    total_gateway_loss = 0.0
    total_spatial_loss = 0.0
    num_graphs = 0

    with torch.no_grad():
        for i, graph in enumerate(graphs):
            graph = graph.to(device)
            cells = cells_per_graph[i]  # needed for spatial/demand-aware loss

            x_dict = graph.x_dict
            edge_index_dict = graph.edge_index_dict
            edge_label_index = graph['satellite', 'serves', 'cell'].edge_index
            gw_labels = graph['satellite'].y

            edge_logits, gw_logits = model(x_dict, edge_index_dict, edge_label_index)

            loss, metrics = compute_custom_loss(
                gw_logits=gw_logits,
                edge_logits=edge_logits,
                edge_index=edge_label_index,
                x_dict=x_dict,
                cells=cells,
                cell_gateway_file=cell_gateway_file,
                gw_labels=gw_labels
            )

            total_loss += loss.item()
            total_coverage += metrics["cell_coverage_loss"]
            total_fairness += metrics["demand_balance_loss"]
            total_gateway_loss += metrics["gateway_coverage_loss"]
            total_spatial_loss += metrics["spatial_loss"]
            num_graphs += 1

    return {
        'loss': total_loss / num_graphs,
        'coverage': total_coverage / num_graphs,
        'fairness': total_fairness / num_graphs,
        'gateway_loss': total_gateway_loss / num_graphs,
        'spatial_loss': total_spatial_loss / num_graphs,
    }
