# src/train.py

import torch
import torch.nn as nn
import torch.optim as optim
from src.loss import compute_custom_loss
import matplotlib.pyplot as plt


def train(model, dataset, cells_per_snapshot, device, epochs=5, inner_steps=5, lr=1e-3,
          cell_gateway_file=r"C:\Users\aruna\Desktop\MS Thesis\Real Data\cells_with_gateways.csv"):
    
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    train_losses = []
    train_coverage = []
    train_fairness = []
    train_gateway_loss = []
    train_spatial_loss = []

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        total_cov = 0.0
        total_fair = 0.0
        total_gw = 0.0
        total_spatial = 0.0

        for i, graph in enumerate(dataset):
            graph = graph.to(device)
            cells = cells_per_snapshot[i]  # Pass the corresponding cell list for this graph

            for step in range(inner_steps):
                x_dict = graph.x_dict
                edge_index_dict = graph.edge_index_dict
                edge_label_index = graph['satellite', 'serves', 'cell'].edge_index
                gw_labels = graph['satellite'].y

                # Forward pass
                edge_logits, gw_logits = model(x_dict, edge_index_dict, edge_label_index)

                # Compute custom loss
                loss, metrics = compute_custom_loss(
                    gw_logits=gw_logits,
                    edge_logits=edge_logits,
                    edge_index=edge_label_index,
                    x_dict=x_dict,
                    cells=cells,
                    cell_gateway_file=cell_gateway_file,
                    gw_labels=gw_labels  # optional
                )

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # Logging
                total_loss += loss.item()
                total_cov += metrics["cell_coverage_loss"]
                total_fair += metrics["demand_balance_loss"]
                total_gw += metrics["gateway_coverage_loss"]
                total_spatial += metrics["spatial_loss"]

        train_losses.append(total_loss)
        train_coverage.append(total_cov)
        train_fairness.append(total_fair)
        train_gateway_loss.append(total_gw)
        train_spatial_loss.append(total_spatial)

        print(f"Epoch {epoch+1} | Total Loss: {total_loss:.4f} | "
              f"Coverage: {total_cov:.4f} | "
              f"Fairness: {total_fair:.4f} | "
              f"Gateway: {total_gw:.4f} | "
              f"Spatial: {total_spatial:.4f}")

    # === Plotting ===
    epochs_range = range(1, epochs + 1)
    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, train_losses, label='Total Loss')
    plt.plot(epochs_range, train_gateway_loss, label='Gateway Loss')
    plt.plot(epochs_range, train_spatial_loss, label='Spatial Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Losses')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, train_coverage, label='Coverage Loss')
    plt.plot(epochs_range, train_fairness, label='Fairness (Demand) Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss Value')
    plt.title('Coverage & Fairness Loss')
    plt.legend()

    plt.tight_layout()
    plt.show()
