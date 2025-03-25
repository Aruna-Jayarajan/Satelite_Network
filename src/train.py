import torch
import numpy as np
from loss import total_loss_fn, get_binary_predictions, top_k_accuracy
from dataloader import prepare_input_for_gnn, NUM_GATEWAYS
import torch.nn.functional as F


def train_model_with_mse(
    gnn_model,
    stage1_model,
    train_loader,
    val_loader,
    optimizer_gnn,
    num_epochs=30,
    rounds=15,
    lambda_global=0.2,
    label_smoothing=0.1
):

    train_losses, val_losses = [], []
    train_top1_acc, train_top3_acc, train_top5_acc = [], [], []
    val_top1_acc, val_top3_acc, val_top5_acc = [], [], []

    for epoch in range(1, num_epochs + 1):
        gnn_model.train()
        total_loss = total_top1 = total_top3 = total_top5 = total_train_samples = 0

        for data in train_loader:
            optimizer_gnn.zero_grad()

            # Stage 1 prediction using scaled input
            input_features = data.x[:, :57].cpu().numpy()
            preds_model1 = stage1_model.predict(input_features, verbose=0)


            # Top-3 one-hot encoding
            top3_model1 = np.argsort(preds_model1, axis=1)[:, -3:]
            binary_preds_model1 = np.zeros_like(preds_model1)
            for i, idx in enumerate(top3_model1):
                binary_preds_model1[i, idx] = 1

            gnn_input_data = prepare_input_for_gnn(data, torch.tensor(binary_preds_model1, dtype=torch.float))

            # Iterative GNN refinement
            for _ in range(rounds):
                preds_gnn_round = gnn_model(gnn_input_data.x, gnn_input_data.edge_index)
                top3_gnn_round = torch.topk(preds_gnn_round, k=3, dim=1).indices
                gnn_input_data = prepare_input_for_gnn(data, top3_gnn_round)

            # Loss calculation
            total_loss_value, _, _ = total_loss_fn(preds_gnn_round, data.y, lambda_global, label_smoothing)
            total_loss_value.backward()
            optimizer_gnn.step()

            # Metrics
            total_loss += total_loss_value.item()
            total_top1 += top_k_accuracy(preds_gnn_round, data.y, k=1)
            total_top3 += top_k_accuracy(preds_gnn_round, data.y, k=3)
            total_top5 += top_k_accuracy(preds_gnn_round, data.y, k=5)
            total_train_samples += 1

        # Training results for epoch
        train_losses.append(total_loss / total_train_samples)
        train_top1_acc.append(total_top1 / total_train_samples)
        train_top3_acc.append(total_top3 / total_train_samples)
        train_top5_acc.append(total_top5 / total_train_samples)

        # Validation
        gnn_model.eval()
        val_loss = val_top1 = val_top3 = val_top5 = total_val_samples = 0
        with torch.no_grad():
            for data in val_loader:
                input_features = data.x[:, :57].cpu().numpy()
                preds_model1 = stage1_model.predict(input_features, verbose=0)

                top3_model1 = np.argsort(preds_model1, axis=1)[:, -3:]
                binary_preds_model1 = np.zeros_like(preds_model1)
                for i, idx in enumerate(top3_model1):
                    binary_preds_model1[i, idx] = 1

                gnn_input_data = prepare_input_for_gnn(data, torch.tensor(binary_preds_model1, dtype=torch.float))
                preds_gnn = gnn_model(gnn_input_data.x, gnn_input_data.edge_index)

                loss_val, _, _ = total_loss_fn(preds_gnn, data.y, lambda_global, label_smoothing)
                val_loss += loss_val.item()
                val_top1 += top_k_accuracy(preds_gnn, data.y, k=1)
                val_top3 += top_k_accuracy(preds_gnn, data.y, k=3)
                val_top5 += top_k_accuracy(preds_gnn, data.y, k=5)
                total_val_samples += 1

        # Validation results for epoch
        val_losses.append(val_loss / total_val_samples)
        val_top1_acc.append(val_top1 / total_val_samples)
        val_top3_acc.append(val_top3 / total_val_samples)
        val_top5_acc.append(val_top5 / total_val_samples)

        if epoch % 2 == 0:
            print(f"Epoch {epoch}, Train Loss: {train_losses[-1]:.4f}, Val Loss: {val_losses[-1]:.4f}")

    return (
        train_losses, val_losses,
        train_top1_acc, train_top3_acc, train_top5_acc,
        val_top1_acc, val_top3_acc, val_top5_acc
    )
