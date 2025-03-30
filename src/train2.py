import torch
import numpy as np
from loss2 import total_loss_fn, top_k_accuracy
from dataloader import prepare_input_for_gnn


def train_model_with_coverage(
    gnn_model,
    stage1_model,
    train_loader,
    val_loader,
    optimizer_gnn,
    gw_to_cells,              # NEW
    total_cells,              # NEW
    num_epochs=30,
    rounds=15,
    lambda_global=0.2,
    lambda_entropy=0.01,
    lambda_coverage=1.0,
    label_smoothing=0.1,
    clip_grad_norm=2.0
):
    train_losses, val_losses = [], []
    train_top1_acc, train_top3_acc, train_top5_acc = [], [], []
    val_top1_acc, val_top3_acc, val_top5_acc = [], [], []

    for epoch in range(1, num_epochs + 1):
        gnn_model.train()
        total_loss = total_top1 = total_top3 = total_top5 = total_train_samples = 0

        for data in train_loader:
            optimizer_gnn.zero_grad()

            # Stage 1 prediction
            input_features = data.x[:, :57].cpu().numpy()
            preds_model1 = stage1_model.predict(input_features, verbose=0)

            top3_model1 = np.argsort(preds_model1, axis=1)[:, -3:]
            binary_preds_model1 = np.zeros_like(preds_model1)
            for i, idx in enumerate(top3_model1):
                binary_preds_model1[i, idx] = 1

            gnn_input_data = prepare_input_for_gnn(data, torch.tensor(binary_preds_model1, dtype=torch.float))

            for _ in range(rounds):
                preds_gnn_round = gnn_model(gnn_input_data.x, gnn_input_data.edge_index)
                top3_gnn_round = torch.topk(preds_gnn_round, k=3, dim=1).indices
                gnn_input_data = prepare_input_for_gnn(data, top3_gnn_round)

            sat_ids = data.sat_ids if hasattr(data, 'sat_ids') else list(range(len(data.y)))

            total_loss_value, _, _, _, _ = total_loss_fn(
                preds=preds_gnn_round,
                targets=data.y,
                sat_ids=sat_ids,
                gw_to_cells=gw_to_cells,       # CHANGED
                total_cells=total_cells,
                lambda_global=lambda_global,
                lambda_entropy=lambda_entropy,
                lambda_coverage=lambda_coverage,
                label_smoothing=label_smoothing
            )

            total_loss_value.backward()
            if clip_grad_norm:
                torch.nn.utils.clip_grad_norm_(gnn_model.parameters(), clip_grad_norm)
            optimizer_gnn.step()

            total_loss += total_loss_value.item()
            total_top1 += top_k_accuracy(preds_gnn_round, data.y, k=1)
            total_top3 += top_k_accuracy(preds_gnn_round, data.y, k=3)
            total_top5 += top_k_accuracy(preds_gnn_round, data.y, k=5)
            total_train_samples += 1

        train_losses.append(total_loss / total_train_samples)
        train_top1_acc.append(total_top1 / total_train_samples)
        train_top3_acc.append(total_top3 / total_train_samples)
        train_top5_acc.append(total_top5 / total_train_samples)

        # === Validation ===
        gnn_model.eval()
        val_loss = val_top1 = val_top3 = val_top5 = val_samples = 0
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

                sat_ids = data.sat_ids if hasattr(data, 'sat_ids') else list(range(len(data.y)))
                loss_val, *_ = total_loss_fn(
                    preds=preds_gnn,
                    targets=data.y,
                    sat_ids=sat_ids,
                    gw_to_cells=gw_to_cells,    # CHANGED
                    total_cells=total_cells,
                    lambda_global=lambda_global,
                    lambda_entropy=lambda_entropy,
                    lambda_coverage=lambda_coverage,
                    label_smoothing=label_smoothing
                )

                val_loss += loss_val.item()
                val_top1 += top_k_accuracy(preds_gnn, data.y, k=1)
                val_top3 += top_k_accuracy(preds_gnn, data.y, k=3)
                val_top5 += top_k_accuracy(preds_gnn, data.y, k=5)
                val_samples += 1

        val_losses.append(val_loss / val_samples)
        val_top1_acc.append(val_top1 / val_samples)
        val_top3_acc.append(val_top3 / val_samples)
        val_top5_acc.append(val_top5 / val_samples)

        if epoch % 2 == 0:
            print(f"[Epoch {epoch:02d}] Train Loss: {train_losses[-1]:.4f} | "
                  f"Val Loss: {val_losses[-1]:.4f} | "
                  f"Top-1: {train_top1_acc[-1]:.3f} / {val_top1_acc[-1]:.3f}")

    return (
        train_losses, val_losses,
        train_top1_acc, train_top3_acc, train_top5_acc,
        val_top1_acc, val_top3_acc, val_top5_acc
    )
