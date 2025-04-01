import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, GINConv, BatchNorm

class Stage2GNN(nn.Module):
    def __init__(self, input_dim, sat_feature_dim, neighbor_feature_dim, hidden_dim, output_dim,
                 gnn_type='gat', dropout=0.3, use_residual=True):
        super(Stage2GNN, self).__init__()
        self.use_residual = use_residual
        self.gnn_type = gnn_type
        self.dropout = dropout

        self.sat_feature_dim = sat_feature_dim
        self.neighbor_feature_dim = neighbor_feature_dim

        # Initial MLPs to process raw features
        self.sat_fc = nn.Linear(sat_feature_dim, hidden_dim)
        self.neigh_fc = nn.Linear(neighbor_feature_dim, hidden_dim)

        # Learnable fusion gate (sat vs neighbor info)
        self.fusion_fc = nn.Linear(hidden_dim * 2, 1)

        # GNN Layers
        if gnn_type == 'gat':
            self.conv1 = GATConv(hidden_dim, hidden_dim)
            self.conv2 = GATConv(hidden_dim, hidden_dim)
        elif gnn_type == 'gin':
            self.conv1 = GINConv(nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim)
            ))
            self.conv2 = GINConv(nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim)
            ))
        else:
            raise ValueError("Unsupported GNN type. Choose 'gat' or 'gin'.")

        # Batch Normalization
        self.norm1 = BatchNorm(hidden_dim)
        self.norm2 = BatchNorm(hidden_dim)

        # Final prediction layers
        self.fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)  # output_dim = number of gateways

    def forward(self, x, edge_index):
        # Separate satellite and neighbor inputs
        sat_feat = F.relu(self.sat_fc(x[:, :self.sat_feature_dim]))
        neigh_feat = F.relu(self.neigh_fc(x[:, self.sat_feature_dim:]))

        # Fusion using learnable gate
        fusion_input = torch.cat([sat_feat, neigh_feat], dim=1)
        fusion_gate = torch.sigmoid(self.fusion_fc(fusion_input))
        x = fusion_gate * sat_feat + (1 - fusion_gate) * neigh_feat

        # GNN Layer 1
        res = x
        x = self.conv1(x, edge_index)
        x = self.norm1(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        if self.use_residual:
            x = x + res

        # GNN Layer 2
        res = x
        x = self.conv2(x, edge_index)
        x = self.norm2(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        if self.use_residual:
            x = x + res

        # Final MLP
        x = F.relu(self.fc1(x))
        x = F.dropout(x, p=self.dropout, training=self.training)
        return self.fc2(x)


def compute_global_gateway_loss(preds, k=1):
    """
    Global gateway loss to penalize uncovered gateways.
    Args:
        preds (torch.Tensor): [N x G] logits or probs for gateway assignments
        k (int): top-k selections per satellite
    Returns:
        torch.Tensor: scalar loss value
    """
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
