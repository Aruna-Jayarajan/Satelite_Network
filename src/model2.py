import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, GINConv, BatchNorm

class Stage2GNN(nn.Module):
    """
    Stage 2 GNN for satellite-to-gateway assignment with optional coverage-aware loss.
    """
    def __init__(self, input_dim, sat_feature_dim, neighbor_feature_dim, hidden_dim, output_dim,
                 gnn_type='gat', dropout=0.3, use_residual=True):
        super(Stage2GNN, self).__init__()

        self.use_residual = use_residual
        self.dropout = dropout
        self.sat_feature_dim = sat_feature_dim
        self.neighbor_feature_dim = neighbor_feature_dim

        # Satellite and neighbor feature encoders
        self.sat_fc = nn.Linear(sat_feature_dim, hidden_dim)
        self.neigh_fc = nn.Linear(neighbor_feature_dim, hidden_dim)

        # Learnable fusion gate to balance satellite and neighbor context
        self.fusion_fc = nn.Linear(hidden_dim * 2, 1)

        # GNN layers (GAT or GIN)
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

        # Normalization layers
        self.norm1 = BatchNorm(hidden_dim)
        self.norm2 = BatchNorm(hidden_dim)

        # Output layers
        self.fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x, edge_index):
        # Split input into satellite-specific and neighbor-specific features
        sat_feat = F.relu(self.sat_fc(x[:, :self.sat_feature_dim]))
        neigh_feat = F.relu(self.neigh_fc(x[:, self.sat_feature_dim:]))

        # Learnable fusion gate to combine features
        fusion_input = torch.cat([sat_feat, neigh_feat], dim=1)
        fusion_gate = torch.sigmoid(self.fusion_fc(fusion_input))
        x = fusion_gate * sat_feat + (1 - fusion_gate) * neigh_feat

        # GNN layer 1 with optional residual connection
        res = x
        x = self.conv1(x, edge_index)
        x = self.norm1(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        if self.use_residual:
            x = x + res

        # GNN layer 2
        res = x
        x = self.conv2(x, edge_index)
        x = self.norm2(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        if self.use_residual:
            x = x + res

        # Final output layers
        x = F.relu(self.fc1(x))
        x = F.dropout(x, p=self.dropout, training=self.training)
        return self.fc2(x)
