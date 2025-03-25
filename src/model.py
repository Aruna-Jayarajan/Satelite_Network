import torch
import torch.nn as nn
import torch_geometric.nn as pyg_nn


class Stage2GNN(nn.Module):
    def __init__(self, input_dim, sat_feature_dim, neighbor_feature_dim, hidden_dim, output_dim):
        """
        GNN for Satellite-to-Gateway Assignment.

        Args:
            input_dim (int): Total input feature dimension per node.
            sat_feature_dim (int): Dimension of satellite features.
            neighbor_feature_dim (int): Dimension of neighbor-derived features.
            hidden_dim (int): Dimension of hidden layers.
            output_dim (int): Number of gateway classes (e.g., 54).
        """
        super(Stage2GNN, self).__init__()
        self.sat_feature_dim = sat_feature_dim
        self.neighbor_feature_dim = neighbor_feature_dim

        self.sat_fc = nn.Linear(sat_feature_dim, hidden_dim)
        self.neigh_fc = nn.Linear(neighbor_feature_dim, hidden_dim)

        self.conv1 = pyg_nn.GATConv(hidden_dim, hidden_dim)
        self.conv2 = pyg_nn.GATConv(hidden_dim, hidden_dim)

        self.fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x, edge_index):
        sat_features = x[:, :self.sat_feature_dim]
        neighbor_features = x[:, self.sat_feature_dim:]

        sat_feat = torch.relu(self.sat_fc(sat_features))
        neigh_feat = torch.relu(self.neigh_fc(neighbor_features))

        x = sat_feat + 0.5 * neigh_feat  # Weighted combination

        x = torch.relu(self.conv1(x, edge_index))
        x = torch.relu(self.conv2(x, edge_index))

        x = torch.relu(self.fc1(x))
        return self.fc2(x)
