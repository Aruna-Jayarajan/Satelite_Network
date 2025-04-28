import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import HeteroConv, GATConv, Linear


class SatGatewayCellGNN(nn.Module):
    

    def __init__(self, hidden_dim, num_gateways, num_cells, input_dims, num_rounds=3):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_rounds = num_rounds
        self.num_gateways = num_gateways
        self.num_cells = num_cells

        # === Node Feature Encoders ===
        self.encoders = nn.ModuleDict({
            'sat': Linear(input_dims['sat'], hidden_dim),
            'gateway': Linear(input_dims['gateway'], hidden_dim),
            'cell': Linear(input_dims['cell'], hidden_dim),
        })

        # === Message Passing Layers ===
        self.hetero_conv = HeteroConv({
            ('sat', 'connected_to', 'sat'): GATConv(hidden_dim, hidden_dim),
            ('sat', 'connects', 'gateway'): GATConv(hidden_dim, hidden_dim, add_self_loops=False),
            ('gateway', 'serves', 'cell'): GATConv(hidden_dim, hidden_dim, add_self_loops=False),
            ('sat', 'covers', 'cell'): GATConv(hidden_dim, hidden_dim, add_self_loops=False),
        }, aggr='sum')

        self.norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(p=0.1)

        # === Satellite Memory ===
        self.gru = nn.GRUCell(hidden_dim, hidden_dim)

        # === Prediction Heads ===
        self.sat_to_gateway_head = nn.Linear(hidden_dim, num_gateways)
        self.sat_to_cell_head = nn.Linear(hidden_dim, num_cells)
        self.cell_to_satellite_head = nn.Linear(hidden_dim, hidden_dim)  # Not used explicitly yet

    def forward(self, data, visibility_matrices, sat_memory=None):
        """
        Args:
            data (HeteroData): Graph input.
            visibility_matrices (dict): Visibility masks for masking logits.
            sat_memory (Tensor, optional): Satellite hidden states from previous step.

        Returns:
            dict: Prediction outputs and memory outputs.
        """

        # === Initial Encoding ===
        x_dict = {}
        for node_type in ['sat', 'gateway', 'cell']:
            x = self.encoders[node_type](data[node_type].x)
            x = F.relu(x)
            x_dict[node_type] = x

        # === Memory Initialization ===
        h_sat = x_dict['sat'] if sat_memory is None else sat_memory.detach()

        # === Message Passing ===
        for _ in range(self.num_rounds):
            out_dict = self.hetero_conv(x_dict, data.edge_index_dict)
            out_dict = {k: self.norm(v) for k, v in out_dict.items()}
            out_dict = {k: F.relu(self.dropout(v)) for k, v in out_dict.items()}

            h_sat = self.gru(out_dict['sat'], h_sat)
            x_dict['sat'] = h_sat

        # === Prediction Heads ===
        sat_gateway_logits = self.sat_to_gateway_head(h_sat)
        sat_gateway_logits = sat_gateway_logits.masked_fill(visibility_matrices['sat_gateway'] == 0, float('-1e9'))
        sat_gateway_probs = F.softmax(sat_gateway_logits, dim=1)

        sat_cell_logits = self.sat_to_cell_head(h_sat)
        sat_cell_logits = sat_cell_logits.masked_fill(visibility_matrices['sat_cell'] == 0, float('-1e9'))
        sat_cell_probs = torch.sigmoid(sat_cell_logits)

        # === Cell to Satellite Matching ===
        cell_embeddings = x_dict['cell']
        sat_embeddings = h_sat

        cell_to_sat_scores = torch.matmul(cell_embeddings, sat_embeddings.T)  # [num_cells, num_sats]
        cell_to_sat_scores = cell_to_sat_scores.masked_fill(visibility_matrices['cell_sat'] == 0, float('-1e9'))
        cell_to_sat_probs = F.softmax(cell_to_sat_scores, dim=1)

        return {
            'sat_gateway_logits': sat_gateway_logits,
            'sat_cell_logits': sat_cell_logits,
            'sat_gateway_probs': sat_gateway_probs,
            'sat_cell_probs': sat_cell_probs,
            'cell_sat_probs': cell_to_sat_probs,
            'sat_memory_out': h_sat,
        }


