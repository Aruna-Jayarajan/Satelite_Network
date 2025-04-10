import torch
import torch.nn.functional as F
from torch.nn import Linear, ModuleList
from torch_geometric.nn import HeteroConv, SAGEConv

class SatGatewayGNN(torch.nn.Module):
    def __init__(self, hidden_dim, num_gateways, num_cells, num_message_passing_rounds=3):
        super(SatGatewayGNN, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_gateways = num_gateways
        self.num_cells = num_cells
        self.num_rounds = num_message_passing_rounds

        # === Initial Message Passing Layer ===
        self.init_conv = HeteroConv({
            ('sat', 'connected_to', 'sat'): SAGEConv((-1, -1), hidden_dim),
            ('sat', 'connects', 'gateway'): SAGEConv((-1, -1), hidden_dim),
            ('gateway', 'serves', 'cell'): SAGEConv((-1, -1), hidden_dim),
            ('sat', 'covers', 'cell'): SAGEConv((-1, -1), hidden_dim),
            ('cell', 'covered_by', 'sat'): SAGEConv((-1, -1), hidden_dim),
        }, aggr='sum')

        # === Message Passing Layers ===
        in_dim = hidden_dim + num_gateways + num_cells
        self.conv_layers = ModuleList([
            HeteroConv({
                ('sat', 'connected_to', 'sat'): SAGEConv((in_dim, in_dim), hidden_dim),
                ('sat', 'covers', 'cell'): SAGEConv((in_dim, -1), hidden_dim),
                ('cell', 'covered_by', 'sat'): SAGEConv((-1, in_dim), hidden_dim),
            }, aggr='sum') for _ in range(num_message_passing_rounds)
        ])

        # === Classification Heads for Each Round ===
        self.gateway_heads = ModuleList([
            Linear(hidden_dim, num_gateways) for _ in range(num_message_passing_rounds + 1)
        ])
        self.cell_heads = ModuleList([
            Linear(hidden_dim, num_cells) for _ in range(num_message_passing_rounds + 1)
        ])

    def forward(self, data, cell_visibility_matrix=None, cell_demands=None):
        x_dict, edge_index_dict = data.x_dict, data.edge_index_dict
        all_gateway_logits = []
        all_cell_logits = []
        all_sat_demands = []

        # === Initial Message Passing ===
        x_dict = self.init_conv(x_dict, edge_index_dict)
        x_dict = {k: F.relu(v) for k, v in x_dict.items()}

        sat_feats = x_dict['sat']
        gw_logits = self.gateway_heads[0](sat_feats)
        cell_logits = self.cell_heads[0](sat_feats)
        all_gateway_logits.append(gw_logits)
        all_cell_logits.append(cell_logits)

        if cell_visibility_matrix is not None and cell_demands is not None:
            cell_probs = torch.sigmoid(cell_logits)
            sat_demand = torch.matmul(cell_probs * cell_visibility_matrix, cell_demands)
            all_sat_demands.append(sat_demand)

        # === Message Passing Rounds ===
        x_dict['sat'] = torch.cat([sat_feats, gw_logits, cell_logits], dim=1)
        for i in range(self.num_rounds):
            x_dict = self.conv_layers[i](x_dict, edge_index_dict)
            x_dict = {k: F.relu(v) for k, v in x_dict.items()}
            sat_feats = x_dict['sat']

            gw_logits = self.gateway_heads[i + 1](sat_feats)
            cell_logits = self.cell_heads[i + 1](sat_feats)
            all_gateway_logits.append(gw_logits)
            all_cell_logits.append(cell_logits)

            if cell_visibility_matrix is not None and cell_demands is not None:
                cell_probs = torch.sigmoid(cell_logits)
                sat_demand = torch.matmul(cell_probs * cell_visibility_matrix, cell_demands)
                all_sat_demands.append(sat_demand)

            x_dict['sat'] = torch.cat([sat_feats, gw_logits, cell_logits], dim=1)

        return all_gateway_logits, all_cell_logits, all_sat_demands
