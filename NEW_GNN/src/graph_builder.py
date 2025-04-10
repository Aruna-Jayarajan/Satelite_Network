# graph_builder.py

from torch_geometric.data import HeteroData
from geopy.distance import geodesic
import torch

def build_hetero_graph(satellites, gateways, cells, sat_distance_threshold_km=1000, gw_cell_threshold_km=100):
    data = HeteroData()

    # === Satellite node features ===
    sat_features = []
    sat_id_map = {}
    for i, sat in enumerate(satellites):
        sat_id_map[sat['sat_id']] = i  # FIXED
        sat_feat = [sat['latitude'], sat['longitude'], sat['altitude']] + sat['visible_gateways']
        sat_features.append(sat_feat)
    data['sat'].x = torch.tensor(sat_features, dtype=torch.float)

    # === Gateway node features ===
    gw_features = []
    gw_id_map = {}
    for i, gw in enumerate(gateways):
        gw_id_map[gw['id']] = i
        gw_feat = [gw['latitude'], gw['longitude']]
        gw_features.append(gw_feat)
    data['gateway'].x = torch.tensor(gw_features, dtype=torch.float)

    # === Cell node features ===
    cell_features = []
    cell_id_map = {}
    for i, cell in enumerate(cells):
        cell_id_map[cell['cell_id']] = i  # FIXED
        cell_feat = [cell['latitude'], cell['longitude'], cell['demand']]
        cell_features.append(cell_feat)
    data['cell'].x = torch.tensor(cell_features, dtype=torch.float)

    # === Satellite ↔ Satellite edges ===
    sat_src, sat_dst = [], []
    for i, sat_a in enumerate(satellites):
        coord_a = (sat_a['latitude'], sat_a['longitude'])
        for j, sat_b in enumerate(satellites):
            if i >= j:
                continue
            coord_b = (sat_b['latitude'], sat_b['longitude'])
            if geodesic(coord_a, coord_b).km <= sat_distance_threshold_km:
                sat_src += [i, j]
                sat_dst += [j, i]
    data['sat', 'connected_to', 'sat'].edge_index = torch.tensor([sat_src, sat_dst], dtype=torch.long)

    # === Satellite → Gateway edges ===
    sat_to_gw_src, sat_to_gw_dst = [], []
    for i, sat in enumerate(satellites):
        for gw_idx, is_visible in enumerate(sat['visible_gateways']):
            if is_visible and gw_idx in gw_id_map:
                sat_to_gw_src.append(i)
                sat_to_gw_dst.append(gw_idx)
    data['sat', 'connects', 'gateway'].edge_index = torch.tensor([sat_to_gw_src, sat_to_gw_dst], dtype=torch.long)

    # === Gateway → Cell edges ===
    gw_to_cell_src, gw_to_cell_dst = [], []
    for gw in gateways:
        gw_idx = gw_id_map[gw['id']]
        gw_coord = (gw['latitude'], gw['longitude'])

        for cell in cells:
            cell_idx = cell_id_map[cell['cell_id']]  # FIXED
            cell_coord = (cell['latitude'], cell['longitude'])

            if geodesic(gw_coord, cell_coord).km <= gw_cell_threshold_km:
                gw_to_cell_src.append(gw_idx)
                gw_to_cell_dst.append(cell_idx)
    data['gateway', 'serves', 'cell'].edge_index = torch.tensor([gw_to_cell_src, gw_to_cell_dst], dtype=torch.long)

    return data
