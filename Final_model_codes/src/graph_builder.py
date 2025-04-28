from torch_geometric.data import HeteroData
from geopy.distance import geodesic
import torch

def build_hetero_graph(
    satellites,
    gateways,
    cells,
    timestep: int = 0,
    sat_distance_threshold_km=2000,
    gw_cell_distance_threshold_km=600,
    structured_neighbors=False
):
    """
    Constructs a heterogeneous graph for satellite, gateway, and cell networks.
    """

    data = HeteroData()

    # === Satellites ===
    sat_features = []
    sat_id_map = {}
    sat_ids = []

    for i, sat in enumerate(satellites):
        sat_id_map[sat['sat_id']] = i
        sat_ids.append(sat['sat_id'])
        features = [sat['latitude'], sat['longitude'], sat['altitude']] + sat['visible_gateways']
        features.append(timestep)
        sat_features.append(features)

    data['sat'].x = torch.tensor(sat_features, dtype=torch.float)
    data['sat'].sat_id = torch.tensor(sat_ids, dtype=torch.long)

    # === Gateways ===
    gw_features = []
    gw_id_map = {}

    for i, gw in enumerate(gateways):
        gw_id_map[gw['id']] = i
        features = [gw['latitude'], gw['longitude'], timestep]
        gw_features.append(features)

    data['gateway'].x = torch.tensor(gw_features, dtype=torch.float)

    # === Cells ===
    cell_features = []
    cell_id_map = {}

    for i, cell in enumerate(cells):
        cell_id_map[cell['cell_id']] = i
        features = [cell['latitude'], cell['longitude'], cell['demand'], timestep]
        cell_features.append(features)

    data['cell'].x = torch.tensor(cell_features, dtype=torch.float)

    # === Satellite ↔ Satellite (Coordination edges) ===
    sat_src = []
    sat_dst = []

    if structured_neighbors:
        sat_positions = [(i, sat['latitude'], sat['longitude']) for i, sat in enumerate(satellites)]

        for idx, (i, lat_i, lon_i) in enumerate(sat_positions):
            min_lat_diff_up = float('inf')
            min_lat_diff_down = float('inf')
            min_lon_diff_left = float('inf')
            min_lon_diff_right = float('inf')

            up_neighbor = None
            down_neighbor = None
            left_neighbor = None
            right_neighbor = None

            for jdx, (j, lat_j, lon_j) in enumerate(sat_positions):
                if i == j:
                    continue

                lat_diff = lat_j - lat_i
                lon_diff = lon_j - lon_i

                if lat_diff > 0 and abs(lon_diff) < 5:
                    if lat_diff < min_lat_diff_up:
                        min_lat_diff_up = lat_diff
                        up_neighbor = j

                if lat_diff < 0 and abs(lon_diff) < 5:
                    if abs(lat_diff) < min_lat_diff_down:
                        min_lat_diff_down = abs(lat_diff)
                        down_neighbor = j

                if lon_diff > 0 and abs(lat_diff) < 5:
                    if lon_diff < min_lon_diff_right:
                        min_lon_diff_right = lon_diff
                        right_neighbor = j

                if lon_diff < 0 and abs(lat_diff) < 5:
                    if abs(lon_diff) < min_lon_diff_left:
                        min_lon_diff_left = abs(lon_diff)
                        left_neighbor = j

            if up_neighbor is not None:
                sat_src.append(i)
                sat_dst.append(up_neighbor)
            if down_neighbor is not None:
                sat_src.append(i)
                sat_dst.append(down_neighbor)
            if right_neighbor is not None:
                sat_src.append(i)
                sat_dst.append(right_neighbor)
            if left_neighbor is not None:
                sat_src.append(i)
                sat_dst.append(left_neighbor)

    else:
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

    # === Satellite → Gateway (Visibility edges) ===
    sat_to_gw_src = []
    sat_to_gw_dst = []

    for i, sat in enumerate(satellites):
        for gw_idx, is_visible in enumerate(sat['visible_gateways']):
            if is_visible and gw_idx in gw_id_map:
                sat_to_gw_src.append(i)
                sat_to_gw_dst.append(gw_idx)

    data['sat', 'connects', 'gateway'].edge_index = torch.tensor([sat_to_gw_src, sat_to_gw_dst], dtype=torch.long)

    # === Gateway → Cell (Service edges) ===
    gw_to_cell_src = []
    gw_to_cell_dst = []
    gw_to_cell_lookup = {}

    for gw in gateways:
        gw_idx = gw_id_map[gw['id']]
        gw_coord = (gw['latitude'], gw['longitude'])
        gw_to_cell_lookup[gw_idx] = []

        for cell in cells:
            cell_idx = cell_id_map[cell['cell_id']]
            cell_coord = (cell['latitude'], cell['longitude'])

            if geodesic(gw_coord, cell_coord).km <= gw_cell_distance_threshold_km:
                gw_to_cell_src.append(gw_idx)
                gw_to_cell_dst.append(cell_idx)
                gw_to_cell_lookup[gw_idx].append(cell_idx)

    data['gateway', 'serves', 'cell'].edge_index = torch.tensor([gw_to_cell_src, gw_to_cell_dst], dtype=torch.long)

    # === Satellite → Cell (Indirect coverage via gateways) ===
    sat_to_cell_src = []
    sat_to_cell_dst = []

    for i, sat in enumerate(satellites):
        for gw_idx, is_visible in enumerate(sat['visible_gateways']):
            if is_visible and gw_idx in gw_to_cell_lookup:
                for cell_idx in gw_to_cell_lookup[gw_idx]:
                    sat_to_cell_src.append(i)
                    sat_to_cell_dst.append(cell_idx)

    data['sat', 'covers', 'cell'].edge_index = torch.tensor([sat_to_cell_src, sat_to_cell_dst], dtype=torch.long)
    data['cell', 'covered_by', 'sat'].edge_index = torch.tensor([sat_to_cell_dst, sat_to_cell_src], dtype=torch.long)

    # === Visibility Masks (for training masking) ===
    num_sats = len(satellites)
    num_gws = len(gateways)
    num_cells = len(cells)

    sat_gateway_visibility = torch.zeros((num_sats, num_gws), dtype=torch.bool)
    sat_cell_visibility = torch.zeros((num_sats, num_cells), dtype=torch.bool)
    cell_sat_visibility = torch.zeros((num_cells, num_sats), dtype=torch.bool)

    for src, dst in zip(sat_to_gw_src, sat_to_gw_dst):
        sat_gateway_visibility[src, dst] = 1

    for src, dst in zip(sat_to_cell_src, sat_to_cell_dst):
        sat_cell_visibility[src, dst] = 1
        cell_sat_visibility[dst, src] = 1

    visibility_matrices = {
        'sat_gateway': sat_gateway_visibility,
        'sat_cell': sat_cell_visibility,
        'cell_sat': cell_sat_visibility
    }

    return data, visibility_matrices

