import os
import ast
import numpy as np
import pandas as pd
import pickle
from scipy.spatial import KDTree
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset
from torch_geometric.data import Data
import torch
from tensorflow.keras.models import load_model
from math import radians, cos, sin, sqrt, atan2

# Constants
NUM_GATEWAYS = 54
NEIGHBOR_COUNT = 4
CELL_VISIBILITY_RADIUS_KM = 1200  # Adjustable based on satellite altitude and FoV

# === Paths ===
GATEWAY_CSV_PATH = r"C:\Users\aruna\Desktop\MS Thesis\Real Data\df_gw.csv"
CELL_CSV_PATH = r"C:\Users\aruna\Desktop\MS Thesis\Real Data\cells_with_gateways.csv"
STAGE1_MODEL_PATH = 'stage_1_model.h5'
STAGE1_SCALER_PATH = 'stage_1_scaler.pkl'
STAGE2_SCALER_PATH = 'stage_2_scaler.pkl'

# === Static Data ===
gateway_df = pd.read_csv(GATEWAY_CSV_PATH)
GATEWAY_POSITIONS = {
    int(row['gw_id']): [row['latitude'], row['longitude']]
    for _, row in gateway_df.iterrows()
}

cell_df = pd.read_csv(CELL_CSV_PATH)
CELL_POSITIONS = {
    int(row['idx']): [row['lat'], row['lng']]
    for _, row in cell_df.iterrows()
}
CELL_GATEWAY_MAP = {
    int(row['idx']): [int(row['closest_gw_id']), int(row['second_closest_gw_id'])]
    for _, row in cell_df.iterrows()
}

# === Load Models ===
stage1_model = load_model(STAGE1_MODEL_PATH)
with open(STAGE1_SCALER_PATH, 'rb') as f:
    stage1_scaler = pickle.load(f)

with open(STAGE2_SCALER_PATH, 'rb') as f:
    stage2_scaler = pickle.load(f)

# === Helper Functions ===
def parse_matrix(matrix_str):
    try:
        return np.array(ast.literal_eval(str(matrix_str)), dtype=np.float32)
    except Exception:
        return np.zeros(NUM_GATEWAYS, dtype=np.float32)

def get_top3_prediction_binary(model, X):
    X_scaled = stage1_scaler.transform(X)
    preds = model.predict(X_scaled, verbose=0)
    top3_idx = np.argsort(preds, axis=1)[:, -3:]
    binary_preds = np.zeros_like(preds)
    for i, idx in enumerate(top3_idx):
        binary_preds[i, idx] = 1
    return binary_preds

def haversine_distance(lat1, lon1, lat2, lon2):
    R = 6371.0
    dlat = radians(lat2 - lat1)
    dlon = radians(lon2 - lon1)
    a = sin(dlat/2)**2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon/2)**2
    return R * 2 * atan2(sqrt(a), sqrt(1 - a))

def count_visible_cells(sat_lat, sat_lon, radius_km=CELL_VISIBILITY_RADIUS_KM):
    count = 0
    for cell_lat, cell_lon in CELL_POSITIONS.values():
        dist = haversine_distance(sat_lat, sat_lon, cell_lat, cell_lon)
        if dist <= radius_km:
            count += 1
    return count

def build_graph_from_file(file_path):
    df = pd.read_csv(file_path, usecols=[
        'feed_sat', 'Latitude', 'Longitude', 'Altitude',
        'visible_gateway_matrix', 'optimal_gateway_matrix'
    ])
    if df.empty:
        return None

    df['visible_gateway_matrix'] = df['visible_gateway_matrix'].apply(parse_matrix)
    df['optimal_gateway_matrix'] = df['optimal_gateway_matrix'].apply(parse_matrix)

    positions = df[['Latitude', 'Longitude', 'Altitude']].values
    visible_gw = np.vstack(df['visible_gateway_matrix'].values)
    stage1_input = np.hstack([positions, visible_gw])
    stage1_preds = get_top3_prediction_binary(stage1_model, stage1_input)

    # Neighbor-aware gateway influence
    kdtree = KDTree(positions)
    neighbor_gateway_vector = []
    for idx, pos in enumerate(positions):
        _, neighbors = kdtree.query(pos, k=NEIGHBOR_COUNT + 1)
        neighbor_top3 = stage1_preds[neighbors[1:]]
        combined_gateways = (neighbor_top3.sum(axis=0) > 0).astype(np.float32)
        neighbor_gateway_vector.append(combined_gateways)
    neighbor_gateway_vector = np.array(neighbor_gateway_vector)

    # === Cell visibility summary ===
    visible_cell_counts = []
    for lat, lon, _ in positions:
        cell_count = count_visible_cells(lat, lon)
        visible_cell_counts.append([cell_count])
    visible_cell_counts = np.array(visible_cell_counts)

    # === Feature Construction ===
    satellite_features = np.hstack([
        positions,
        visible_gw,
        stage1_preds,
        neighbor_gateway_vector,
        visible_cell_counts  # new cell count feature
    ])
    node_features = stage2_scaler.transform(satellite_features)

    # === Labels ===
    labels = np.vstack(df['optimal_gateway_matrix'].values).astype(np.float32)
    labels = torch.tensor(labels, dtype=torch.float)

    # === Edges (from KDTree) ===
    edges = []
    for idx, pos in enumerate(positions):
        _, neighbors = kdtree.query(pos, k=NEIGHBOR_COUNT + 1)
        for neighbor_idx in neighbors[1:]:
            edges.append([idx, neighbor_idx])
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()

    return Data(
        x=torch.tensor(node_features, dtype=torch.float),
        edge_index=edge_index,
        y=labels
    )

# === Dataset Class ===
class SatelliteDataset(Dataset):
    def __init__(self, file_list):
        self.file_list = file_list

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        file_path = self.file_list[idx]
        graph = build_graph_from_file(file_path)
        if graph is None:
            raise ValueError(f"Error processing file: {file_path}")
        return graph
