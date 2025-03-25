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
import torch.nn.functional as F

# Constants
NUM_GATEWAYS = 54
NEIGHBOR_COUNT = 4

# Load trained Stage 1 model and its scaler
STAGE1_MODEL_PATH = 'stage_1_model.h5'
STAGE1_SCALER_PATH = 'stage_1_scaler.pkl'
stage1_model = load_model(STAGE1_MODEL_PATH)
with open(STAGE1_SCALER_PATH, 'rb') as f:
    stage1_scaler = pickle.load(f)

# Load Stage 2 scaler (for GNN input features)
STAGE2_SCALER_PATH = 'stage_2_scaler.pkl'
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

def prepare_input_for_gnn(data, top3_gateway_predictions):
    updated_x = data.x.clone()
    expanded_predictions = torch.zeros(updated_x.size(0), NUM_GATEWAYS, dtype=torch.float)
    top3_gateway_predictions = top3_gateway_predictions.long()
    for i in range(updated_x.size(0)):
        for gateway_idx in top3_gateway_predictions[i]:
            expanded_predictions[i, gateway_idx] = 1
    updated_x[:, -NUM_GATEWAYS:] = expanded_predictions
    return Data(x=updated_x, edge_index=data.edge_index)

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

    kdtree = KDTree(positions)
    neighbor_gateway_vector = []
    for idx, pos in enumerate(positions):
        _, neighbors = kdtree.query(pos, k=NEIGHBOR_COUNT + 1)
        neighbor_top3 = stage1_preds[neighbors[1:]]
        combined_gateways = (neighbor_top3.sum(axis=0) > 0).astype(np.float32)
        neighbor_gateway_vector.append(combined_gateways)
    neighbor_gateway_vector = np.array(neighbor_gateway_vector)

    satellite_features = np.hstack([positions, visible_gw, stage1_preds])
    node_features = np.hstack([satellite_features, neighbor_gateway_vector])

    node_features = stage2_scaler.transform(node_features)

    labels = np.argmax(np.vstack(df['optimal_gateway_matrix'].values), axis=1)
    labels = torch.tensor(labels, dtype=torch.long)

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
