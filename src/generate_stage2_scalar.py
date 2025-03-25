import os
import pickle
import numpy as np
import pandas as pd
import ast
from sklearn.preprocessing import StandardScaler
from scipy.spatial import KDTree
from tensorflow.keras.models import load_model

NUM_GATEWAYS = 54
DATA_FOLDER = r"C:\Users\aruna\Desktop\MS Thesis\Real Data\Final folder real data"
STAGE1_MODEL_PATH = 'stage_1_model.h5'
STAGE1_SCALER_PATH = 'stage_1_scaler.pkl'

# Load models
stage1_model = load_model(STAGE1_MODEL_PATH)
with open(STAGE1_SCALER_PATH, 'rb') as f:
    stage1_scaler = pickle.load(f)

def parse_matrix(matrix_str):
    try:
        return np.array(ast.literal_eval(str(matrix_str)), dtype=np.float32)
    except:
        return np.zeros(NUM_GATEWAYS, dtype=np.float32)

def get_top3_binary(model, X):
    X_scaled = stage1_scaler.transform(X)
    preds = model.predict(X_scaled, verbose=0)
    top3_idx = np.argsort(preds, axis=1)[:, -3:]
    binary_preds = np.zeros_like(preds)
    for i, idx in enumerate(top3_idx):
        binary_preds[i, idx] = 1
    return binary_preds

# Collect training files
file_list = sorted([
    os.path.join(DATA_FOLDER, f) for f in os.listdir(DATA_FOLDER) if f.endswith('.csv')
])[:10]
train_size = int(0.8 * len(file_list))
train_files = file_list[:train_size]

all_node_features = []

for file in train_files:
    df = pd.read_csv(file, usecols=[
        'Latitude', 'Longitude', 'Altitude',
        'visible_gateway_matrix', 'optimal_gateway_matrix'
    ])
    df['visible_gateway_matrix'] = df['visible_gateway_matrix'].apply(parse_matrix)
    df['optimal_gateway_matrix'] = df['optimal_gateway_matrix'].apply(parse_matrix)

    pos = df[['Latitude', 'Longitude', 'Altitude']].values
    visible = np.vstack(df['visible_gateway_matrix'].values)
    X_stage1 = np.hstack([pos, visible])
    top3 = get_top3_binary(stage1_model, X_stage1)

    kdtree = KDTree(pos)
    neighbor_features = []
    for i, p in enumerate(pos):
        _, neighbors = kdtree.query(p, k=5)
        neighbor_top3 = top3[neighbors[1:]]
        combined = (neighbor_top3.sum(axis=0) > 0).astype(np.float32)
        neighbor_features.append(combined)
    neighbor_features = np.array(neighbor_features)

    sat_features = np.hstack([pos, visible, top3])
    node_features = np.hstack([sat_features, neighbor_features])
    all_node_features.append(node_features)

# Fit scaler
all_node_features = np.vstack(all_node_features)
scaler = StandardScaler()
scaler.fit(all_node_features)

# Save it
with open('stage_2_scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)

print("✅ Saved stage_2_scaler.pkl")
