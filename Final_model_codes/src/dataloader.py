import os
import pandas as pd
from geopy.distance import geodesic

# === Constants ===
GATEWAY_VISIBILITY_THRESHOLD_KM = 1200  # Max sat→gateway visibility distance (km)

# === Helper: Load static cell coordinates (lat/lon) ===
def load_static_cell_coords(cell_file):
    df = pd.read_csv(cell_file)
    df.columns = df.columns.str.lower()
    if 'cell_id' not in df.columns:
        raise ValueError("Expected 'cell_id' column in cell_file")
    return df.set_index('cell_id')[['latitude', 'longitude']].to_dict(orient='index')

# === Main: Load everything for a snapshot ===
def load_all_data(snapshot_folder, cell_file, gateway_file, snapshot_filename=None, debug_columns=False):
    """
    Loads satellites, gateways, and cells for a given snapshot.
    
    Args:
        snapshot_folder: folder containing satellite-cell snapshot CSVs
        cell_file: static cell information (lat, lon)
        gateway_file: gateway locations
        snapshot_filename: optional specific snapshot file
        debug_columns: print column names for inspection

    Returns:
        satellites, gateways, cells (lists of dicts)
    """
    cell_coord_lookup = load_static_cell_coords(cell_file)
    gateways = load_gateways(gateway_file)

    if snapshot_filename:
        file_path = os.path.join(snapshot_folder, snapshot_filename)
    else:
        selected_file = select_file_from_folder(snapshot_folder)
        file_path = os.path.join(snapshot_folder, selected_file)

    satellites, cells = load_satellite_snapshot(file_path, gateways, cell_coord_lookup, debug_columns)
    return satellites, gateways, cells

# === Helper: Load gateways ===
def load_gateways(gateway_file):
    df = pd.read_csv(gateway_file)
    df.columns = df.columns.str.lower()
    if not {'gw_id', 'latitude', 'longitude'}.issubset(df.columns):
        raise ValueError("Expected 'gw_id', 'latitude', and 'longitude' columns in gateway_file")

    gateways = [
        {
            'id': int(row['gw_id']),
            'latitude': row['latitude'],
            'longitude': row['longitude']
        }
        for _, row in df.iterrows()
    ]
    return gateways

# === Helper: Auto-select snapshot file ===
def select_file_from_folder(folder_path):
    files = sorted([f for f in os.listdir(folder_path) if f.endswith(".csv")])
    if not files:
        raise FileNotFoundError(f"No CSV files found in {folder_path}")
    return files[0]

# === Helper: Load satellite and cell data from snapshot ===
def load_satellite_snapshot(file_path, gateways, cell_coord_lookup, debug_columns=False):
    df = pd.read_csv(file_path)
    df.columns = df.columns.str.lower()

    if debug_columns:
        print(f"\nSnapshot file columns ({os.path.basename(file_path)}):")
        print(df.columns.tolist())

    if 'feed_sat' in df.columns:
        df.rename(columns={'feed_sat': 'sat_id'}, inplace=True)

    # === Load cell demand information ===
    if 'cell_id' not in df.columns or 'demand' not in df.columns:
        raise ValueError("Snapshot file must contain 'cell_id' and 'demand' columns")
    
    cell_demand_df = df[['cell_id', 'demand']].groupby('cell_id').mean().reset_index()

    cells = []
    for row in cell_demand_df.itertuples(index=False):
        cell_id = int(row.cell_id)
        if cell_id in cell_coord_lookup:
            coords = cell_coord_lookup[cell_id]
            cells.append({
                'cell_id': cell_id,
                'latitude': coords['latitude'],
                'longitude': coords['longitude'],
                'demand': row.demand,
                'timestamp': None
            })

    # === Load satellite positions and compute visibility ===
    sats_df = df[['sat_id', 'latitude', 'longitude', 'altitude']].drop_duplicates()
    if sats_df.empty:
        raise ValueError(f"No satellite data found in {file_path}")

    satellites = []
    for row in sats_df.itertuples(index=False):
        sat_id = int(row.sat_id)
        sat_coord = (row.latitude, row.longitude)

        visible_gateways = []
        for gw in gateways:
            gw_coord = (gw['latitude'], gw['longitude'])
            distance_km = geodesic(sat_coord, gw_coord).km
            visible_gateways.append(1 if distance_km <= GATEWAY_VISIBILITY_THRESHOLD_KM else 0)

        satellites.append({
            'sat_id': sat_id,
            'latitude': row.latitude,
            'longitude': row.longitude,
            'altitude': row.altitude,
            'visible_gateways': visible_gateways,
            'memory': [],  # Optional temporal memory
            'timestamp': None
        })

    return satellites, cells
