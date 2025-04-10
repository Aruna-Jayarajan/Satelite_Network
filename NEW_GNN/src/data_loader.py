import os
import pandas as pd
from geopy.distance import geodesic

def load_static_cell_coords(cell_file):
    df = pd.read_csv(cell_file)
    df.columns = df.columns.str.lower()
    return df.set_index('cell_id')[['latitude', 'longitude']].to_dict(orient='index')

def load_all_data(snapshot_folder, cell_file, gateway_file, snapshot_filename=None, debug_columns=False):
    cell_coord_lookup = load_static_cell_coords(cell_file)
    gateways = load_gateways(gateway_file)

    if snapshot_filename:
        file_path = os.path.join(snapshot_folder, snapshot_filename)
    else:
        selected_file = select_file_from_folder(snapshot_folder)
        file_path = os.path.join(snapshot_folder, selected_file)

    satellites, cells = load_satellite_snapshot(file_path, gateways, cell_coord_lookup, debug_columns)
    return satellites, gateways, cells

def load_gateways(gateway_file):
    df = pd.read_csv(gateway_file)
    df.columns = df.columns.str.lower()
    gateways = [
        {'id': int(row['gw_id']), 'latitude': row['latitude'], 'longitude': row['longitude']}
        for _, row in df.iterrows()
    ]
    return gateways

def select_file_from_folder(folder_path):
    files = sorted([f for f in os.listdir(folder_path) if f.endswith(".csv")])
    if not files:
        raise FileNotFoundError(f"No CSV files found in {folder_path}")
    return files[0]

def load_satellite_snapshot(file_path, gateways, cell_coord_lookup, debug_columns=False):
    df = pd.read_csv(file_path)
    df.columns = df.columns.str.lower()

    if debug_columns:
        print(f"\nColumns in snapshot file '{os.path.basename(file_path)}':")
        print(df.columns.tolist())

    # Optional renaming for older naming conventions
    if 'feed_sat' in df.columns:
        df.rename(columns={'feed_sat': 'sat_id'}, inplace=True)

    # === Load cell data ===
    cell_demand_df = df[['cell_id', 'demand']].groupby('cell_id').mean().reset_index()
    cells = []
    for row in cell_demand_df.itertuples(index=False):
        if row.cell_id in cell_coord_lookup:
            coords = cell_coord_lookup[row.cell_id]
            cells.append({
                'cell_id': row.cell_id,
                'latitude': coords['latitude'],
                'longitude': coords['longitude'],
                'demand': row.demand
            })

    # === Load satellite data ===
    sats_df = df[['sat_id', 'latitude', 'longitude', 'altitude']].drop_duplicates()

    satellites = []
    for row in sats_df.itertuples(index=False):
        sat_coord = (row.latitude, row.longitude)

        # Dynamically compute visible gateways within 1000 km
        visible_gateways = []
        for gw in gateways:
            gw_coord = (gw['latitude'], gw['longitude'])
            distance_km = geodesic(sat_coord, gw_coord).km
            visible_gateways.append(1 if distance_km <= 1200 else 0)

        satellites.append({
            'sat_id': int(row.sat_id),
            'latitude': row.latitude,
            'longitude': row.longitude,
            'altitude': row.altitude,
            'visible_gateways': visible_gateways,
            'optimal_gateway': []  # Optional: clear this out unless used later
        })

    return satellites, cells
