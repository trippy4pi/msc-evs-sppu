#5.AR_Flood_Linkage.py

import os
import xarray as xr
import pandas as pd
import numpy as np
import geopandas as gpd
import matplotlib.pyplot as plt
from shapely.geometry import Polygon, MultiPolygon

# --- FILES (Assumed in current directory) ---
AR_MASK_FILE = 'ar_binary_mask.nc'
FLOOD_EVENT_FILE = 'floodevents_indofloods.csv'
METADATA_FILE = 'metadata_indofloods.csv'
CATCHMENT_SHP_ROOT = 'catchments_shapefiles_indofloods/'
INDIA_SHAPE = 'india_st.shp'

# --- CONFIGURATION ---
CATCHMENT_ID = 'INDOFLOODS-gauge-394'  # Example gauge station
JJAS_MONTHS = [6, 7, 8, 9]

def load_catchment_data(catchment_id):
    """Loads catchment geometry and metadata."""
    print(f"Loading data for catchment: {catchment_id}...")
    
    # Load Metadata
    df_meta = pd.read_csv(METADATA_FILE)
    meta = df_meta[df_meta['GaugeID'] == catchment_id].iloc[0]
    
    # Load Shapefile
    shp_path = os.path.join(CATCHMENT_SHP_ROOT, f"{catchment_id}.shp")
    gdf_catchment = gpd.read_file(shp_path)
    
    return gdf_catchment, meta

def calculate_spatial_overlap(ar_mask_da, catchment_geom):
    """
    Checks daily overlap between the AR mask and catchment geometry.
    Returns a pandas Series with dates as index and overlap (1/0) as values.
    """
    print("Calculating spatial overlap (AR vs. Catchment)...")
    
    overlap_values = []
    dates = []
    
    # Calculate resolution for polygon grid
    lat_res = abs(ar_mask_da.latitude.diff(dim='latitude').mean().item())
    lon_res = abs(ar_mask_da.longitude.diff(dim='longitude').mean().item())
    h_lat = lat_res / 2
    h_lon = lon_res / 2
    
    # Get the single polygon/multipolygon from the catchment
    target_geom = catchment_geom.geometry.iloc[0]

    for time_step in ar_mask_da.valid_time:
        current_time = pd.to_datetime(time_step.item())
        ar_2d = ar_mask_da.sel(valid_time=time_step)
        
        # Get coordinates where AR mask is 1
        ar_lat_idx, ar_lon_idx = np.where(ar_2d == 1)
        
        ar_polygons = []
        for lat_idx, lon_idx in zip(ar_lat_idx, ar_lon_idx):
            c_lat = ar_2d.latitude.values[lat_idx]
            c_lon = ar_2d.longitude.values[lon_idx]
            
            # Create grid cell polygon
            cell = Polygon([
                (c_lon - h_lon, c_lat - h_lat),
                (c_lon + h_lon, c_lat - h_lat),
                (c_lon + h_lon, c_lat + h_lat),
                (c_lon - h_lon, c_lat + h_lat),
                (c_lon - h_lon, c_lat - h_lat)
            ])
            ar_polygons.append(cell)
            
        if ar_polygons:
            ar_multipoly = MultiPolygon(ar_polygons)
            is_overlapping = target_geom.intersects(ar_multipoly)
            overlap_values.append(1 if is_overlapping else 0)
        else:
            overlap_values.append(0)
            
        dates.append(current_time)
        
    return pd.Series(overlap_values, index=dates)

def create_flood_time_series(catchment_id, date_index):
    """Converts the flood event catalog to a daily binary time series."""
    print("Creating flood time series...")
    df_floods = pd.read_csv(FLOOD_EVENT_FILE)
    
    # Filter for specific gauge
    gauge_floods = df_floods[df_floods['EventID'].str.startswith(catchment_id)].copy()
    gauge_floods['Start Date'] = pd.to_datetime(gauge_floods['Start Date'])
    gauge_floods['End Date'] = pd.to_datetime(gauge_floods['End Date'])
    
    # Initialize series
    flood_series = pd.Series(0, index=date_index)
    
    # Mark flood days
    for _, row in gauge_floods.iterrows():
        flood_series.loc[row['Start Date']:row['End Date']] = 1
        
    # Filter for JJAS
    flood_series = flood_series[flood_series.index.month.isin(JJAS_MONTHS)]
    return flood_series

def main():
    try:
        # 1. Load Data
        ds_ar = xr.open_dataset(AR_MASK_FILE)
        ar_mask = ds_ar['AR_mask'] # Adjust if name differs
        
        gdf_catchment, meta = load_catchment_data(CATCHMENT_ID)
        
        # 2. Slice AR mask to available catchment date range
        start_date = meta['Start_date']
        end_date = meta['End_date']
        ar_mask_sub = ar_mask.sel(valid_time=slice(start_date, end_date))
        
        # 3. Calculate Spatial Overlap
        ar_overlap_series = calculate_spatial_overlap(ar_mask_sub, gdf_catchment)
        
        # 4. Create Flood Series
        flood_series = create_flood_time_series(CATCHMENT_ID, ar_overlap_series.index)
        
        # 5. Linkage Logic (Lagged by 1 day)
        # Shift AR overlap forward (so AR on day T is matched with flood on day T+1)
        ar_overlap_lagged = ar_overlap_series.shift(1)
        
        # Linked events: (AR on prev day) AND (Flood on current day)
        linked_floods = (ar_overlap_lagged == 1) & (flood_series == 1)
        total_linked = linked_floods.sum()
        total_floods = flood_series.sum()
        
        # 6. Statistics
        print("\n--- LINKAGE RESULTS ---")
        print(f"Catchment ID: {CATCHMENT_ID}")
        print(f"Total JJAS Flood Days: {total_floods}")
        print(f"Total AR-Linked Flood Days (Day After AR): {total_linked}")
        
        if total_floods > 0:
            percentage = (total_linked / total_floods) * 100
            print(f"Percentage of Floods Linked to ARs: {percentage:.2f}%")
        
        # 7. Visualization
        plt.figure(figsize=(12, 6))
        flood_series.plot(color='grey', alpha=0.5, label='Total Flood Days')
        linked_floods.astype(int).plot(color='red', linewidth=1.5, label='AR-Linked Floods')
        
        plt.title(f'JJAS Flood Events for {CATCHMENT_ID} Linked to ARs\n(Total Linked: {total_linked})')
        plt.xlabel('Date')
        plt.ylabel('Event (1=Yes, 0=No)')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)
        
        output_plot = f'ar_flood_linkage_{CATCHMENT_ID}.png'
        plt.savefig(output_plot, dpi=300)
        print(f"\nPlot saved as {output_plot}")
        
    except Exception as e:
        print(f"Error during linkage analysis: {e}")

if __name__ == "__main__":
    main()
