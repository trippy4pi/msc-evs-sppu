#4.Orographic Forcing Analysis Script

import os
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import geopandas as gpd
import datetime as dt

# --- FILES ---
IVT_FILE = 'era5_ivt_1940_2025.nc' # The one created in the previous script ERA5_IVT_Calculation.py
ORO_FILE = 'ERA5_Geopotential.nc'
SHAPE_FILE = 'india_st.shp'

# Constants from original notebook
INSET_MIN_LAT = 8
INSET_MAX_LAT = 20
MIN_LON = 60
G_CONST = 9.80665  # Standard gravity constant for geopotential height conversion

def load_datasets():
    """Loads required ERA5 datasets."""
    print("Loading datasets...")
    ds_ivt = xr.open_dataset(IVT_FILE)
    ds_oro_raw = xr.open_dataset(ORO_FILE)
    
    # Process ERA5 Geopotential (z) to Height (Orography)
    oro = ds_oro_raw.z / G_CONST
    oro.attrs['units'] = 'm'
    oro.attrs['long_name'] = 'ERA5 Orography (Geopotential Height)'
    
    return ds_ivt, oro

def plot_ivt_map_with_inset(ivt_da, date_str, shapefile_path):
    """
    Plots IVT map for a specific date with an inset box.
    """
    print(f"Plotting IVT map for {date_str}...")
    
    # Parse date
    date_obj = dt.datetime.strptime(date_str, "%Y-%m-%d")
    ivt_date = ivt_da.sel(valid_time=date_obj)
    
    fig, ax = plt.subplots(figsize=(16, 8))
    
    # Colormap settings
    levels = np.arange(300, 1301, 50)
    cmap = plt.get_cmap('jet')
    cmap.set_under('white')
    cmap.set_over('DarkRed')
    
    # Plot IVT
    im = ivt_date.plot(ax=ax, cmap=cmap, levels=levels, vmin=300, 
                       cbar_kwargs={'extend': 'max', 'label': 'IVT (kg/m/s)'})
    
    # Overlay India Shapefile
    if os.path.exists(shapefile_path):
        india_states = gpd.read_file(shapefile_path)
        india_states.plot(ax=ax, edgecolor='black', facecolor='none', linewidth=0.7)
    
    # Add Red Inset Box
    max_lon = ivt_date.longitude.max().item()
    inset_box = plt.Rectangle((MIN_LON, INSET_MIN_LAT), max_lon - MIN_LON, INSET_MAX_LAT - INSET_MIN_LAT,
                              facecolor='none', edgecolor='red', linewidth=3, linestyle='-')
    ax.add_patch(inset_box)
    
    ax.set_title(f'Integrated Vapor Transport (IVT) - {date_str}')
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    plt.grid(True, linestyle='--', alpha=0.6)
    
    output_path = f"ivt_map_{date_str}.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()

def plot_longitude_cross_section(ivt_da, oro_da, date_str):
    """
    Plots mean height and IVT across longitude within the defined latitude inset.
    """
    print(f"Plotting longitudinal cross-section for {date_str}...")
    
    # Parse date
    date_obj = dt.datetime.strptime(date_str, "%Y-%m-%d")
    
    # 1. Selection & Cropping
    ivt_sub = ivt_da.sel(valid_time=date_obj, 
                         latitude=slice(INSET_MAX_LAT, INSET_MIN_LAT),
                         longitude=slice(MIN_LON, None))
    
    # Handle ERA5 Orography selection
    if 'valid_time' in oro_da.coords:
        try:
            oro_sub = oro_da.sel(valid_time=date_obj, method='nearest')
        except:
            oro_sub = oro_da.isel(valid_time=0)
    else:
        oro_sub = oro_da
        
    oro_sub = oro_sub.sel(latitude=slice(INSET_MAX_LAT, INSET_MIN_LAT),
                          longitude=slice(MIN_LON, None))

    # 2. Latitudinal Means
    mean_ivt = ivt_sub.mean(dim='latitude')
    mean_oro = oro_sub.mean(dim='latitude').squeeze()
    
    # Ensure plots use a consistent longitude range (IVT's range)
    target_lons = mean_ivt.longitude
    mean_oro_interp = mean_oro.interp(longitude=target_lons, method='linear')

    # 3. Plotting
    fig, ax1 = plt.subplots(figsize=(12, 6))
    
    # Left Axis: Heights
    l1, = ax1.plot(target_lons, mean_oro_interp, color='saddlebrown', linewidth=2, label='ERA5 Orography (Height)')
    
    ax1.set_xlabel('Longitude')
    ax1.set_ylabel('Height (m)')
    ax1.tick_params(axis='y', labelcolor='black')
    
    # Right Axis: IVT
    ax2 = ax1.twinx()
    l2, = ax2.plot(target_lons, mean_ivt, color='blue', linewidth=2, label='Mean IVT')
    ax2.set_ylabel('Mean IVT (kg/m/s)', color='blue')
    ax2.tick_params(axis='y', labelcolor='blue')
    
    # Legend
    lines = [l1, l2]
    labels = [line.get_label() for line in lines]
    ax1.legend(lines, labels, loc='upper right', frameon=False)
    
    plt.title(f'Mean IVT & Orography over Longitude ({date_str})\nWithin {INSET_MIN_LAT}°N - {INSET_MAX_LAT}°N')
    plt.grid(True, linestyle='--', alpha=0.7)
    
    output_path = f"oro_ivt_cross_section_{date_str}.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()

def main():
    try:
        ds_ivt, oro = load_datasets()
        
        # Select target date
        target_date = "2024-07-24"
        
        # Generate Plots
        plot_ivt_map_with_inset(ds_ivt.IVT, target_date, SHAPE_FILE)
        plot_longitude_cross_section(ds_ivt.IVT, oro, target_date)
        
        print("\nAnalysis complete. Plot files generated in current directory.")
        
    except Exception as e:
        print(f"Error during analysis: {e}")

if __name__ == "__main__":
    main()