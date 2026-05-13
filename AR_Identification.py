#3.AR_Identification.py
"""
Atmospheric River (AR) Identification for India
==========================================================
Based on: Mahto et al. (2023), Communications Earth & Environment
"Atmospheric rivers that make landfall in India are associated with flooding"
DOI: https://doi.org/10.1038/s43247-023-00775-9
"""

import numpy as np
import xarray as xr
import pandas as pd
import matplotlib.pyplot as plt
import geopandas as gpd
from scipy import ndimage
from skimage import measure
import regionmask
import warnings

warnings.filterwarnings("ignore")

# ==========================================
# Configuration & Paths
# ==========================================
# Paths to your input files
ivt_file = 'era5_ivt_1940_2025.nc'  # The one created in the previous script ERA5_IVT_Calculation.py
qu_file = 'eastward_flux_1940_2025.nc'
qv_file = 'northward_flux_1940_2025.nc'
india_shapefile = 'india_st.shp'

# Detection Thresholds (Mahto et al. 2023 Defaults)
MIN_IVT_ABS = 150.0        # kg/m/s absolute lower bound
MIN_INDIA_AREA = 50_000    # km^2 coverage over India
MIN_LENGTH = 2000          # km minimum major axis length
MIN_LW_RATIO = 2.0         # Length/Width ratio
MAX_CIRCULARITY = 0.5      # Avoid circular cyclones (1=circle)

# ==========================================
# Logic Functions
# ==========================================

def grid_cell_area_km2(lat, lon, res_deg=0.25):
    R = 6371.0  # Earth radius
    dp = np.radians(res_deg)
    dl = np.radians(res_deg)
    area_1d = R**2 * np.abs(np.cos(np.radians(lat))) * dp * dl
    return np.outer(area_1d, np.ones(len(lon)))

def compute_p85_climatology(ivt_da):
    """Compute 85th-percentile for each Day of Year using a 15-day window."""
    print("Computing 85th-percentile climatology (this may take a minute)...")
    doys = ivt_da['valid_time'].dt.dayofyear
    unique_doys = np.unique(doys)
    p85_lookup = {}
    
    for doy in unique_doys:
        # 15-day window centered on DOY
        window = np.where((doys >= doy - 7) & (doys <= doy + 7))[0]
        subset = ivt_da.values[window]
        p85_lookup[int(doy)] = np.nanpercentile(subset, 85, axis=0)
    
    # Expand lookup to full time series
    p85_values = np.array([p85_lookup[int(d)] for d in doys])
    return xr.DataArray(p85_values, coords=ivt_da.coords, dims=ivt_da.dims)

def get_ar_properties(labeled_arr, cluster_id, ivt_2d, lat_2d, area_2d, india_mask, cfg):
    """Checks a candidate moisture plume against scientific AR criteria."""
    mask = (labeled_arr == cluster_id)
    
    # 1. India Coverage check
    india_overlap = mask & india_mask
    india_area = area_2d[india_overlap].sum()
    if india_area < cfg['min_india_area']: return None
    
    # 2. Geometric check
    props = measure.regionprops(mask.astype(np.uint8))
    if not props: return None
    prop = props[0]
    
    # Rough km conversion (approx 27.75km per 0.25 deg)
    px_km = 27.75 
    length = prop.major_axis_length * px_km
    width = prop.minor_axis_length * px_km if prop.minor_axis_length > 0 else 1.0
    
    if length < cfg['min_length'] or (length/width) < cfg['min_lw_ratio']:
        return None
    
    # 3. Circularity (Exclude cyclones)
    circularity = (4 * np.pi * prop.area) / (prop.perimeter**2) if prop.perimeter > 0 else 0
    if circularity > cfg['max_circularity']:
        return None

    return {
        "mask": mask,
        "mean_ivt_india": np.nanmean(ivt_2d[india_overlap]),
        "india_area": india_area,
        "length": length
    }

def main():
    print("Loading datasets...")
    try:
        ds_ivt = xr.open_dataset(ivt_file)
        ivt = ds_ivt.IVT
        lat, lon = ivt.latitude.values, ivt.longitude.values
    except Exception as e:
        print(f"Error: {e}. Ensure {ivt_file} exists.")
        return

    # Prep Spatial Helpers
    area_2d = grid_cell_area_km2(lat, lon)
    countries = regionmask.defined_regions.natural_earth_v5_0_0.countries_110
    mask_int = countries.mask(lon, lat)
    india_mask = (mask_int == countries.map_keys("India")).values

    # Step 1: Calculate Percentile Thresholds
    p85_da = compute_p85_climatology(ivt)
    binary_mask = (ivt > p85_da) & (ivt > MIN_IVT_ABS)

    # Step 2: Identify AR Objects
    print(f"Identifying AR events across {len(ivt.valid_time)} days...")
    cfg = {'min_india_area': MIN_INDIA_AREA, 'min_length': MIN_LENGTH, 
           'min_lw_ratio': MIN_LW_RATIO, 'max_circularity': MAX_CIRCULARITY}
    
    ar_daily_mask = np.zeros_like(ivt.values, dtype=np.uint8)
    daily_records = []

    for t in range(len(ivt.valid_time)):
        labeled, n_labels = ndimage.label(binary_mask.values[t])
        for cid in range(1, n_labels + 1):
            props = get_ar_properties(labeled, cid, ivt.values[t], lat, area_2d, india_mask, cfg)
            if props:
                ar_daily_mask[t] |= props['mask']
                daily_records.append({
                    "date": pd.Timestamp(ivt.valid_time.values[t]),
                    "mean_ivt_india": props['mean_ivt_india'],
                    "india_area_km2": props['india_area'],
                    "length_km": props['length']
                })

    # Step 3: Create Catalog Table
    df = pd.DataFrame(daily_records)
    if not df.empty:
        # Simple grouping into events (consecutive days)
        df['group'] = (df['date'].diff().dt.days > 1).cumsum()
        catalog = df.groupby('group').agg({
            'date': ['min', 'max', 'count'],
            'mean_ivt_india': 'mean',
            'india_area_km2': 'mean',
            'length_km': 'mean'
        })
        catalog.columns = ['start_date', 'end_date', 'duration_days', 'avg_ivt', 'avg_area', 'avg_length']
        catalog.to_csv('ar_event_catalog.csv', index=False)
        print("Success! Catalog saved to ar_event_catalog.csv")
    else:
        print("No AR events detected with current thresholds.")

    # Step 4: Plot AR Frequency
    print("Generating AR Frequency Map...")
    freq = (ar_daily_mask.sum(axis=0) / len(ivt.valid_time)) * 100
    
    plt.figure(figsize=(12, 8))
    plt.contourf(lon, lat, freq, cmap='Blues', levels=np.linspace(0, 15, 11))
    plt.colorbar(label='AR Frequency (% of days)')
    
    try:
        india_states = gpd.read_file(india_shapefile)
        india_states.plot(ax=plt.gca(), edgecolor='black', facecolor='none', alpha=0.5)
    except: pass
    
    plt.title('Identified Atmospheric River (AR) Frequency - JJAS 1940-2025')
    plt.savefig('ar_frequency_map.png', dpi=300)
    print("Success! Frequency map saved to ar_frequency_map.png")

if __name__ == "__main__":
    main()
