#2.ERA5_IVT_Calculation.py

import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import geopandas as gpd

# ==========================================
# Configuration & Paths
# ==========================================
# Update these paths to point to your ERA5 NetCDF files
u_flux_file = 'eastward_flux_1940_2025.nc'
v_flux_file = 'northward_flux_1940_2025.nc'
india_shapefile = 'india_st.shp'

# Parameters
analysis_months = [6, 7, 8, 9]  # JJAS
analysis_years = (1940, 2025)

# Spatial Bounding Box (India/Indian Ocean region)
min_lon, max_lon = 35, 120
min_lat, max_lat = -10, 40

def main():
    print("Loading ERA5 flux datasets...")
    try:
        ds_u = xr.open_dataset(u_flux_file)
        ds_v = xr.open_dataset(v_flux_file)
    except Exception as e:
        print(f"Error loading files: {e}")
        return

    # Handle dimension name differences (ERA5 often uses 'valid_time' or 'time')
    time_dim = 'valid_time' if 'valid_time' in ds_u.dims else 'time'
    lat_dim = 'latitude' if 'latitude' in ds_u.dims else 'lat'
    lon_dim = 'longitude' if 'longitude' in ds_u.dims else 'lon'

    print(f"Filtering data for months {analysis_months} and period {analysis_years}...")
    # Selection
    u_subset = ds_u.sel({
        time_dim: (ds_u[time_dim].dt.month.isin(analysis_months)) & 
                  (ds_u[time_dim].dt.year >= analysis_years[0]) & 
                  (ds_u[time_dim].dt.year <= analysis_years[1]),
        lat_dim: slice(40, -10),
        lon_dim: slice(min_lon, max_lon)
    })
    
    v_subset = ds_v.sel({
        time_dim: (ds_v[time_dim].dt.month.isin(analysis_months)) & 
                  (ds_v[time_dim].dt.year >= analysis_years[0]) & 
                  (ds_v[time_dim].dt.year <= analysis_years[1]),
        lat_dim: slice(40, -10),
        lon_dim: slice(min_lon, max_lon)
    })

    # Extract DataArrays (Variable names in ERA5: viwve, viwvn)
    u = u_subset.viwve if 'viwve' in u_subset.data_vars else u_subset[list(u_subset.data_vars)[0]]
    v = v_subset.viwvn if 'viwvn' in v_subset.data_vars else v_subset[list(v_subset.data_vars)[0]]

    print("Calculating IVT magnitude...")
    # IVT = sqrt(u^2 + v^2)
    ivt = np.sqrt(u**2 + v**2)
    ivt.name = "IVT"
    ivt.attrs['units'] = 'kg/m/s'

    # Time-mean for the analysis period
    ivt_mean = ivt.mean(dim=time_dim)

    # Save the raw calculated data to NetCDF for archival
    output_nc = 'era5_ivt_1940_2025.nc'
    print(f"Saving calculated IVT data to {output_nc}...")
    ivt.to_netcdf(output_nc)

    # Visualization
    print("Generating plot...")
    plt.figure(figsize=(12, 7))
    ax = plt.gca()

    # Plot spatial distribution
    ivt_mean.plot(ax=ax, cmap='jet', cbar_kwargs={'label': 'Mean IVT (kg/m/s)'})

    # Overlay Boundaries
    try:
        india_states = gpd.read_file(india_shapefile)
        india_states.plot(ax=ax, edgecolor='black', facecolor='none', linewidth=0.7)
    except:
        print("Warning: Shapefile not found. Plotting without boundaries.")

    plt.title(f'Mean ERA5 IVT JJAS ({analysis_years[0]}-{analysis_years[1]})')
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.grid(True, linestyle='--', alpha=0.5)

    # Save output plot
    output_plot = 'era5_ivt_mean_plot.png'
    plt.savefig(output_plot, dpi=300)
    print(f"Success! Plot saved as {output_plot} and data saved as {output_nc}")
    plt.show()

if __name__ == "__main__":
    main()
