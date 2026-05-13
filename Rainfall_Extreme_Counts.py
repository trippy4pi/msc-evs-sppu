#1.Rainfall_Extreme_Counts.py

import xarray as xr
import matplotlib.pyplot as plt
import geopandas as gpd
import numpy as np

# ==========================================
# Configuration & Paths
# ==========================================
# Note: Update these paths to match your local file locations
rf_file = 'RF_p25_1901_2025.nc'
india_st_shape = 'india_st.shp'

# Parameters for Analysis
month_for_threshold = [6, 7, 8, 9]  # JJAS
year_for_threshold = (1901, 1950)     # Historical Baseline
month_for_analysis = [6, 7, 8, 9]   # JJAS
year_for_analysis = (1951, 2025)     # Analysis Period

# Threshold Configuration
THRESHOLD_MODE = "percentile"  # Options: "percentile" or "absolute"
THRESHOLD_VALUE = 150         # mm/day (used if mode is "absolute")
THRESHOLD_PERCENTILE = 99     # percentile (used if mode is "percentile")

def calculate_extreme_events_per_jjas(rf_data, threshold_data, analysis_months, analysis_year):
    """
    Calculates the number of extreme rainfall events per JJAS season.

    Args:
        rf_data (xr.DataArray): The rainfall data (time, lat, lon).
        threshold_data (xr.DataArray): The 2D threshold array (lat, lon).
        analysis_months (list): A list of months to include in the analysis.
        analysis_year (tuple): Start and end year for analysis.

    Returns:
        xr.DataArray: Count of extreme events per year (year, lat, lon).
    """
    # Select data for the analysis months and years
    rf_jjas = rf_data.sel(time=(rf_data['time'].dt.month.isin(analysis_months)) &
                                 (rf_data['time'].dt.year >= analysis_year[0]) &
                                 (rf_data['time'].dt.year <= analysis_year[1]))

    # Identify extreme events (where rf > threshold)
    is_extreme = (rf_jjas > threshold_data)

    # Convert boolean to float to allow NaN values
    extreme_events_numeric = is_extreme.astype(float)

    # Create a mask to preserve NaNs from the original data
    nan_mask = rf_jjas.isnull()
    
    # If threshold_data is an xarray object (spatial threshold), also mask its NaNs
    if hasattr(threshold_data, "isnull"):
        nan_mask = nan_mask | threshold_data.isnull()
        
    extreme_events = extreme_events_numeric.where(~nan_mask)

    # Group by year and sum the extreme events
    extreme_event_per_jjas = extreme_events.groupby('time.year').sum(dim='time', skipna=False)

    return extreme_event_per_jjas

def main():
    print("Loading rainfall dataset...")
    try:
        ds_rf = xr.open_dataset(rf_file)
        rf = ds_rf.rf
    except Exception as e:
        print(f"Error loading file: {e}")
        return

    # Data pre-processing: Replace zeros with small value if required by logic
    rf = rf.where(rf != 0, 0.1)

    if THRESHOLD_MODE == "percentile":
        print(f"Calculating {THRESHOLD_PERCENTILE}th percentile threshold from {year_for_threshold[0]}-{year_for_threshold[1]}...")
        historical_subset = rf.sel(time=(rf['time'].dt.month.isin(month_for_threshold)) & 
                                        (rf['time'].dt.year >= year_for_threshold[0]) & 
                                        (rf['time'].dt.year <= year_for_threshold[1]))
        
        threshold_data = historical_subset.quantile(THRESHOLD_PERCENTILE / 100, dim='time')
        title_threshold = f"{THRESHOLD_PERCENTILE}th percentile ({year_for_threshold[0]}-{year_for_threshold[1]})"
    else:
        print(f"Using absolute threshold of {THRESHOLD_VALUE} mm/day...")
        threshold_data = THRESHOLD_VALUE
        title_threshold = f"Absolute {THRESHOLD_VALUE} mm/day"

    print(f"Counting extreme events for period {year_for_analysis[0]}-{year_for_analysis[1]}...")
    extreme_event_per_jjas = calculate_extreme_events_per_jjas(
        rf, threshold_data, month_for_analysis, year_for_analysis
    )

    # Sum of extreme events over the entire analysis period
    sum_extreme_events = extreme_event_per_jjas.sum(dim='year', skipna=False)

    # Visualization
    print("Generating plot...")
    plt.figure(figsize=(10, 8))
    ax = plt.gca()

    # Plot the regional/India data
    sum_extreme_events.plot(ax=ax, cmap='YlOrRd', cbar_kwargs={'label': 'Number of Extreme Events'})

    # Try to overlay shapefile if available
    try:
        india_states = gpd.read_file(india_st_shape)
        india_states.boundary.plot(ax=ax, color='black', linewidth=0.8)
    except:
        print("Warning: Could not load shapefile for boundaries.")

    plt.title(f'Total Extreme Events JJAS {year_for_analysis[0]}-{year_for_analysis[1]}\n'
              f'(Threshold: {title_threshold})')
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    
    # Save or Show
    output_plot = 'rainfall_extreme_counts_plot.png'
    plt.savefig(output_plot)
    print(f"Plot saved to {output_plot}")
    plt.show()

if __name__ == "__main__":
    main()
