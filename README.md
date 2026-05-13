# Atmospheric River & Orographic Forcing Analysis (JJAS Monsoon)

This project provides a modularized Python pipeline for analyzing the relationship between Atmospheric Rivers (ARs), Integrated Vapor Transport (IVT), Orographic Forcing, and Flood events in the Indian region during the JJAS (June-September) monsoon season.

## Project Overview
The pipeline transitions from raw ERA5 flux data to identified AR events, analyzes the topographic influence (Orographic Forcing) of the Western Ghats, and finally links these atmospheric phenomena to catchment-level flood events using the INDOFLOODS dataset.

## Pipeline Structure
The scripts are numbered in the recommended order of execution:

### 1. [Rainfall_Extreme_Counts.py](pyscripts/Rainfall_Extreme_Counts.py)
Calculates historical rainfall thresholds (e.g., 99th percentile) and counts extreme rainfall events across the Indian landmass for the analysis period.

### 2. [ERA5_IVT_Calculation.py](pyscripts/ERA5_IVT_Calculation.py)
Integrates eastward and northward water vapor flux to calculate the Integrated Vapor Transport (IVT) magnitude. Outputs a refined NetCDF file used by subsequent scripts.

### 3. [AR_Identification.py](pyscripts/AR_Identification.py)
Identifies Atmospheric River events based on the Mahto et al. (2023) methodology. It performs cluster detection, spatial filtering, and land-masking to generate an AR Event Catalog and a binary mask.

### 4. [Orographic_Forcing.py](pyscripts/Orographic_Forcing.py)
Analyzes the interaction between moisture transport (IVT) and topography. It generates longitudinal cross-sections comparing ERA5 Geopotential Height with IVT intensity across the Western Ghats corridor.

### 5. [AR_Flood_Linkage.py](pyscripts/AR_Flood_Linkage.py)
Links detected ARs to specific gauge-level flood events. It implements a spatial intersection logic between AR shapes and catchment geometries, incorporating a 1-day hydrological lag.

---

## Data Requirements
To run the full pipeline, the following datasets should be placed in the execution directory (or updated in the script headers):

- **ERA5 Data:** `eastward_flux_1940_2025.nc`, `northward_flux_1940_2025.nc`, `ERA5_Geopotential.nc`
- **Rainfall Data (IMD):** `RF_p25_1901_2025.nc`
- **INDOFLOODS Data:** `floodevents_indofloods.csv`, `metadata_indofloods.csv`, and the `catchments_shapefiles_indofloods/` directory.
- **Shapefile:** `india_st.shp` (India State boundaries).

## Dependencies
The environment requires the following libraries:
- `xarray`, `numpy`, `pandas`, `matplotlib`
- `geopandas`, `shapely`, `regionmask`
- `scipy`, `scikit-image` (for cluster detection)

## Scientific Reference
The AR identification and linkage logic are based on:
> Mahto, S.S., et al. (2023). **"Atmospheric rivers that make landfall in India are associated with flooding."** *Communications Earth & Environment*. [DOI: 10.1038/s43247-023-00775-9](https://doi.org/10.1038/s43247-023-00775-9)

INDOFLOOD Dataset (Partial): https://zenodo.org/records/14584655
