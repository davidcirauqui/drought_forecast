# Study of the 2022-2024 drought episode in Catalonia

🚧 Project under construction

I monitor and map the drought episode that affected parts of the Mediterranean during the 2022-2024 summer seasons, by using satellite and weather data. The main final goal is to train a model that is able to predict similar episodes in the future. 

To this end, I will focus on the following data:
- **Study area:** I focus on the Baix Empordà area, which has many types of vegetation
- **Period:** 2017–2025, with monthly measures. NDMI/NDVI anomalies are computed against 2017–2021 (due to a lack of more data), and the climatic reference for SPI is computed from ERA5-Land 1991–2020 data
- **Sources:** Sentinel-2 L2A, ERA5-Land monthly means (CDS), Copernicus DEM GLO-30, ESA WorldCover 2021 


## First findings
- Usually-irrigated fields did not stay wet during the drought, possibly due to watering restrictions.
- NDMI anomalies lead NDVI anomalies (vegetation loses water before it loses greenness)



## Repository structure

    src/         shared code and utility functions
    notebooks/   run in order


## Setup

You will first need to create a [Copernicus Data Space Ecosystem](https://dataspace.copernicus.eu/) account and register an OAuth client to get a `client_id` / `client_secret` pair, in order to access Sentinel-2 data through SentinelHub. The first time you run `01_build_cube.ipynb`, the configuration cell will prompt for your `client_id` and `client_secret` (via `getpass`, so they are not stored in the notebook), and save the SentinelHub profile locally.

ERA5-Land data is downloaded from the [Climate Data Store](https://cds.climate.copernicus.eu/): create an account, accept the ERA5-Land licence, and save your API key in `~/.cdsapirc` as described in the [CDS API instructions](https://cds.climate.copernicus.eu/how-to-api). The Copernicus DEM and ESA WorldCover are public Cloud-Optimised GeoTIFFs and need no account.

Once downloaded, all data is cached in `data/raw/` and read from disk on later runs.

The project requires Python 3.12 or later (developed with 3.14) and the following packages:
- numpy
- pandas
- scipy
- xarray
- dask
- zarr
- rioxarray
- rasterio
- matplotlib
- sentinelhub
- cdsapi

Then install the shared code in `src/` as a package, together with its dependencies, so that the notebooks can import it:

    pip install -e .


