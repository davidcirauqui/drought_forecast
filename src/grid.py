import xarray as xr
import rioxarray
from rasterio.enums import Resampling
from rasterio.transform import from_bounds

from src.features import terrain_features, one_hot_landcover


def fill_sea(ds):
    """Function that fills NaN cells (e.g. sea cells in ERA5-Land, which only covers land)
    with the values of their neighbouring cells, first along longitude then along latitude
    inputs:
        - ds: xarray Dataset with dimensions (..., lat, lon)
    outputs:
        - ds: same Dataset without NaN cells (as long as each row or column has at least one valid cell)
    """
    return ds.ffill("lon").bfill("lon").ffill("lat").bfill("lat")


def regrid(ds, target, method):
    """Function that resamples a dataset onto the lat/lon grid of a target dataset
    inputs:
        - ds: xarray Dataset or DataArray with dimensions (..., lat, lon) in WGS84, e.g. ERA5, DEM or WorldCover
        - target: xarray Dataset defining the output grid through its lat and lon coordinates (pixel centres),
        e.g. the NDMI dataset
        - method: rasterio resampling method
            - "bilinear" for continuous variables (weather, elevation)
            - "mode" for categories (land cover): most common class under each target pixel
            - "average" for continuous variables going from fine to coarse
    outputs:
        - ds: same type as the input (Dataset or DataArray) on the target lat and lon
    """

    # target grid: pixel edges from pixel centres (lat goes from north to south)
    lat, lon = target.lat.values, target.lon.values
    dlat, dlon = abs(lat[1] - lat[0]), abs(lon[1] - lon[0])
    transform = from_bounds(lon.min() - dlon / 2, lat.min() - dlat / 2,
                            lon.max() + dlon / 2, lat.max() + dlat / 2, len(lon), len(lat))

    out = (ds.rio.set_spatial_dims(x_dim = "lon", y_dim = "lat")
             .rio.write_crs("EPSG:4326")
             .rio.reproject("EPSG:4326", shape = (len(lat), len(lon)), transform = transform,
                            resampling = Resampling[method]))

    # use the exact target coordinates so that xr.merge aligns without rounding differences
    out = out.rename({"y": "lat", "x": "lon"}) if "x" in out.dims else out
    out = out.assign_coords(lat = lat, lon = lon).drop_vars("spatial_ref")

    # drop the nodata attribute added by rioxarray, it blocks saving the cube to netcdf/zarr
    for v in (out.data_vars.values() if isinstance(out, xr.Dataset) else [out]):
        v.attrs.pop("_FillValue", None)
    return out


def era5_to_grid(era5, target):
    """Function that brings ERA5-Land data onto the target grid
    (sea cells filled from neighbouring land cells first, then bilinear interpolation)
    inputs:
        - era5: xarray Dataset from get_era5, dimensions (time, lat, lon) at ~9 km
        - target: xarray Dataset defining the output grid, e.g. the NDMI dataset
    outputs:
        - ds: xarray Dataset with the same variables on the target lat and lon
    """
    return regrid(fill_sea(era5), target, "bilinear")


def dem_to_grid(dem, target, flat_slope = 1):
    """Function that brings the DEM onto the target grid and adds the terrain variables
    (elevation interpolated bilinearly first, terrain variables computed after on the target grid)
    inputs:
        - dem: xarray Dataset from get_dem, dimensions (lat, lon) at ~30 m
        - target: xarray Dataset defining the output grid, e.g. the NDMI dataset
        - flat_slope: slope in degrees below which ground is considered flat (aspect set to NaN)
    outputs:
        - ds: xarray Dataset on the target lat and lon with elevation, slope, aspect, northness and eastness
    """
    dem_grid = regrid(dem, target, "bilinear")
    return xr.merge([dem_grid, terrain_features(dem_grid.elevation, flat_slope)])


def worldcover_to_grid(wc, target):
    """Function that brings the WorldCover map onto the target grid and adds the one-hot encoding
    (most common class under each target pixel first, one-hot encoding computed after)
    inputs:
        - wc: xarray Dataset from get_worldcover, dimensions (lat, lon) at 10 m
        - target: xarray Dataset defining the output grid, e.g. the NDMI dataset
    outputs:
        - ds: xarray Dataset on the target lat and lon with the landcover class codes
        and one 0/1 variable per class (lc_tree_cover, lc_cropland, ...)
    """
    wc_grid = regrid(wc, target, "mode")
    return xr.merge([wc_grid, one_hot_landcover(wc_grid.landcover)])
