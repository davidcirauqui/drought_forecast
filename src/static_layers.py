import numpy as np
import rioxarray
from rasterio.errors import RasterioIOError
from rioxarray.exceptions import NoDataInBounds
from rioxarray.merge import merge_arrays

from src.paths import RAW, _area_name
from src.features import WORLDCOVER_CLASSES


def _get_cog_mosaic(urls, coords, raw_path, save_data, redownload):
    """
    function that reads the window over coords from several Cloud-Optimised GeoTIFF tiles and merges them,
    or reads the merged map from raw_path if it was already saved
    outputs:
        - da: xarray DataArray with dimensions (lat, lon) at the native resolution of the tiles
    """
    if redownload or not raw_path.exists():
        parts = []
        for url in urls:
            try:
                da = rioxarray.open_rasterio(url)
                parts.append(da.rio.clip_box(*coords).load())
            # ocean-only tiles don't exist, and a tile can touch the box only on its edge
            except (RasterioIOError, NoDataInBounds):
                continue
        da = merge_arrays(parts)
        if save_data:
            raw_path.parent.mkdir(parents = True, exist_ok = True)
            da.rio.to_raster(raw_path, compress = "deflate")
    else:
        with rioxarray.open_rasterio(raw_path) as src:
            da = src.load()

    return da.squeeze("band", drop = True).rename({"y": "lat", "x": "lon"}).drop_vars("spatial_ref")


def _dem_tiles(coords):
    """
    function that gives the names of the 1x1 degree Copernicus DEM tiles covering a coordinate box,
    tiles are named after their south-west corner
    example: [2.97, 41.92, 3.19, 42.10] -> ['Copernicus_DSM_COG_10_N41_00_E002_00_DEM', ...] (4 tiles)
    """
    min_lon, min_lat, max_lon, max_lat = coords
    tiles = []
    for lat in range(int(np.floor(min_lat)), int(np.floor(max_lat)) + 1):
        for lon in range(int(np.floor(min_lon)), int(np.floor(max_lon)) + 1):
            ns = f"{'N' if lat >= 0 else 'S'}{abs(lat):02d}_00"
            ew = f"{'E' if lon >= 0 else 'W'}{abs(lon):03d}_00"
            tiles.append(f"Copernicus_DSM_COG_10_{ns}_{ew}_DEM")
    return tiles


def get_dem(coords, save_data = True, redownload = False):
    """Function that returns the Copernicus DEM (GLO-30) at its native ~30 m resolution
    inputs:
        - coords: Coordinate box [min_lon, min_lat, max_lon, max_lat] in WGS84
        - save_data: boolean determining whether to save the DEM on disk
        - redownload: boolean determining whether to download data again (gets data saved on disk if not)
    outputs:
        - ds: xarray Dataset with dimensions (lat, lon) and variable elevation in metres above sea level
    """

    # public Cloud-Optimised GeoTIFFs of the 30 m Copernicus DEM on AWS, only the window over coords is read
    # (on CDSE Sentinel Hub the 30 m DEM is restricted to registered CCM users)
    urls = [f"https://copernicus-dem-30m.s3.amazonaws.com/{tile}/{tile}.tif" for tile in _dem_tiles(coords)]
    raw_path = RAW / "dem" / _area_name(coords) / "copernicus_dem_30.tif"

    elevation = _get_cog_mosaic(urls, coords, raw_path, save_data, redownload)
    return elevation.astype(np.float32).to_dataset(name = "elevation")


def _worldcover_tiles(coords):
    """
    function that gives the names of the 3x3 degree WorldCover tiles covering a coordinate box,
    tiles are named after their south-west corner
    example: [2.97, 41.92, 3.19, 42.10] -> ['N39E000', 'N39E003', 'N42E000', 'N42E003']
    """
    min_lon, min_lat, max_lon, max_lat = coords
    tiles = []
    for lat in range(int(np.floor(min_lat / 3)) * 3, int(np.floor(max_lat / 3)) * 3 + 1, 3):
        for lon in range(int(np.floor(min_lon / 3)) * 3, int(np.floor(max_lon / 3)) * 3 + 1, 3):
            tiles.append(f"{'N' if lat >= 0 else 'S'}{abs(lat):02d}{'E' if lon >= 0 else 'W'}{abs(lon):03d}")
    return tiles


def get_worldcover(coords, year = 2021, save_data = True, redownload = False):
    """Function that returns the ESA WorldCover land cover map at its native 10 m resolution
    inputs:
        - coords: Coordinate box [min_lon, min_lat, max_lon, max_lat] in WGS84
        - year: 2020 (algorithm v100) or 2021 (algorithm v200)
        - save_data: boolean determining whether to save the map on disk
        - redownload: boolean determining whether to download data again (gets data saved on disk if not)
    outputs:
        - ds: xarray Dataset with dimensions (lat, lon) and one hot encoded variables for land cover types
    """

    # public Cloud-Optimised GeoTIFFs from ESA, only the window over coords is read
    version = {2020: "v100", 2021: "v200"}[year]
    urls = [f"https://esa-worldcover.s3.eu-central-1.amazonaws.com/{version}/{year}/map/"
            f"ESA_WorldCover_10m_{year}_{version}_{tile}_Map.tif" for tile in _worldcover_tiles(coords)]
    raw_path = RAW / "worldcover" / _area_name(coords) / f"worldcover_{year}.tif"

    landcover = _get_cog_mosaic(urls, coords, raw_path, save_data, redownload)
    ds = landcover.astype(np.uint8).to_dataset(name = "landcover")
    ds['landcover'].attrs = {'flag_values': list(WORLDCOVER_CLASSES),
                             'flag_meanings': " ".join(WORLDCOVER_CLASSES.values())}
    return ds
