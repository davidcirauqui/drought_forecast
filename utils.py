import numpy as np
import pandas as pd
import xarray as xr
from pathlib import Path
import shutil
import zipfile
import cdsapi
import rioxarray
from rasterio.enums import Resampling
from rasterio.errors import RasterioIOError
from rasterio.transform import from_bounds
from rioxarray.exceptions import NoDataInBounds
from rioxarray.merge import merge_arrays


from sentinelhub import (
    BBox,
    bbox_to_dimensions,
    CRS,
    SentinelHubCatalog,
    SentinelHubRequest,
    DataCollection,
    MimeType,
)


def _fmt_coord(value, pos, neg, ndigits):
    """
    function that converts coordinates with negatives values to cardinal directions
    example: 2.97 -> '2.97E', -0.5 -> '0.50W'
    """
    return f"{abs(value):.{ndigits}f}{pos if value >= 0 else neg}"

def make_data_folder(bbox_coords, time_interval, root="data/raw/s2", ndigits=2):
    """
    function that gives a data_folder name based on coordinates and dates
    Inputs:
        - bbox_coords:   [min_lon, min_lat, max_lon, max_lat] in WGS84
        - time_interval: (start, end) as 'YYYY-MM-DD' strings or datetimes
    Outputs:
        - folder name: example data/raw/s2/2.97E-3.19E_41.92N-42.10N/2023-07
    """

    min_lon, min_lat, max_lon, max_lat = bbox_coords
    area = (f"{_fmt_coord(min_lon, 'E', 'W', ndigits)}-{_fmt_coord(max_lon, 'E', 'W', ndigits)}_"
            f"{_fmt_coord(min_lat, 'N', 'S', ndigits)}-{_fmt_coord(max_lat, 'N', 'S', ndigits)}")

    start, end = (pd.Timestamp(t).normalize() for t in time_interval)
    # If period spans exactly one calendar month -> '2023-07', 
    # otherwise '2023-07-01_2023-07-15'
    if start.day == 1 and end == start + pd.offsets.MonthEnd(0):
        period = start.strftime("%Y-%m")
    else:
        period = f"{start:%Y-%m-%d}_{end:%Y-%m-%d}"

    return str(Path(root) / area / period)


def _month_interval(dates):
    """
    function that converts a [year, month] pair into the first and last day of that month
    example: [2023, 7] -> (Timestamp('2023-07-01'), Timestamp('2023-07-31'))
    """
    if len(dates) != 2:
        raise ValueError(f"dates must be [year, month], got {dates}")
    year, month = (int(d) for d in dates)
    if not 1 <= month <= 12:
        raise ValueError(f"month must be between 1 and 12, got {month}")

    start = pd.Timestamp(year = year, month = month, day = 1)
    end = start + pd.offsets.MonthEnd(0)
    return start, end


def _month_list(dates):
    """
    function that converts a list of [year, month] pairs into sorted, unique (start, end) intervals
    example: [[2023, 8], [2023, 7]] -> [(2023-07-01, 2023-07-31), (2023-08-01, 2023-08-31)]
    """
    if len(dates) == 0 or not isinstance(dates[0], (list, tuple)):
        raise ValueError(f"dates must be a list of [year, month] pairs, e.g. [[2023, 7]], got {dates}")
    return sorted({_month_interval(d) for d in dates})


def _ndmi_dataset(img, start, coords):
    """
    function that converts one monthly NDMI image (height, width, 2) into a dataset
    with dimensions (time, lat, lon) and variables NDMI median and clear observation count
    """
    height, width = img.shape[:2]

    # pixel-center coordinates: the image is on a WGS84 grid,
    # rows go from north (max_lat) to south, columns from west to east
    min_lon, min_lat, max_lon, max_lat = coords
    lons = min_lon + (np.arange(width) + 0.5) * (max_lon - min_lon) / width
    lats = max_lat - (np.arange(height) + 0.5) * (max_lat - min_lat) / height

    img = img.astype(np.float32)[np.newaxis]
    return xr.Dataset(
        {
            'ndmi_median': (('time', 'lat', 'lon'), img[..., 0]),
            'clear_count': (('time', 'lat', 'lon'), img[..., 1]),
        },
        coords = {'time': [start], 'lat': lats, 'lon': lons},
    )



def get_ndmi(coords, resolution, dates, config, save_data = True, redownload = False):
    """Function that returns the monthly median NDMI index, after filtering out
      clouds, shadows, water and snow
    inputs:
        - coords: Coordinate box delimiting the square over which to compute the NDMI index
        - resolution: Resolution of the requested data
        - dates: list of [[year, month]] dates for which to request data, one request per month
        - config: SentinelHub configuration
        - save_data: boolean determining whether to save requested data on disk
        - redownload: boolean determining whether to download data again (gets data saved on disk if not)
    outputs:
        - ds: xarray Dataset with dimensions (time, lat, lon) on pixel centers, with monthly NDMI median
        values and the total number of clear (non-masked) observations used to compute each of them
    """


    months = _month_list(dates)

    # SentinelHub specifications
    m_bbox = BBox(bbox = coords, crs=CRS.WGS84)
    m_size = bbox_to_dimensions(m_bbox, resolution = resolution)
    catalog = SentinelHubCatalog(config = config)

    # select NIR (B8A) and SWIR (B11) channels from sentinel data
    evalscript_ndmi = """
        //VERSION=3
        function setup() {
            return {
                input: [{bands: ["B8A","B11", "SCL", "dataMask"],units: "DN"}],
                output: {bands: 2, sampleType: "FLOAT32"},
                mosaicking: "ORBIT"
            };
        }

        // drop clouds and shadows using Scene Classification Band (SCL)
        // SCL classes kept: 4 = vegetation, 5 = bare soil, 7 = unclass
        // classes dropped: clouds, shadows, water, snow
        function isClear(s) { return s.dataMask == 1 && [4, 5, 7].includes(s.SCL);}

        // return median
        function evaluatePixel(sample) {
          const v = sample.filter(isClear).map(s => (s.B8A - s.B11) / (s.B8A + s.B11));
          if (v.length == 0) return [NaN, 0];
          v.sort((a, b) => a - b);
          const m = Math.floor(v.length / 2);
          const median = v.length % 2 ? v[m] : (v[m - 1] + v[m]) / 2;
          return [median, v.length];
        }
    """

    # define requesting function, one request per month
    def request_ndmi(start, end):
        return SentinelHubRequest(
            evalscript = evalscript_ndmi,
            input_data=[
                SentinelHubRequest.input_data(
                    data_collection=DataCollection.SENTINEL2_L2A.define_from(
                        "s2l2a", service_url=config.sh_base_url
                    ),
                    time_interval=(f"{start:%Y-%m-%d}", f"{end:%Y-%m-%d}"),
                )
            ],
            responses=[SentinelHubRequest.output_response("default", MimeType.TIFF)],
            bbox = m_bbox,
            size = m_size,
            config = config,
            data_folder = make_data_folder(coords, (start, end))
        )

    # request data from SentinelHub month by month
    # (already saved months are read from disk unless redownload is True)
    parts = []
    for start, end in months:
        data = request_ndmi(start, end).get_data(save_data = save_data, redownload = redownload)
        parts.append(_ndmi_dataset(data[0], start, coords))

    return xr.concat(parts, dim = 'time')





ERA5_SHORT_NAMES = {
    "total_precipitation": "tp",
    "2m_temperature": "t2m",
    "potential_evaporation": "pev",
    "total_evaporation": "e",
    "volumetric_soil_water_layer_1": "swvl1",
    "volumetric_soil_water_layer_2": "swvl2",
}


def get_era5(coords, dates, variables, margin=0.1, save_data = True, redownload = False):
    """Function that returns ERA5-Land monthly means of requested variables
    inputs:
        - coords: Coordinate box delimiting the square over which to download the data [min_lon, min_lat, max_lon, max_lat] in WGS84
        - dates: list of [[year, month]] dates for which to request data
        - variables: list of ERA5-Land variable names to download (keys of ERA5_SHORT_NAMES)
        - margin: degrees added around the box, since the ERA5-Land grid is ~9 km
        - save_data: boolean determining whether to keep the downloaded file on disk
        - redownload: boolean determining whether to download data again (gets data saved on disk if not)
    outputs:
        - ds: xarray Dataset with dimensions (time, lat, lon) and the requested monthly variables
    """

    unknown = [v for v in variables if v not in ERA5_SHORT_NAMES]
    if unknown:
        raise ValueError(f"unknown ERA5-Land variables {unknown}, add their short names to ERA5_SHORT_NAMES")
    short_names = [ERA5_SHORT_NAMES[v] for v in variables]

    months = _month_list(dates)
    month_paths = {start: Path(make_data_folder(coords, (start, end), root="data/raw/era5")) / "era5_land_monthly.nc"
                   for start, end in months}

    # read saved months; a month is complete only if its file has all the requested variables
    parts, saved = {}, {}
    for start, _ in months:
        if month_paths[start].exists():
            with xr.open_dataset(month_paths[start], engine="scipy") as month_ds:
                saved[start] = month_ds.load()
            if not redownload and all(v in saved[start] for v in short_names):
                parts[start] = saved[start][short_names]
    missing = [(start, end) for start, end in months if start not in parts]

    # download all missing months in a single CDS request, then split them into one file per month
    if missing:
        # CDS area is [N, W, S, E]
        min_lon, min_lat, max_lon, max_lat = coords
        area = [max_lat + margin, min_lon - margin, min_lat - margin, max_lon + margin]

        # years x months is a cartesian product, extra months are dropped below
        request = {
            "product_type": ["monthly_averaged_reanalysis"],
            "variable": list(variables),
            "year": sorted({f"{start.year}" for start, _ in missing}),
            "month": sorted({f"{start.month:02d}" for start, _ in missing}),
            "time": ["00:00"],
            "area": area,
            "data_format": "netcdf",
            "download_format": "unarchived",
        }
        tmp_dir = month_paths[missing[0][0]].parent.parent / "_cds_download"
        tmp_dir.mkdir(parents=True, exist_ok=True)
        tmp_path = tmp_dir / "era5_land_monthly.nc"
        cdsapi.Client().retrieve("reanalysis-era5-land-monthly-means", request, str(tmp_path))
        ds_new = _open_era5(tmp_path)
        shutil.rmtree(tmp_dir, ignore_errors=True)

        not_available = []
        for start, end in missing:
            month_ds = ds_new.sel(time=slice(start, end))[short_names]
            if month_ds.time.size == 0:
                not_available.append(f"{start:%Y-%m}")
                continue
            # saved as NETCDF3: much faster to open than NETCDF4/HDF5 for many small files,
            # keeping the other variables already saved for that month
            if save_data:
                to_save = month_ds
                if start in saved:
                    other = saved[start].drop_vars(short_names, errors="ignore")
                    to_save = xr.merge([other, month_ds], join="override", compat="override")
                month_paths[start].parent.mkdir(parents=True, exist_ok=True)
                to_save.to_netcdf(month_paths[start], format="NETCDF3_64BIT", engine="scipy")
            parts[start] = month_ds
        if not_available:
            raise ValueError(f"CDS returned no data for months: {not_available}")

    ds = xr.concat([parts[start] for start, _ in months], dim="time")
    ds = ds.rename({"latitude": "lat", "longitude": "lon"})

    return ds


def _open_era5(path):
    """
    function that loads an ERA5-Land netcdf file (or the zip CDS sometimes returns instead)
    into memory and normalises its coordinates
    """
    # CDS sometimes returns a zip with one .nc per variable group
    if zipfile.is_zipfile(path):
        extract_dir = path.parent / "unzipped"
        with zipfile.ZipFile(path) as z:
            z.extractall(extract_dir)
        files = sorted(extract_dir.glob("*.nc"))
    else:
        files = [path]

    # load into memory so files are released (needed on Windows before deleting)
    parts = []
    for f in files:
        with xr.open_dataset(f) as part:
            parts.append(part.load())
    ds = xr.merge(parts, compat="override")

    # new CDS names the time dimension 'valid_time'
    if "valid_time" in ds.dims:
        ds = ds.rename({"valid_time": "time"})
    return ds.drop_vars(["number", "expver"], errors="ignore")





def _area_name(bbox_coords, ndigits=2):
    """
    function that gives a folder name for a coordinate box
    example: [2.97, 41.92, 3.19, 42.10] -> '2.97E-3.19E_41.92N-42.10N'
    """
    min_lon, min_lat, max_lon, max_lat = bbox_coords
    return (f"{_fmt_coord(min_lon, 'E', 'W', ndigits)}-{_fmt_coord(max_lon, 'E', 'W', ndigits)}_"
            f"{_fmt_coord(min_lat, 'N', 'S', ndigits)}-{_fmt_coord(max_lat, 'N', 'S', ndigits)}")


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
    raw_path = Path("data/raw/dem") / _area_name(coords) / "copernicus_dem_30.tif"

    elevation = _get_cog_mosaic(urls, coords, raw_path, save_data, redownload)
    return elevation.astype(np.float32).to_dataset(name = "elevation")



WORLDCOVER_CLASSES = {
    10: "tree_cover",
    20: "shrubland",
    30: "grassland",
    40: "cropland",
    50: "built_up",
    60: "bare_sparse",
    70: "snow_ice",
    80: "water",
    90: "herbaceous_wetland",
    95: "mangroves",
    100: "moss_lichen",
}


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
    raw_path = Path("data/raw/worldcover") / _area_name(coords) / f"worldcover_{year}.tif"

    landcover = _get_cog_mosaic(urls, coords, raw_path, save_data, redownload)
    ds = landcover.astype(np.uint8).to_dataset(name = "landcover")
    ds['landcover'].attrs = {'flag_values': list(WORLDCOVER_CLASSES),
                             'flag_meanings': " ".join(WORLDCOVER_CLASSES.values())}
    return ds



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
        - ds: xarray Dataset with dimensions (..., lat, lon) in WGS84, e.g. ERA5, DEM or WorldCover
        - target: xarray Dataset defining the output grid through its lat and lon coordinates (pixel centres),
        e.g. the NDMI dataset
        - method: rasterio resampling method
            - "bilinear" for continuous variables (weather, elevation)
            - "mode" for categories (land cover): most common class under each target pixel
            - "average" for continuous variables going from fine to coarse
    outputs:
        - ds: xarray Dataset with the same variables on the target lat and lon
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
    for v in out.data_vars:
        out[v].attrs.pop("_FillValue", None)
    return out



def terrain_features(elevation, flat_slope = 1):
    """Function that computes terrain variables from an elevation map in WGS84
    (compute them after regridding, never interpolate aspect: averaging 359 and 1 degrees gives 180)
    inputs:
        - elevation: xarray DataArray with dimensions (lat, lon) in metres, lat from north to south
        - flat_slope: slope in degrees below which ground is considered flat (aspect set to NaN)
    outputs:
        - ds: xarray Dataset with dimensions (lat, lon) and variables
            - slope in degrees
            - aspect: direction the slope faces, in degrees clockwise from north (NaN on flat ground)
            - northness: cos(aspect) * sin(slope), +1 steep north-facing, -1 steep south-facing, ~0 flat
            - eastness: sin(aspect) * sin(slope), +1 steep east-facing, -1 steep west-facing, ~0 flat
    """
    lats = elevation.lat.values
    lons = elevation.lon.values

    # pixel size in metres (1 degree of latitude ~ 111.32 km, longitude shrinks with cos(lat))
    dy = abs(lats[1] - lats[0]) * 111_320
    dx = abs(lons[1] - lons[0]) * 111_320 * np.cos(np.deg2rad(lats))[:, np.newaxis]

    # elevation change per metre towards north and east (rows go from north to south)
    dz_row, dz_col = np.gradient(elevation.values)
    dz_north = -dz_row / dy
    dz_east = dz_col / dx

    # slope and aspect (downhill direction, clockwise from north) in radians
    slope = np.arctan(np.hypot(dz_east, dz_north))
    aspect = np.arctan2(-dz_east, -dz_north)

    # aspect weighted by slope, model inputs: fade smoothly to 0 on flat ground, no threshold needed
    northness = np.cos(aspect) * np.sin(slope)
    eastness = np.sin(aspect) * np.sin(slope)

    # degrees for the output, aspect NaN on flat ground where it has no meaning
    slope_deg = np.degrees(slope)
    aspect_deg = np.where(slope_deg < flat_slope, np.nan, (np.degrees(aspect) + 360) % 360)

    return xr.Dataset(
        {
            'slope': (('lat', 'lon'), slope_deg.astype(np.float32)),
            'aspect': (('lat', 'lon'), aspect_deg.astype(np.float32)),
            'northness': (('lat', 'lon'), northness.astype(np.float32)),
            'eastness': (('lat', 'lon'), eastness.astype(np.float32)),
        },
        coords = {'lat': lats, 'lon': lons},
    )


def one_hot_landcover(landcover):
    """Function that one-hot encodes a WorldCover class map
    (compute it after regridding with method "mode", never interpolate the one-hot variables)
    inputs:
        - landcover: xarray DataArray with dimensions (lat, lon) and WORLDCOVER_CLASSES codes
    outputs:
        - ds: xarray Dataset with one 0/1 variable per class (lc_tree_cover, lc_cropland, ...)
    """
    return xr.Dataset({f'lc_{name}': (landcover == code).astype(np.uint8)
                       for code, name in WORLDCOVER_CLASSES.items()})



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
