import numpy as np
import pandas as pd
import xarray as xr
from pathlib import Path
import shutil
import zipfile
import cdsapi

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
        - redownload: boolean determining whether to download data again (takes data from saved if not)
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





def get_era5(coords, dates, variables, margin=0.1, save_data = True, redownload = False):
    """Function that returns ERA5-Land monthly means of requested variables
    inputs:
        - coords: Coordinate box delimiting the square over which to download the data [min_lon, min_lat, max_lon, max_lat] in WGS84
        - dates: list of [[year, month]] dates for which to request data
        - variables: list of ERA5-Land variable names to download
        - margin: degrees added around the box, since the ERA5-Land grid is ~9 km
        - save_data: boolean determining whether to keep the downloaded file on disk
        - redownload: boolean determining whether to download data again (takes data from saved if not)
    outputs:
        - ds: xarray Dataset with dimensions (time, lat, lon) and the requested monthly variables
    """

    months = _month_list(dates)
    month_paths = {start: Path(make_data_folder(coords, (start, end), root="data/raw/era5")) / "era5_land_monthly.nc"
                   for start, end in months}
    missing = [(start, end) for start, end in months if redownload or not month_paths[start].exists()]

    # download all missing months in a single CDS request, then split them into one file per month
    parts = {}
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
            month_ds = ds_new.sel(time=slice(start, end))
            if month_ds.time.size == 0:
                not_available.append(f"{start:%Y-%m}")
                continue
            if save_data:
                month_paths[start].parent.mkdir(parents=True, exist_ok=True)
                month_ds.to_netcdf(month_paths[start])
            parts[start] = month_ds
        if not_available:
            raise ValueError(f"CDS returned no data for months: {not_available}")

    # read already saved months from disk
    for start, _ in months:
        if start not in parts:
            parts[start] = _open_era5(month_paths[start])

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