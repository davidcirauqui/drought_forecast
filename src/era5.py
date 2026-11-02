import shutil
import zipfile
from pathlib import Path

import cdsapi
import xarray as xr

from src.paths import RAW, make_data_folder, _month_list


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
    month_paths = {start: Path(make_data_folder(coords, (start, end), root=RAW / "era5")) / "era5_land_monthly.nc"
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
