import pandas as pd
import numpy as np
import xarray as xr


SPLITS = {"train": (2017, 2021),
          "val"  : (2022, 2022),
          "test" : (2023, 2025)}

SCALED_VARS = ['ndmi_anomaly_filled', 'tp', 't2m', 'pev', 'e', 'swvl1', 
               'swvl2', 'spi3', 'spi6', 'spi12', 'elevation', 'slope']

DYNAMIC_VARS = ['ndmi_anomaly_filled', 'tp', 't2m', 'pev', 'e',
                'swvl1', 'swvl2', 'spi3', 'spi6', 'spi12']

STATIC_VARS = ['elevation', 'slope', 'northness', 'eastness', 'irrigated']

LEAD_VARS = ['tp', 't2m', 'pev', 'e']


def split_of_year(year, splits = SPLITS):
    for split, (yyyy_0, yyyy_f) in splits.items():
        if yyyy_0 <= year <= yyyy_f:
            return split

def make_samples(times, T = 12, horizon = 3, splits = SPLITS):
    """Return the reference months t0 that form valid samples, and the split of each.

    A sample uses inputs from months t0-T+1 ... t0 and predicts targets at
    t0+1 ... t0+horizon. Its split is decided by its target months only:
    a model is scored on targets, so no target month may appear in more
    than one split.

    Parameters
    ----------
    times : pd.DatetimeIndex
        Monthly time axis of the cube, sorted and without gaps.
    T : int
        Length of the input window, in months.
    horizon : int
        Number of months ahead to predict.
    splits : dict[str, tuple[int, int]]
        Split name -> (first year, last year), both inclusive.

    Returns
    -------
    pd.DataFrame
        One row per valid sample, with columns:
        - t0_idx : int, position of t0 in `times`
        - t0     : Timestamp, the reference month
        - split  : str, one of the keys of `splits`

    Notes
    -----
    - Samples whose targets fall in more than one split, or in no split,
      are dropped.
    - The input window may reach into an earlier split. This is not
      leakage: a real forecast issued at t0 has that past data.
    """

    times = pd.DatetimeIndex(times)

    samples = []
    for i in range(T - 1, len(times) - horizon):
        
        # add the split of each target month
        splits_set = set()
        for h in range(1, horizon + 1):
            split_h = split_of_year(times[i + h].year, splits = splits)
            splits_set.add(split_h)

        # if all targets are in the same split, add 
        if (len(splits_set) == 1) and (None not in splits_set):
            samples.append((i, times[i], splits_set.pop()))

    return pd.DataFrame(samples, columns = ["t0_idx", "t0", "split"])


def land_mask(cube):
    """ Creates a mask that defines which pixels can be used (for training, statistics, tiles, etc),
    based on wether they are valid climate-related signals or not
    
    Parameters
    ----------
    cube : xr.Dataset 
        cube containing landcover and valid_mask

    Returns
    -------
    xr.DataArray
        Boolean for each (lat, lon) pair, True if the pixel is used  

    Notes
    -----
    Pixels are masked out if one of the following conditions apply:
        - they represent bodies of water
        - they represent built up structures (roads, buildings...) 
        - they have never been observed
    Water and built up areas are masked out because NDMI there does not reflect any climate-driven variable
    """

    not_water = cube.landcover != 80
    not_built = cube.landcover != 50 
    observed = cube.valid_mask.any("time")

    return not_water & not_built & observed



        

def holdout_mask(cube, frac = .2, tile = 64):
    """ Creates a mask that defines which pixels are held out. Held out pixels are never used 
    for either training, validation or normalisation, so that they are kept apart to test 
    the model on completely new places.
    
    Parameters
    ----------
    cube : xr.Dataset 
        cube containing (lat, lon) coordinates, with lon increasing west to east
    frac : float
        defines the minimum fraction of columns that are masked out
    tile : int
        Tile size in pixels.

    Returns
    -------
    xr.DataArray
        Boolean for each (lat, lon) pair, True if the pixel is held out  

    Notes
    -----
    - Pixels are held out if they belong to the easternmost strip of the map.
    - For some models (like ConvLSTM), the area is cut into square tiles of 
    64x64 pixels, each tile-month being one training example. The held-out 
    boundary is accordingly rounded down to the edges of such tiles. 
    """

    lon = cube.lon
    if not (np.diff(lon.values) > 0).all():
        raise ValueError("lon must increase west to east")

    n_lon = cube.sizes['lon']
    cut = (int(n_lon * (1 - frac)) // tile) * tile 

    mask = np.arange(n_lon) >= cut
    mask = xr.DataArray(mask, dims = 'lon', coords = {'lon': lon})
    mask = mask.broadcast_like(cube.lat).transpose('lat', 'lon')

    return mask





def normalisation_stats(cube, train_mask, variables = SCALED_VARS, years = SPLITS["train"]):

    """ Returns mean and std for all requested variables, computed over training
    splits and pixels only. Since test years span over the drought period, including 
    them would shift the mean towards dry values.

    Parameters
    ----------
    cube : xr.Dataset
        cube containing variables for which to compute the statistics
    train_mask: xr.DataArray (bool, lat x lon)
        lat-lon mask determining which pixels are used for training (True if are used)
    variables: [str]
        list of variables for which statistics are computed
    years: (int, int)
        Start and end year of the training period. Default SPLITS["train"], both years inclusive

    Returns
    -------
    stats : dict
        Dictionary with one entry per variable. Each entry is a dictionary with 'mean' 
        and 'std' values: {"tp": {"mean": float, "std": float}, ...}


    Notes
    -----
    - NaN values are skipped
    - variables without a time dim are selected only by pixel 
    - raises ValueError if any statistic is not finite or std is 0
    
    """
    
    y0, yf = years
    stats = {}
    for variable in variables:

        # select data
        data = cube[variable]
        
        # if variable has time dimension, apply time mask
        if 'time' in data.dims:
            data = data.sel(time = slice(str(y0), str(yf)))

        # apply training-pixel mask
        data = data.where(train_mask)

        # compute mean and std over all dimensions
        mean = float(data.mean())
        std = float(data.std())

        if not (np.isfinite(mean) and np.isfinite(std) and std > 0):
            raise ValueError(
                f"{variable}: invalid statistics (mean={mean}, std={std}) "
                f"over training years {y0}-{yf} and training pixels"
            )

        # store in dict
        stats[variable] = {"mean" : mean, "std" : std}

    return stats



def sample_pixels(cube, land, holdout, frac = .01, min_per_class = 500, seed = 0):
    """Picks the pixels that are fed to the LightGBM table, to avoid using a table 
    with every pixel for every sample month t0, which would be too large. Stratifying
    by class keeps those rare ones (grasslands, wetland) from being swamped by those
    more common (cropland, forest).

    Parameters
    ----------
    cube : xr.Dataset
        Cube with `landcover` (WorldCover codes) and (lat, lon) coordinates.
    land : xr.DataArray (bool, lat x lon)
        Usable pixels, from `land_mask`.
    holdout : xr.DataArray (bool, lat x lon)
        Held-out pixels, from `holdout_mask`.
    frac : float
        Fraction of each class's pixels to draw.
    min_per_class : int
        Minimum pixels per class, capped at the number available.
    seed : int
        Seed of the random generator, for reproducibility.

    Returns
    -------
    pd.DataFrame
        One row per sampled pixel, with columns:
        - y_idx, x_idx : int, positions along lat and lon
        - lat, lon     : float, coordinates
        - lc_class     : int, WorldCover code
        - holdout      : bool, True if the pixel is in the held-out strip
        - weight       : float, pixels of its class in its region / pixels sampled.
                         Each sampled pixel stands for `weight` real pixels, so
                         weighted metrics are area-representative.

    Notes
    -----
    - Held-out pixels are sampled too (test needs them) but flagged, never dropped.
    - The rule is applied separately inside and outside the holdout, so each
      region gets every class at its own minimum.
    - The same pixels are used in all time splits. This isn't leakage, since splits
      are separated by target month and generalisation to new places is measured
      by the holdout and by spatial block Cross Validation.
    """

    rng = np.random.default_rng(seed)

    lc = cube.landcover.transpose("lat", "lon").values
    land_v = land.transpose("lat", "lon").values
    hold_v = holdout.transpose("lat", "lon").values

    parts = []
    for is_holdout in (False, True):
        region = land_v & (hold_v == is_holdout)

        for cls in np.unique(lc[region]):
            y, x = np.nonzero(region & (lc == cls))
            n = min(len(y), max(int(round(frac * len(y))), min_per_class))
            pick = rng.choice(len(y), size = n, replace = False)

            parts.append(pd.DataFrame({"y_idx": y[pick], "x_idx": x[pick],
                                       "lc_class": int(cls), "holdout": is_holdout,
                                       "weight": len(y) / n}))

    pixels = pd.concat(parts, ignore_index = True)
    pixels = pixels.sort_values(["y_idx", "x_idx"]).reset_index(drop = True)
    pixels["lat"] = cube.lat.values[pixels.y_idx]
    pixels["lon"] = cube.lon.values[pixels.x_idx]

    return pixels[["y_idx", "x_idx", "lat", "lon", "lc_class", "holdout", "weight"]]





def _at_pixels(da, pixels):
    """Values of `da` at the sampled pixels: (time, pixel) if `da` has time, else (pixel,)"""
    if "time" in da.dims:
        da = da.transpose("time", "lat", "lon")
    return da.isel(lat = xr.DataArray(pixels.y_idx.values, dims = "pixel"),
                   lon = xr.DataArray(pixels.x_idx.values, dims = "pixel")).values



def build_feature_table(cube, samples, pixels, T = 12, horizon = 3, weather = 'past'):
    """Combines a cube and the two sample lists returned by make_samples and sample_pixels
    into a flat table that can be fed to LightGBM, with one row of data per (t0, pixel) combination

    Parameters
    ----------
    cube : xr.Dataset
        Cube with DYNAMIC_VARS, STATIC_VARS, ndmi_anomaly, valid_mask, month_sin, month_cos
    samples : pd.DataFrame
        table returned by make_samples, containing t0_idx, t0, split
    pixels : pd.DataFrame
        table returned by sample_pixels, with x_idx, y_idx, lat, lon, lc_class, holdout, weight
    T : int
        number of previous months a sample has information about
    horizon : int
        number of future months the model targets
    weather : {'past', 'perfect'}
        if 'perfect' is selected, the actual LEAD_VARS of the target months are added as 
        input, to simulate what a seasonal forecast would add

    Returns
    -------
    table : pd.DataFrame
        contains the following columns, from left to right:
        - every column of 'samples' 
        - every column of 'pixels'
        - calendar encoded in sin/cos
        - static variables (elevation, slope, irrigated...)
        - lagged dynamic variables, referred to as {var}_lag{k}
        - leads, only if weather == 'perfect', referred to as {var}_lead{k}
        - target ndmi_anomaly at each horizon, referred to as y_h{h}


    Notes
    -----
    - key columns (lat, lon, lc_class, weight...) are not all features: lat/lon would
    let the model memorise location. Select features from static onwards
    

    """


    if weather not in ["past", "perfect"]:
        raise ValueError(f"weather must be 'past' or 'perfect', got {weather}")


    t0 = samples.t0_idx.values
    n_t = len(t0)
    n_p = len(pixels)

    # we need a column for each (t0, pixel) point: repeat t0 and tile pixel
    columns = {}
    shared = set(samples.columns) & set(pixels.columns)
    if shared:
        raise ValueError(f"samples and pixels share column names: {sorted(shared)}")
    for col in samples.columns:
        columns[col] = np.repeat(samples[col].values, n_p)
    for col in pixels.columns:
        columns[col] = np.tile(pixels[col].values, n_t)


    # add encoded calendar
    for col in ["month_sin", "month_cos"]:
        columns[col] = np.repeat(cube[col].values[t0].astype(np.float32), n_p)


    # add static inputs
    for col in STATIC_VARS:
        columns[col] = np.tile(_at_pixels(cube[col], pixels), n_t)


    # add dynamic lagged inputs
    for var in DYNAMIC_VARS:
        data = _at_pixels(cube[var], pixels).astype(np.float32)
        for k in range(T):
            columns[f"{var}_lag{k}"] = data[t0 - k].ravel()

    valid = _at_pixels(cube.valid_mask, pixels).astype(bool)
    for k in range(T):
        columns[f"valid_mask_lag{k}"] = valid[t0 - k].ravel().astype(np.uint8)


    # add actual weather of the target months (as input, not target, if weather == "perfect")
    if weather == "perfect":
        for var in LEAD_VARS:
            data = _at_pixels(cube[var], pixels).astype(np.float32)
            for h in range(1, horizon + 1):
                columns[f"{var}_lead{h}"] = data[t0 + h].ravel()


    # add ndmi_anomaly at h months lead as a target
    y = np.where(valid, _at_pixels(cube.ndmi_anomaly, pixels), np.nan).astype(np.float32)
    targets = [f"y_h{h}" for h in range(1, horizon + 1)]
    for h, name in zip(range(1, horizon + 1), targets):
        columns[name] = y[t0 + h].ravel()

    table = pd.DataFrame(columns)


    # mask: keep the held out pixels only for test, and ensure at least one valid target
    keep = (~(table.holdout & (table.split != "test"))) & table[targets].notna().any(axis = 1)
    
    return table[keep].reset_index(drop = True)













