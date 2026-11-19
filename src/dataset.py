import pandas as pd
import numpy as np
import xarray as xr


SPLITS = {"train": (2017, 2021),
          "val"  : (2022, 2022),
          "test" : (2023, 2025)}


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




