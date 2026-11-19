import pandas as pd
import xarray as xr
import numpy as np
from src.dataset import make_samples, SPLITS, split_of_year, land_mask, holdout_mask
import itertools
import pytest


TIMES = pd.date_range("2017-04", "2025-12", freq = "MS")
T = 12
H = 3


def test_splits_do_not_overlap():
    ranges = {}
    for split, (yyyy_0, yyyy_f) in SPLITS.items():
        ranges[split] = set(range(yyyy_0, yyyy_f + 1))

    for a, b in itertools.combinations(ranges, 2):
        common = ranges[a] & ranges[b]
        assert common == set(), f"splits {a} and {b} overlap in years {common}"


def test_splits_cover_all_years():
    years = set()
    for split, (yyyy_0, yyyy_f) in SPLITS.items():
        years.update(range(yyyy_0, yyyy_f + 1))

    missing_years = set(TIMES.year) - years
    assert not missing_years, f"years not covered by any split: {sorted(missing_years)}"



def test_make_samples_split_counts():
    samples = make_samples(TIMES, T = T, horizon = H)
    counts = samples["split"].value_counts().to_dict()
    assert counts == {"train" : 43, "val" : 10, "test": 34}


def assert_samples_in_splits(samples, splits = SPLITS):
    for sample in samples.itertuples():
        (yyyy_0, yyyy_f) = splits[sample.split]
        for h in range(1, 1 + H):
            year = TIMES[int(sample.t0_idx) + h].year
            assert yyyy_0 <= year <= yyyy_f, f"t0 = {sample.t0:%Y-%m}, h={h}: target year {year} is outside {sample.split} split {yyyy_0}-{yyyy_f}"


def test_make_samples_targets_inside_split():
    """check no sample leaks info between splits"""

    samples = make_samples(TIMES, T = T, horizon = H)
    assert_samples_in_splits(samples)


def test_make_samples_uses_custom_splits():
    new_splits = {"train": (2017, 2020),
                  "val"  : (2021, 2021),
                  "test" : (2022, 2025)}

    samples = make_samples(TIMES, T = T, horizon = H, splits = new_splits)
    assert_samples_in_splits(samples, splits = new_splits)



def test_make_samples_windows_inside_time_axis():
    samples = make_samples(TIMES, T = T, horizon = H, splits = SPLITS)

    assert (samples.t0_idx - T + 1 >= 0).all()
    assert (samples.t0_idx + H < len(TIMES)).all()


def test_make_samples_drops_years_outside_splits():
    times_long = pd.date_range("2017-04", "2026-12", freq = "MS")
    samples = make_samples(times_long, T = T, horizon = H, splits = SPLITS)
    counts = samples["split"].value_counts().to_dict()

    assert not samples.split.isna().any()
    assert counts == {"train": 43, "val": 10, "test": 34}


    
def make_toy_cube():
    ds = xr.Dataset(
        data_vars = {
            "landcover" : (["lat", "lon"], 40 * np.ones([3, 3], dtype = np.uint8)),
            "valid_mask": (["time", "lat", "lon"], np.ones([2, 3, 3], dtype = np.uint8))
        }
    )

    ds["landcover"][0, 0] = 80       # water 
    ds["landcover"][0, 1] = 50       # built 
    ds["valid_mask"][:, 2, 2] = 0    # never seen
    ds["valid_mask"][0, 2, 1] = 0    # seen sometimes

    return ds


def test_land_mask_excludes_water_built_up_and_unobserved():
    mask = land_mask(make_toy_cube())
    correct_mask = np.ones([3, 3], dtype = bool)
    correct_mask[0, 0] = False
    correct_mask[0, 1] = False
    correct_mask[2, 2] = False

    assert np.array_equal(mask.values, correct_mask), f"\nmask:\n {mask.values}  \ncorrect_mask:\n {correct_mask}"


def test_land_mask_is_2d_lat_lon():
    mask = land_mask(make_toy_cube())
    assert mask.dims == ('lat', 'lon')

def test_land_mask_is_boolean():
    mask = land_mask(make_toy_cube())
    assert mask.dtype == bool



def make_toy_grid(n_lat, n_lon):
    lats = np.linspace(42.1, 41.9, n_lat)
    lons = np.linspace(3.0, 3.2, n_lon)
    return xr.Dataset(coords = {"lat": lats, "lon": lons})


def test_holdout_mask_selects_eastern_columns():
    n_lat = 3
    n_lon = 10
    grid = make_toy_grid(n_lat = n_lat, n_lon = n_lon)
    mask = holdout_mask(grid, frac = 0.2, tile = 2)

    expected = np.zeros([n_lat, n_lon], dtype = bool)
    expected[:, 8:] = True

    assert np.array_equal(mask.values, expected), f"\nmask: \n{mask.values}   \nexpected_mask: \n{expected}"


def test_holdout_mask_boundary_snaps_west_to_tile_edge():
    n_lat = 3
    n_lon = 9
    tile = 2
    grid = make_toy_grid(n_lat = n_lat, n_lon = n_lon)
    mask = holdout_mask(grid, frac = 0.2, tile = tile)
    first_true = np.argmax(mask[0].values)
    
    expected = np.zeros([n_lat, n_lon], dtype = bool)
    expected[:, 6:] = True
    assert np.array_equal(mask.values, expected), f"\nmask: \n{mask.values}   \nexpected_mask: \n{expected}"
    
    leftover = first_true % tile
    assert leftover == 0, f"first held_out column is {first_true}, must be multiple of tile size {tile}"


def test_holdout_mask_raises_on_decreasing_lon():
    n_lat = 3
    n_lon = 9
    grid = make_toy_grid(n_lat = n_lat, n_lon = n_lon)
    reversed_grid = grid.isel(lon = slice(None, None, -1)) 

    with pytest.raises(ValueError, match = "increase"):
        holdout_mask(reversed_grid, frac = .2, tile = 2)



def test_holdout_mask_is_2d_bool():
    grid = make_toy_grid(3, 10)
    mask = holdout_mask(grid, frac = .2, tile = 2)

    assert mask.dims == ("lat", "lon"), f"mask.dims are {mask.dims}, should be ('lat', 'lon')"
    assert mask.dtype == bool, f"mask.dtype is {mask.dtype}, should be bool"





