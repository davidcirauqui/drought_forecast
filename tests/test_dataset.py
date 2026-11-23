import pandas as pd
import xarray as xr
import numpy as np
from src.dataset import make_samples, SPLITS, split_of_year, land_mask, holdout_mask
from src.dataset import normalisation_stats, sample_pixels
import itertools
import pytest
import json


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







def make_toy_stats_cube():
    time = pd.to_datetime(["2020-01-01", "2020-02-01", "2024-01-01", "2024-02-01"])
    lat = np.linspace(0, 1, 2)
    lon = np.linspace(0, 1, 4)

    n_lat = len(lat)
    n_lon = len(lon)
    n_time = len(time)

    time_dependent = np.zeros([n_time, n_lat, n_lon])
    time_dependent[0:2, 0, :-1] = 2   # training months and pixels: half twos and half zeros
    time_dependent[2:, :, :] = 100    # test months: 100 everywhere
    time_dependent[:, :, -1] = 1000   # holdout (east strip): 1000 everywhere

    with_nans = time_dependent.copy()
    with_nans[1, :, :] = np.nan   # month with no data

    time_independent = np.zeros([n_lat, n_lon])
    time_independent[0, :-1] = 2     # training pixels: half twos and half zeros
    time_independent[:, -1] = 1000   # holdout (east strip): 1000 everywhere

    constant = np.ones([n_lat, n_lon])

    ds = xr.Dataset(
        data_vars = {
            'time_dependent' : (["time", "lat", "lon"], time_dependent),
            'with_nans' : (["time", "lat", "lon"], with_nans),
            'time_independent' : (["lat", "lon"], time_independent),
            'constant' : (["lat", "lon"], constant)
        },
        coords = {
            'lat' : lat,
            'lon' : lon,
            'time': time
        }
    )


    train_mask = np.ones([n_lat, n_lon], dtype = bool)
    train_mask[:, -1] = False
    train_mask = xr.DataArray(train_mask, dims = ['lat', 'lon'])

    return ds, train_mask




def test_normalisation_stats_excludes_test_years_and_holdout():
    cube, train_mask = make_toy_stats_cube()
    stats = normalisation_stats(cube, train_mask, variables = ['time_dependent'], years = (2020, 2020))

    mean = stats['time_dependent']['mean']
    assert mean == pytest.approx(1), f"mean of time dependent variable should be 1, is {mean}"

    std = stats['time_dependent']['std']
    assert std == pytest.approx(1), f"std of time dependent variable should be 1, is {std}"


def test_normalisation_stats_static_variable_excludes_holdout():
    cube, train_mask = make_toy_stats_cube()
    stats = normalisation_stats(cube, train_mask, variables = ['time_independent'], years = (2020, 2020))

    mean = stats['time_independent']['mean']
    assert mean == pytest.approx(1), f"mean of time independent variable should be 1, is {mean}"

    std = stats['time_independent']['std']
    assert std == pytest.approx(1), f"std of time independent variable should be 1, is {std}"



def test_normalisation_stats_ignores_nan():
    cube, train_mask = make_toy_stats_cube()
    stats = normalisation_stats(cube, train_mask, variables = ['with_nans'], years = (2020, 2020))

    mean = stats['with_nans']['mean']
    assert mean == pytest.approx(1), f"mean of variable with nans should be 1, is {mean}"

    std = stats['with_nans']['std']
    assert std == pytest.approx(1), f"std of variable with nans should be 1, is {std}"



def test_normalisation_stats_raises_null_std():
    cube, train_mask = make_toy_stats_cube()
    with pytest.raises(ValueError, match = 'std=0'):
        normalisation_stats(cube, train_mask, variables = ['constant'], years = (2020, 2020))



def test_normalisation_stats_output_is_json_serialisable():
    cube, train_mask = make_toy_stats_cube()
    stats = normalisation_stats(cube, train_mask, variables = ['time_dependent', 'time_independent'],
                                 years = (2020, 2020))

    assert json.loads(json.dumps(stats)) == stats




def make_toy_lc_cube(n_lat = 20, n_lon = 20):
    lc = np.full((n_lat, n_lon), 40, dtype = np.uint8)     # cropland
    lc[:, n_lon // 2:] = 10                                # tree cover on the east half
    lc[:3, :3] = 80                                        # water patch
    ds = xr.Dataset({"landcover": (["lat", "lon"], lc)},
                    coords = {"lat": np.linspace(42.1, 41.9, n_lat),
                              "lon": np.linspace(2.97, 3.19, n_lon)})
    land = ds.landcover != 80
    holdout = holdout_mask(ds, frac = .2, tile = 4)
    return ds, land, holdout


def test_sample_pixels_only_land_and_no_duplicates():
    ds, land, holdout = make_toy_lc_cube()
    px = sample_pixels(ds, land, holdout, frac = .5, min_per_class = 1)
    assert land.values[px.y_idx, px.x_idx].all()
    assert not px.duplicated(["y_idx", "x_idx"]).any()


def test_sample_pixels_holdout_flag_matches_mask():
    ds, land, holdout = make_toy_lc_cube()
    px = sample_pixels(ds, land, holdout, frac = 1, min_per_class = 1)
    assert (px.holdout.values == holdout.values[px.y_idx, px.x_idx]).all()
    assert px.holdout.any() and (~px.holdout).any()


def test_sample_pixels_min_per_class_capped_at_available():
    ds, land, holdout = make_toy_lc_cube()
    px = sample_pixels(ds, land, holdout, frac = 0, min_per_class = 5)
    counts = px.groupby(["holdout", "lc_class"]).size()
    for (is_hold, cls), n in counts.items():
        available = int((land & (holdout == is_hold) & (ds.landcover == cls)).sum())
        assert n == min(5, available)


def test_sample_pixels_frac_wins_and_no_group_missing():
    ds, land, holdout = make_toy_lc_cube()
    px = sample_pixels(ds, land, holdout, frac = .5, min_per_class = 1)
    counts = px.groupby(["holdout", "lc_class"]).size().to_dict()

    expected = {}
    for is_hold in (False, True):
        region = land & (holdout == is_hold)
        for cls in np.unique(ds.landcover.values[region.values]):
            available = int((region & (ds.landcover == cls)).sum())
            expected[(is_hold, int(cls))] = int(round(.5 * available))

    assert counts == expected


def test_sample_pixels_weights_sum_to_available_pixels():
    ds, land, holdout = make_toy_lc_cube()
    px = sample_pixels(ds, land, holdout, frac = .1, min_per_class = 3)
    sums = px.groupby(["holdout", "lc_class"]).weight.sum()
    for (is_hold, cls), w in sums.items():
        available = int((land & (holdout == is_hold) & (ds.landcover == cls)).sum())
        assert w == pytest.approx(available)
    assert px.weight.sum() == pytest.approx(int(land.sum()))


def test_sample_pixels_seed_reproducible():
    ds, land, holdout = make_toy_lc_cube()
    a = sample_pixels(ds, land, holdout, frac = .1, min_per_class = 3, seed = 0)
    b = sample_pixels(ds, land, holdout, frac = .1, min_per_class = 3, seed = 0)
    c = sample_pixels(ds, land, holdout, frac = .1, min_per_class = 3, seed = 1)
    pd.testing.assert_frame_equal(a, b)
    assert not a.equals(c)



