import pandas as pd
from src.dataset import make_samples, SPLITS, split_of_year
import itertools


TIMES = pd.date_range("2017-04", "2025-12", freq = "MS")
T = 12
H = 3

def test_split_counts():
    samples = make_samples(TIMES, T = T, horizon = H)
    counts = samples["split"].value_counts().to_dict()
    assert counts == {"train" : 43, "val" : 10, "test": 34}


def test_targets_inside_split():
    """check no sample leaks info between splits"""

    samples = make_samples(TIMES, T = T, horizon = H)

    for sample in samples.itertuples():
        (yyyy_0, yyyy_f) = SPLITS[sample.split]
        for h in range(1, 1 + H):
            year = TIMES[int(sample.t0_idx) + h].year
            assert yyyy_0 <= year <= yyyy_f, f"t0 = {sample.t0:%Y-%m}, h={h}: target year {year} is outside {sample.split} split {yyyy_0}-{yyyy_f}"


def test_splits_do_not_overlap():
    ranges = {}
    for split, (yyyy_0, yyyy_f) in SPLITS.items():
        ranges[split] = set(range(yyyy_0, yyyy_f + 1))

    for a, b in itertools.combinations(ranges, 2):
        common = ranges[a] & ranges[b]
        assert common == set(), f"splits {a} and {b} overlap in years {common}"


