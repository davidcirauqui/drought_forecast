import pandas as pd


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

        

        



