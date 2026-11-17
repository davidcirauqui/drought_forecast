import pandas as pd


SPLITS = {"train": (2017, 2021),
          "val"  : (2021, 2022),
          "test" : (2023, 2025)}


def split_of_year(year):
    for split, (yyyy_0, yyyy_f) in SPLITS.items():
        if yyyy_0 <= year <= yyyy_f:
            return split

def make_samples(times, T = 12, horizon = 3, splits = SPLITS):
    """function that defines which months can be used as references for a given 
    month of a training example, and to which split each one belongs to
    inputs:
        - times: monthly range defining the cube's time axis
        - T: length of the input window
        - horizon: number of predicted months ahead
        - splits: constant dictionary defining the train-val-test split years
    outputs:
        - sample 
       
    """


    samples = []
    for i in range(T - 1, len(times) - horizon):

        # add the split of each month
        splits_set = set([])
        for h in range(1, horizon + 1):
            split_h = split_of_year(times[i + h].year)
            splits_set.add(split_h)

        # if all targets are in the same split, add 
        if (len(splits_set) == 1):
            samples.append((i, times[i], splits_set.pop()))

    return pd.DataFrame(samples, columns = ["t0_idx", "t0", "split"])

        

        



