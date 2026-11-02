import pandas as pd
from pathlib import Path


# absolute paths from the repo root, so that notebooks find the data wherever they are run from
ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"


def _fmt_coord(value, pos, neg, ndigits):
    """
    function that converts coordinates with negatives values to cardinal directions
    example: 2.97 -> '2.97E', -0.5 -> '0.50W'
    """
    return f"{abs(value):.{ndigits}f}{pos if value >= 0 else neg}"


def _area_name(bbox_coords, ndigits=2):
    """
    function that gives a folder name for a coordinate box
    example: [2.97, 41.92, 3.19, 42.10] -> '2.97E-3.19E_41.92N-42.10N'
    """
    min_lon, min_lat, max_lon, max_lat = bbox_coords
    return (f"{_fmt_coord(min_lon, 'E', 'W', ndigits)}-{_fmt_coord(max_lon, 'E', 'W', ndigits)}_"
            f"{_fmt_coord(min_lat, 'N', 'S', ndigits)}-{_fmt_coord(max_lat, 'N', 'S', ndigits)}")


def make_data_folder(bbox_coords, time_interval, root=RAW / "s2", ndigits=2):
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
