import numpy as np
import xarray as xr
from scipy import stats


def compute_anomaly(da, train_years, min_obs = 30):
    """Function that computes standardised anomalies per pixel and calendar month
    (mean and std from the training years only, so that no information leaks from validation or test)
    inputs:
        - da: xarray DataArray with dimensions (time, lat, lon), e.g. ndmi_median
        - train_years: time slice of the training period, e.g. slice("2017", "2021")
        - min_obs: minimum number of training observations per pixel, NaN below it
    outputs:
        - da: xarray DataArray with the same dimensions, (value - monthly mean) / std
    """
    # compute means among training split
    train = da.sel(time = train_years)

    # mean per pixel (mean is stable with few observations)
    mean = train.groupby("time.month").mean("time")

    # deseasonalise
    deviations = train.groupby("time.month") - mean

    # compute std over deseasonalised data
    std = deviations.std("time")

    return ((da.groupby("time.month") - mean) / std).where(train.count("time") >= min_obs)


def spi(tp_mm, k, ref = slice("1991", "2020")):
    """SPI-k per grid cell: k-month precipitation total, compared with the same calendar month
    in the reference period through a gamma distribution, expressed as a standard normal value"""
    acc = tp_mm.rolling(time = k).sum()
    out = xr.full_like(acc, np.nan)
    for m in range(1, 13):
        in_m = (acc.time.dt.month == m).values
        ref_m = acc.sel(time = ref)
        ref_m = ref_m.isel(time = (ref_m.time.dt.month == m).values)
        for i in range(acc.sizes["lat"]):
            for j in range(acc.sizes["lon"]):
                shape, loc, scale = stats.gamma.fit(ref_m[:, i, j].dropna("time"), floc = 0)
                p = stats.gamma.cdf(acc[in_m, i, j], shape, loc, scale)
                out[in_m, i, j] = stats.norm.ppf(np.clip(p, 1e-6, 1 - 1e-6))
    return out
