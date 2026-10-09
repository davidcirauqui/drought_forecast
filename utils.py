import numpy as np
import pandas as pd
from pathlib import Path

from sentinelhub import (
    BBox,
    bbox_to_dimensions,
    CRS,
    SentinelHubCatalog,
    SentinelHubRequest,
    DataCollection,
    MimeType,
)


def _fmt_coord(value, pos, neg, ndigits):
    """
    function that converts coordinates with negatives values to cardinal directions
    example: 2.97 -> '2.97E', -0.5 -> '0.50W'
    """
    return f"{abs(value):.{ndigits}f}{pos if value >= 0 else neg}"

def make_data_folder(bbox_coords, time_interval, root="data/raw/s2", ndigits=2):
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



def get_ndmi(coords, resolution, dates, config, save_data = True, redownload = False):
    """Function that returns the monthly median NDMI index, after filtering out
      clouds, shadows, water and snow
    inputs:
        - coords: Coordinate box delimiting the square over which to compute the NDMI index
        - resolution: Resolution of the requested data
        - dates: Dates delimiting the time span over which the data is requested.
        - config: SentinelHub configuration
        - save_data: boolean determining whether to save requested data on disk
        - redownload: boolean determining whether to download data again (takes data from saved if not)
    outputs:
        - ndmi_mat: matrix containing the computed NDMI index over the specified coordinate box and dates
        - df: dataframe with NDMI median values and the total number of clear (non-masked)
        observations used to compute each of them
    """


    # SentinelHub specifications
    m_bbox = BBox(bbox = coords, crs=CRS.WGS84)
    m_size = bbox_to_dimensions(m_bbox, resolution = resolution)
    catalog = SentinelHubCatalog(config = config)

    # select NIR (B8A) and SWIR (B11) channels from sentinel data
    evalscript_ndmi = """
        //VERSION=3
        function setup() {
            return {
                input: [{bands: ["B8A","B11", "SCL", "dataMask"],units: "DN"}],
                output: {bands: 2, sampleType: "FLOAT32"},
                mosaicking: "ORBIT"
            };
        }

        // drop clouds and shadows using Scene Classification Band (SCL)
        // SCL classes kept: 4 = vegetation, 5 = bare soil, 7 = unclass
        // classes dropped: clouds, shadows, water, snow
        function isClear(s) { return s.dataMask == 1 && [4, 5, 7].includes(s.SCL);}

        // return median
        function evaluatePixel(sample) {
          const v = sample.filter(isClear).map(s => (s.B8A - s.B11) / (s.B8A + s.B11));
          if (v.length == 0) return [NaN, 0];
          v.sort((a, b) => a - b);
          const m = Math.floor(v.length / 2);
          const median = v.length % 2 ? v[m] : (v[m - 1] + v[m]) / 2;
          return [median, v.length];
        }
    """

    # define requesting function
    request_ndmi = SentinelHubRequest(
        evalscript = evalscript_ndmi,
        input_data=[
            SentinelHubRequest.input_data(
                data_collection=DataCollection.SENTINEL2_L2A.define_from(
                    "s2l2a", service_url=config.sh_base_url
                ),
                time_interval=(dates[0], dates[1]),
            )
        ],
        responses=[SentinelHubRequest.output_response("default", MimeType.TIFF)],
        bbox = m_bbox,
        size = m_size,
        config = config,
        data_folder = make_data_folder(coords, dates)
    )

    # request data from SentinelHub
    data = request_ndmi.get_data(save_data = save_data, redownload = redownload)
    n = data[0].reshape(data[0].shape[0] * data[0].shape[1], 2)
    df = pd.DataFrame(n.astype(np.float32))
    df.columns = ['ndmi_median', 'clear_count']


    # reshape into matrix for plotting purposes
    ndmi_mat = df.ndmi_median.to_numpy()
    ndmi_mat = ndmi_mat.reshape(data[0].shape[0], data[0].shape[1])


    return ndmi_mat, df