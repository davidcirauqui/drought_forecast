import numpy as np
import pandas as pd
from scipy import stats
import requests

from sentinelhub import (
    BBox,
    bbox_to_dimensions,
    CRS,
    SentinelHubCatalog,
    SentinelHubRequest,
    DataCollection,
    MimeType,
    MosaickingOrder,
)


def get_ndmi(coords, resolution, dates, config):
    """Function that returns the monthly median NDMI index, after filtering out
      clouds, shadows, water and snow
    inputs:
        - coords: Coordinate box delimiting the square over which to compute the NDMI index
        - resolution: Resolution of the requested data
        - dates: Dates delimiting the time span over which the data is requested.
        - config: SentinelHub configuration
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
    )

    # request data from SentinelHub
    data = request_ndmi.get_data()
    n = data[0].reshape(data[0].shape[0] * data[0].shape[1], 2)
    df = pd.DataFrame(n.astype(np.float32))
    df.columns = ['ndmi_median', 'clear_count']


    # reshape into matrix for plotting purposes
    ndmi_mat = df.ndmi_median.to_numpy()
    ndmi_mat = ndmi_mat.reshape(data[0].shape[0], data[0].shape[1])


    return ndmi_mat, df