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


def get_ndmi_mat(coords, resolution, dates, config):
    """Function that returns the NDMI index
    inputs:
        - coords: Coordinate box delimiting the square over which to compute the NDMI index
        - resolution: Resolution of the requested data
        - dates: Dates delimiting the time span over which the data is requested. The function computes the NDMI index on the less cloudy day within that period of time
        - config: SentinelHub configuration
    outputs:
        - ndmi_mat: array containing the computed NDMI index over the specified coordinate box and dates
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
                input: [{
                    bands: ["B8A","B11"],
                    units: "DN"
                }],
                output: {
                    bands: 2,
                    sampleType: "INT16"
                }
            };
        }

        function evaluatePixel(sample) {
            return [sample.B8A, sample.B11];
        }
    """

    # define requesting function
    request_ndmi = SentinelHubRequest(
        evalscript = evalscript_ndmi,
        input_data=[
            SentinelHubRequest.input_data(
                data_collection=DataCollection.SENTINEL2_L1C.define_from(
                    "s2l1c", service_url=config.sh_base_url
                ),
                time_interval=(dates[0], dates[1]),
                mosaicking_order=MosaickingOrder.LEAST_CC,
            )
        ],
        responses=[SentinelHubRequest.output_response("default", MimeType.TIFF)],
        bbox = m_bbox,
        size = m_size,
        config = config,
    )

    # request data from SentinelHub
    data = request_ndmi.get_data()

    # process NIR and SWIR chanels' data into NDMI index
    n = data[0].reshape(data[0].shape[0] * data[0].shape[1], 2)
    df = pd.DataFrame(n.astype(np.float32))
    df.columns = ['nir', 'swir']
    df['ndmi'] = (df.nir - df.swir) / (df.nir + df.swir)
    

    # reshape into matrix for plotting purposes
    ndmi_mat = df.ndmi.to_numpy()
    ndmi_mat = ndmi_mat.reshape(data[0].shape[0], data[0].shape[1])

    return ndmi_mat