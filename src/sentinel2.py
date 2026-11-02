import numpy as np
import xarray as xr

from sentinelhub import (
    BBox,
    bbox_to_dimensions,
    CRS,
    SentinelHubRequest,
    DataCollection,
    MimeType,
)

from src.paths import RAW, make_data_folder, _month_list


def _index_dataset(img, start, coords, index):
    """
    function that converts one monthly index image (height, width, 2) into a dataset
    with dimensions (time, lat, lon) and variables index median (e.g. ndmi_median) and clear observation count
    """
    height, width = img.shape[:2]

    # pixel-center coordinates: the image is on a WGS84 grid,
    # rows go from north (max_lat) to south, columns from west to east
    min_lon, min_lat, max_lon, max_lat = coords
    lons = min_lon + (np.arange(width) + 0.5) * (max_lon - min_lon) / width
    lats = max_lat - (np.arange(height) + 0.5) * (max_lat - min_lat) / height

    img = img.astype(np.float32)[np.newaxis]
    return xr.Dataset(
        {
            f'{index}_median': (('time', 'lat', 'lon'), img[..., 0]),
            'clear_count': (('time', 'lat', 'lon'), img[..., 1]),
        },
        coords = {'time': [start], 'lat': lats, 'lon': lons},
    )



# normalised difference indices
INDEX_FORMULAS = {
    "ndmi": (["B8A", "B11"], "(s.B8A - s.B11) / (s.B8A + s.B11)"),   # vegetation water content
    "ndvi": (["B04", "B08"], "(s.B08 - s.B04) / (s.B08 + s.B04)"),   # vegetation greenness
}


def get_index(coords, resolution, dates, config, index = "ndmi", save_data = True, redownload = False):
    """Function that returns the monthly median of a Sentinel-2 vegetation index (NDMI, NDVI, ...),
    after filtering out clouds, shadows, water and snow
    inputs:
        - coords: Coordinate box delimiting the square over which to compute the index
        - resolution: Resolution of the requested data
        - dates: list of [[year, month]] dates for which to request data, one request per month
        - config: SentinelHub configuration
        - index: name of the index to be requested, from the keys of INDEX_FORMULAS
        - save_data: boolean determining whether to save requested data on disk
        - redownload: boolean determining whether to download data again (gets data saved on disk if not)
    outputs:
        - ds: xarray Dataset with dimensions (time, lat, lon) on pixel centers, with the monthly index median
        (e.g. ndmi_median) and the total number of clear (non-masked) observations used to compute each of them
    """

    if index not in INDEX_FORMULAS:
        raise ValueError(f"unknown index '{index}', available: {list(INDEX_FORMULAS)}")
    bands, formula = INDEX_FORMULAS[index]

    months = _month_list(dates)

    # SentinelHub specifications
    m_bbox = BBox(bbox = coords, crs=CRS.WGS84)
    m_size = bbox_to_dimensions(m_bbox, resolution = resolution)

    # select the index bands from sentinel data
    evalscript = """
        //VERSION=3
        function setup() {
            return {
                input: [{bands: [__BANDS__, "SCL", "dataMask"],units: "DN"}],
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
          const v = sample.filter(isClear).map(s => __FORMULA__);
          if (v.length == 0) return [NaN, 0];
          v.sort((a, b) => a - b);
          const m = Math.floor(v.length / 2);
          const median = v.length % 2 ? v[m] : (v[m - 1] + v[m]) / 2;
          return [median, v.length];
        }
    """.replace("__BANDS__", ",".join(f'"{b}"' for b in bands)).replace("__FORMULA__", formula)

    # define requesting function, one request per month
    def request_index(start, end):
        return SentinelHubRequest(
            evalscript = evalscript,
            input_data=[
                SentinelHubRequest.input_data(
                    data_collection=DataCollection.SENTINEL2_L2A.define_from(
                        "s2l2a", service_url=config.sh_base_url
                    ),
                    time_interval=(f"{start:%Y-%m-%d}", f"{end:%Y-%m-%d}"),
                )
            ],
            responses=[SentinelHubRequest.output_response("default", MimeType.TIFF)],
            bbox = m_bbox,
            size = m_size,
            config = config,
            data_folder = make_data_folder(coords, (start, end), root = RAW / "s2" / index)
        )

    # request data from SentinelHub month by month
    # (already saved months are read from disk unless redownload is True)
    parts = []
    for start, end in months:
        data = request_index(start, end).get_data(save_data = save_data, redownload = redownload)
        parts.append(_index_dataset(data[0], start, coords, index))

    return xr.concat(parts, dim = 'time')
