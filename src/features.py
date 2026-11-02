import numpy as np
import xarray as xr


WORLDCOVER_CLASSES = {
    10: "tree_cover",
    20: "shrubland",
    30: "grassland",
    40: "cropland",
    50: "built_up",
    60: "bare_sparse",
    70: "snow_ice",
    80: "water",
    90: "herbaceous_wetland",
    95: "mangroves",
    100: "moss_lichen",
}


def terrain_features(elevation, flat_slope = 1):
    """Function that computes terrain variables from an elevation map in WGS84
    (compute them after regridding, never interpolate aspect: averaging 359 and 1 degrees gives 180)
    inputs:
        - elevation: xarray DataArray with dimensions (lat, lon) in metres, lat from north to south
        - flat_slope: slope in degrees below which ground is considered flat (aspect set to NaN)
    outputs:
        - ds: xarray Dataset with dimensions (lat, lon) and variables
            - slope in degrees
            - aspect: direction the slope faces, in degrees clockwise from north (NaN on flat ground)
            - northness: cos(aspect) * sin(slope), +1 steep north-facing, -1 steep south-facing, ~0 flat
            - eastness: sin(aspect) * sin(slope), +1 steep east-facing, -1 steep west-facing, ~0 flat
    """
    lats = elevation.lat.values
    lons = elevation.lon.values

    # pixel size in metres (1 degree of latitude ~ 111.32 km, longitude shrinks with cos(lat))
    dy = abs(lats[1] - lats[0]) * 111_320
    dx = abs(lons[1] - lons[0]) * 111_320 * np.cos(np.deg2rad(lats))[:, np.newaxis]

    # elevation change per metre towards north and east (rows go from north to south)
    dz_row, dz_col = np.gradient(elevation.values)
    dz_north = -dz_row / dy
    dz_east = dz_col / dx

    # slope and aspect (downhill direction, clockwise from north) in radians
    slope = np.arctan(np.hypot(dz_east, dz_north))
    aspect = np.arctan2(-dz_east, -dz_north)

    # aspect weighted by slope, model inputs: fade smoothly to 0 on flat ground, no threshold needed
    northness = np.cos(aspect) * np.sin(slope)
    eastness = np.sin(aspect) * np.sin(slope)

    # degrees for the output, aspect NaN on flat ground where it has no meaning
    slope_deg = np.degrees(slope)
    aspect_deg = np.where(slope_deg < flat_slope, np.nan, (np.degrees(aspect) + 360) % 360)

    return xr.Dataset(
        {
            'slope': (('lat', 'lon'), slope_deg.astype(np.float32)),
            'aspect': (('lat', 'lon'), aspect_deg.astype(np.float32)),
            'northness': (('lat', 'lon'), northness.astype(np.float32)),
            'eastness': (('lat', 'lon'), eastness.astype(np.float32)),
        },
        coords = {'lat': lats, 'lon': lons},
    )


def one_hot_landcover(landcover):
    """Function that one-hot encodes a WorldCover class map
    (compute it after regridding with method "mode", never interpolate the one-hot variables)
    inputs:
        - landcover: xarray DataArray with dimensions (lat, lon) and WORLDCOVER_CLASSES codes
    outputs:
        - ds: xarray Dataset with one 0/1 variable per class (lc_tree_cover, lc_cropland, ...)
    """
    return xr.Dataset({f'lc_{name}': (landcover == code).astype(np.uint8)
                       for code, name in WORLDCOVER_CLASSES.items()})
