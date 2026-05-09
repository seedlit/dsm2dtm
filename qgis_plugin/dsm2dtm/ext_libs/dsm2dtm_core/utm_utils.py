"""
Utilities for UTM CRS estimation.

Pure pyproj — no rasterio. The QGIS plugin vendors this module verbatim.
Callers wrap the returned EPSG code in whatever CRS type they need.
"""

from pyproj.aoi import AreaOfInterest
from pyproj.database import query_utm_crs_info


def estimate_utm_crs(lon: float, lat: float) -> int:
    """
    Estimate the EPSG code of the UTM CRS for a given longitude and latitude.

    Queries the PROJ database (via pyproj) so non-standard zones (e.g. Norway's
    32V, Svalbard's 31X/33X/35X/37X) are honored. Falls back to simple math
    if the database query fails.

    Args:
        lon (float): Longitude in decimal degrees.
        lat (float): Latitude in decimal degrees.

    Returns:
        int: EPSG code (32601..32660 for North, 32701..32760 for South).
    """
    try:
        utm_crs_list = query_utm_crs_info(
            datum_name="WGS 84",
            area_of_interest=AreaOfInterest(
                west_lon_degree=lon,
                south_lat_degree=lat,
                east_lon_degree=lon,
                north_lat_degree=lat,
            ),
        )
        if utm_crs_list:
            return int(utm_crs_list[0].code)
    except Exception:
        pass

    # Fallback: standard zones via simple math. Wrap so lon=180 → zone 1.
    zone = int((lon + 180) / 6) % 60 + 1
    base = 32700 if lat < 0 else 32600
    return base + zone
