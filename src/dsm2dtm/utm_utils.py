"""
Utilities for UTM CRS estimation.
"""

from pyproj.aoi import AreaOfInterest
from pyproj.database import query_utm_crs_info
from rasterio.crs import CRS


def estimate_utm_crs(lon: float, lat: float) -> CRS:
    """
    Estimate the UTM CRS for a given longitude and latitude using the EPSG database.

    This function queries the PROJ database (via pyproj) to find the official UTM zone
    for the given coordinate. This is more robust than simple longitude math as it
    correctly handles non-standard UTM zones (e.g., around Norway and Svalbard).

    Args:
        lon (float): Longitude in decimal degrees.
        lat (float): Latitude in decimal degrees.

    Returns:
        CRS: A rasterio CRS object corresponding to the best UTM zone.
             Returns a default estimate based on longitude if the database query fails.
    """
    try:
        # Query PROJ database for the best UTM projection for this point
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
            # The list is usually sorted by "best match". We take the first one.
            return CRS.from_epsg(utm_crs_list[0].code)
    except Exception as e:
        print(f"Warning: pyproj database query failed ({e}). Falling back to simple calculation.")

    # --- Fallback: Simple math (Standard UTM zones) ---
    # Zone calculation: (lon + 180) / 6 rounded up
    zone = int((lon + 180) / 6) + 1
    if zone > 60:
        zone = 60

    south = lat < 0
    # EPSG 326xx (North) or 327xx (South)
    base = 32700 if south else 32600
    epsg = base + zone
    return CRS.from_epsg(epsg)
