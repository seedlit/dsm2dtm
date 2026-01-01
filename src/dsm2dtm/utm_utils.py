from rasterio.crs import CRS


def estimate_utm_crs(lon: float, lat: float) -> CRS:
    """Estimated UTM CRS for a given lat/lon."""
    zone = int((lon + 180) / 6) + 1
    south = lat < 0
    # EPSG 326xx (North) or 327xx (South)
    base = 32700 if south else 32600
    epsg = base + zone
    return CRS.from_epsg(epsg)
