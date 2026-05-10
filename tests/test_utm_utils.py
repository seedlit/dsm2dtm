from dsm2dtm.utm_utils import estimate_utm_crs


def test_estimate_utm_crs_north():
    """Test standard Northern Hemisphere location."""
    # New York City: ~74W, 40N -> Zone 18N (EPSG:32618)
    assert estimate_utm_crs(-74.0, 40.7) == 32618


def test_estimate_utm_crs_south():
    """Test standard Southern Hemisphere location."""
    # Sydney: ~151E, 33S -> Zone 56S (EPSG:32756)
    assert estimate_utm_crs(151.2, -33.8) == 32756


def test_estimate_utm_crs_equator_north():
    """Test point on Equator (treated as North by default logic)."""
    assert estimate_utm_crs(0.5, 0.0) == 32631


def test_estimate_utm_crs_equator_south():
    """Test point just south of Equator."""
    assert estimate_utm_crs(0.5, -0.0001) == 32731


def test_estimate_utm_crs_zone_boundaries():
    """Test longitude near zone boundaries."""
    assert estimate_utm_crs(-5.5, 40.0) == 32630
    assert estimate_utm_crs(0.5, 40.0) == 32631


def test_estimate_utm_crs_dateline():
    """Test International Date Line handling."""
    assert estimate_utm_crs(179.9, 20.0) == 32660
    assert estimate_utm_crs(-179.9, 20.0) == 32601


def test_estimate_utm_crs_edge_180():
    """Test explicit 180 longitude."""
    assert estimate_utm_crs(180.0, 20.0) == 32660


def test_estimate_utm_crs_norway_exception():
    """Test Norway/Svalbard region — pyproj returns the standard EPSG zone."""
    # Bergen, Norway ~5.3E, 60.4N falls in standard zone 31N (EPSG 32631).
    assert estimate_utm_crs(5.3, 60.4) == 32631
    # Longyearbyen, Svalbard ~15.6E, 78.2N falls in standard zone 33N (EPSG 32633).
    assert estimate_utm_crs(15.6, 78.2) == 32633
