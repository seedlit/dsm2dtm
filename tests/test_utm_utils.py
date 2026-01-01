from rasterio.crs import CRS

from dsm2dtm.utm_utils import estimate_utm_crs


def test_estimate_utm_crs_north():
    """Test standard Northern Hemisphere location."""
    # New York City: ~74W, 40N -> Zone 18N (EPSG:32618)
    crs = estimate_utm_crs(-74.0, 40.7)
    assert crs == CRS.from_epsg(32618)


def test_estimate_utm_crs_south():
    """Test standard Southern Hemisphere location."""
    # Sydney: ~151E, 33S -> Zone 56S (EPSG:32756)
    crs = estimate_utm_crs(151.2, -33.8)
    assert crs == CRS.from_epsg(32756)


def test_estimate_utm_crs_equator_north():
    """Test point on Equator (treated as North by default logic)."""
    # 0.5 E is clearly Zone 31
    crs = estimate_utm_crs(0.5, 0.0)
    assert crs == CRS.from_epsg(32631)


def test_estimate_utm_crs_equator_south():
    """Test point just south of Equator."""
    # 0.5 E is clearly Zone 31 South
    crs = estimate_utm_crs(0.5, -0.0001)
    assert crs == CRS.from_epsg(32731)


def test_estimate_utm_crs_zone_boundaries():
    """Test longitude near zone boundaries."""
    # -5.5 should be Zone 30
    crs = estimate_utm_crs(-5.5, 40.0)
    assert crs == CRS.from_epsg(32630)
    # 0.5 should be Zone 31
    crs = estimate_utm_crs(0.5, 40.0)
    assert crs == CRS.from_epsg(32631)


def test_estimate_utm_crs_dateline():
    """Test International Date Line handling."""
    # 179.9 E -> Zone 60
    crs = estimate_utm_crs(179.9, 20.0)
    assert crs == CRS.from_epsg(32660)
    # -179.9 W -> Zone 1
    crs = estimate_utm_crs(-179.9, 20.0)
    assert crs == CRS.from_epsg(32601)


def test_estimate_utm_crs_edge_180():
    """Test explicit 180 longitude."""
    crs = estimate_utm_crs(180.0, 20.0)
    assert crs == CRS.from_epsg(32660)


def test_estimate_utm_crs_norway_exception():
    """Test Norway Zone 32V exception (extends to 3E-12E instead of 6E-12E)."""
    # Bergen, Norway: ~5.3E, 60.4N.
    crs = estimate_utm_crs(5.3, 60.4)
    assert crs.to_epsg() == 32631
    # Svalbard exception: Longyearbyen ~15.6E, 78.2N.
    crs_sval = estimate_utm_crs(15.6, 78.2)
    assert crs_sval.to_epsg() == 32633
