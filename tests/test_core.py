import os

import numpy as np
import pytest
import rasterio

from dsm2dtm import core


@pytest.fixture(scope="module")
def dsm_path():
    path = "data/test_dsm.tif"
    if not os.path.exists(path):
        pytest.fail(f"Test data not found at {path}")
    return path


def test_main_numerical_properties(dsm_path):
    """Verify numerical properties of the generated DTM."""
    with rasterio.open(dsm_path) as src:
        dsm = src.read(1)

    dtm_path = core.main(dsm_path, "temp_data")
    with rasterio.open(dtm_path) as src:
        dtm = src.read(1)

    assert np.mean(dtm) <= np.mean(dsm)
    assert np.max(dtm) <= np.max(dsm)
    assert round(np.mean(dtm), 1) == 58.8
    assert round(np.max(dtm), 1) == 60.6
    assert round(np.min(dtm), 1) == 0.0
    # TODO handle nodata


def test_main_function(dsm_path, tmp_path):
    """Test the main entry point function end-to-end."""
    dtm_path = core.main(dsm_path, str(tmp_path))
    assert os.path.isfile(dtm_path)
    assert dtm_path.endswith(".tif")
    with rasterio.open(dtm_path) as src:
        assert src.count == 1
        dtm_array = src.read(1)
        assert dtm_array.shape[0] > 0
        assert dtm_array.shape[1] > 0


def test_calculate_terrain_slope_flat():
    """Test slope calculation on a perfectly flat surface."""
    # 10x10 flat surface
    dsm = np.zeros((10, 10), dtype=np.float32)
    resolution = 1.0
    nodata = -9999.0
    slope = core.calculate_terrain_slope(dsm, resolution, nodata)
    # Gradient is 0, median slope should be clamped to min (0.01)
    assert slope == 0.01


def test_calculate_terrain_slope_gradient():
    """Test slope calculation on a constant gradient."""
    # 1m rise per 1m run in x-direction
    x = np.linspace(0, 9, 10)
    y = np.linspace(0, 9, 10)
    xv, yv = np.meshgrid(x, y)
    dsm = xv.astype(np.float32)  # z = x
    resolution = 1.0
    nodata = -9999.0
    slope = core.calculate_terrain_slope(dsm, resolution, nodata)
    # Expected: dx=1, dy=0. Slope = sqrt(1^2 + 0^2) = 1.0
    # Allow some floating point tolerance
    assert abs(slope - 1.0) < 1e-4


def test_calculate_terrain_slope_nodata():
    """Test slope calculation ignores nodata values."""
    dsm = np.zeros((10, 10), dtype=np.float32)
    nodata = -9999.0
    # Set half to nodata
    dsm[:, 5:] = nodata
    resolution = 1.0
    slope = core.calculate_terrain_slope(dsm, resolution, nodata)
    assert slope == 0.01


def test_get_adaptive_parameters():
    """Test parameter scaling based on resolution."""
    # Test 1m resolution
    params_1m = core.get_adaptive_parameters(resolution=1.0)
    # Test 0.5m resolution (pixels should be approx double the 1m values for window sizes)
    params_05m = core.get_adaptive_parameters(resolution=0.5)
    # Window size in pixels should increase as resolution gets finer (smaller meters/pixel)
    assert params_05m.pmf_initial_window >= params_1m.pmf_initial_window
    assert abs(params_05m.pmf_slope - 0.5 * params_1m.pmf_slope) < 1e-6


def test_progressive_morphological_filter_basic():
    """Test PMF removes a simple object on flat ground."""
    # 20x20 flat ground at z=10
    dsm = np.full((20, 20), 10.0, dtype=np.float32)
    # Add a building: 4x4 square at z=20 in the center
    dsm[8:12, 8:12] = 20.0
    nodata = -9999.0
    ground = core.progressive_morphological_filter(
        dsm,
        nodata,
        initial_window=3,
        max_window=9,  # Should be enough to cover the 4x4 object
        slope=0.1,
        initial_threshold=0.5,
        max_threshold=3.0,
    )
    assert ground[10, 10] < 11.0  # Should be closer to 10 than 20
    assert ground[0, 0] == 10.0  # Unaffected ground


def test_refine_ground_surface():
    """Test refinement removes spikes."""
    # Flat ground
    ground = np.full((10, 10), 10.0, dtype=np.float32)
    nodata = -9999.0
    # Add a single pixel spike that PMF might have missed or created
    ground[5, 5] = 15.0
    refined = core.refine_ground_surface(ground, nodata, smoothen_radius=1.0, elevation_threshold=2.0)
    assert refined[5, 5] == nodata
    assert refined[0, 0] == 10.0


def test_dsm_to_dtm_integration_small():
    """Test the full dsm_to_dtm pipeline on a small synthetic array."""
    dsm = np.full((50, 50), 100.0, dtype=np.float32)
    # Add some 'trees' (single pixel spikes)
    dsm[10, 10] = 110.0
    dsm[20, 25] = 115.0
    # Add a 'building'
    dsm[30:35, 30:35] = 120.0
    resolution = (1.0, 1.0)  # 1m resolution
    dtm = core.dsm_to_dtm(dsm, resolution, kernel_radius_meters=10.0, slope=0.1)
    assert dtm[10, 10] < 105.0
    # Building center
    assert dtm[32, 32] < 110.0
    # Ground should be preserved
    assert abs(dtm[0, 0] - 100.0) < 0.1


def test_dsm_context_loading(tmp_path):
    """Test _load_dsm handles a mock file."""
    # Create a dummy GTiff
    import rasterio
    from rasterio.transform import from_origin

    dsm_path = tmp_path / "test.tif"
    arr = np.zeros((10, 10), dtype=np.float32)
    transform = from_origin(500000, 4000000, 1, 1)

    with rasterio.open(
        dsm_path,
        "w",
        driver="GTiff",
        height=10,
        width=10,
        count=1,
        dtype=arr.dtype,
        crs="EPSG:32631",  # UTM
        transform=transform,
    ) as dst:
        dst.write(arr, 1)
    ctx = core._load_dsm(str(dsm_path))
    assert ctx.is_reprojected is False
    assert ctx.dsm.shape == (10, 10)
    assert ctx.resolution[0] == 1.0
