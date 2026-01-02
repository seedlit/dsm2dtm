import os
from collections.abc import Mapping

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin

from dsm2dtm import algorithm, core


@pytest.fixture
def synthetic_dsm_path(tmp_path):
    """
    Creates a synthetic DSM file for testing.
    Scenario: 100x100 flat ground at elev=100.0 with a few objects.
    """
    path = tmp_path / "synthetic_dsm.tif"
    height, width = 100, 100
    data = np.full((height, width), 100.0, dtype=np.float32)
    # Add a "building"
    data[40:60, 40:60] = 120.0
    # Add "trees" (spikes)
    data[10, 10] = 115.0
    transform = from_origin(500000, 4000000, 1.0, 1.0)  # UTM-like
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=height,
        width=width,
        count=1,
        dtype=data.dtype,
        crs="EPSG:32631",
        transform=transform,
        nodata=-9999.0,
    ) as dst:
        dst.write(data, 1)
    return str(path)


def test_generate_dtm_numerical_properties(synthetic_dsm_path):
    """Verify numerical properties of the generated DTM using synthetic data."""
    dtm, profile = core.generate_dtm(synthetic_dsm_path)
    assert isinstance(dtm, np.ndarray)
    assert isinstance(profile, Mapping)
    assert dtm.shape == (100, 100)
    assert dtm.shape == (100, 100)
    # Building area (center) should be removed (i.e., < 120.0, close to 100.0)
    center_val = dtm[50, 50]
    assert center_val < 110.0, f"Building not removed, elev={center_val}"
    assert abs(center_val - 100.0) < 2.0, "Ground not restored correctly"
    # Corner (ground) should stay ~100.0
    assert abs(dtm[0, 0] - 100.0) < 0.5


def test_full_pipeline_save(synthetic_dsm_path, tmp_path):
    """Test the full pipeline: generate -> save."""
    dtm_path = str(tmp_path / "output_dtm.tif")
    dtm, profile = core.generate_dtm(synthetic_dsm_path)
    core.save_dtm(dtm, profile, dtm_path)
    assert os.path.isfile(dtm_path)
    with rasterio.open(dtm_path) as src:
        assert src.count == 1
        dtm_array = src.read(1)
        assert dtm_array.shape == dtm.shape
        assert dtm_array.dtype == dtm.dtype


def test_calculate_terrain_slope_flat():
    """Test slope calculation on a perfectly flat surface."""
    # 10x10 flat surface
    dsm = np.zeros((10, 10), dtype=np.float32)
    resolution = 1.0
    nodata = -9999.0
    slope = algorithm.calculate_terrain_slope(dsm, resolution, nodata)
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
    slope = algorithm.calculate_terrain_slope(dsm, resolution, nodata)
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
    slope = algorithm.calculate_terrain_slope(dsm, resolution, nodata)
    assert slope == 0.01


def test_get_adaptive_parameters():
    """Test parameter scaling based on resolution."""
    params_1m = algorithm.get_adaptive_parameters(resolution=1.0)
    params_05m = algorithm.get_adaptive_parameters(resolution=0.5)
    assert params_05m.pmf_initial_window >= params_1m.pmf_initial_window
    assert abs(params_05m.pmf_slope - 0.5 * params_1m.pmf_slope) < 1e-6


def test_progressive_morphological_filter_basic():
    """Test PMF removes a simple object on flat ground."""
    # 20x20 flat ground at z=10
    dsm = np.full((20, 20), 10.0, dtype=np.float32)
    # Add a building: 4x4 square at z=20 in the center
    dsm[8:12, 8:12] = 20.0
    nodata = -9999.0
    ground = algorithm.progressive_morphological_filter(
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
    refined = algorithm.refine_ground_surface(ground, nodata, smoothen_radius=1.0, elevation_threshold=2.0)
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
    dtm = algorithm.dsm_to_dtm(dsm, resolution, kernel_radius_meters=10.0, slope=0.1)
    assert dtm[10, 10] < 105.0
    # Building center
    assert dtm[32, 32] < 110.0
    # Ground should be preserved
    assert abs(dtm[0, 0] - 100.0) < 0.1


def test_dsm_context_loading(tmp_path):
    """Test _load_dsm handles a mock file."""
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
