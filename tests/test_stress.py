import time

import numpy as np

from dsm2dtm.algorithm import dsm_to_dtm

# --- Helpers ---


def create_synthetic_array(shape=(100, 100), base_val=100.0, nodata=-9999.0):
    """Creates a flat float32 array."""
    return np.full(shape, base_val, dtype=np.float32)


def add_building(data, x, y, w, h, height=20.0):
    """Adds a rectangular block."""
    data[y : y + h, x : x + w] += height


def add_noise(data, magnitude=1.0):
    """Adds Gaussian noise."""
    noise = np.random.normal(0, magnitude, data.shape).astype(np.float32)
    data += noise


def add_vegetation(data, density=0.1, height_range=(2.0, 10.0)):
    """Adds scattered spikes (vegetation)."""
    num_trees = int(data.size * density)
    idx = np.random.choice(data.size, num_trees, replace=False)
    heights = np.random.uniform(height_range[0], height_range[1], num_trees)
    data.ravel()[idx] += heights


# --- Tests ---


def test_large_dsm_performance():
    """
    Stress test with a larger DSM (2048 x 2048 pixels).
    Evaluate execution time to ensure it finishes in reasonable time.
    """
    # 2048x2048 is approx 4MP.
    shape = (2048, 2048)
    dsm = create_synthetic_array(shape)
    add_noise(dsm, magnitude=0.5)

    start_time = time.time()
    dtm = dsm_to_dtm(dsm, resolution=(1.0, 1.0), nodata=-9999.0)
    end_time = time.time()

    duration = end_time - start_time
    print(f"\nLarge DSM ({shape}) processed in {duration:.2f}s")

    # Assert it takes less than 60s (generous for 4MP on varied hardware)
    assert duration < 60.0
    assert dtm.shape == shape


def test_very_high_resolution_dsm():
    """
    Test algorithm behavior with very high resolution data (e.g., 1cm/pixel).
    Checks if adaptive parameters scale correctly.
    """
    shape = (500, 500)
    dsm = create_synthetic_array(shape)
    # Add a small object (1m wide = 100 pixels at 1cm res)
    add_building(dsm, 200, 200, 100, 100, height=5.0)

    # Resolution 0.01m (1cm)
    dtm = dsm_to_dtm(dsm, resolution=(0.01, 0.01), nodata=-9999.0)

    # Object should be removed (eroded)
    # Center pixel was 105.0, should become ~100.0
    center_val = dtm[250, 250]
    assert abs(center_val - 100.0) < 0.5


def test_mixed_resolution_handling():
    """
    Test with non-square pixels or unusual aspect ratios.
    """
    shape = (100, 100)
    dsm = create_synthetic_array(shape)

    # 0.5m x 2.0m pixels
    dtm = dsm_to_dtm(dsm, resolution=(0.5, 2.0), nodata=-9999.0)

    assert dtm.shape == shape
    # Basic sanity check that it runs without crashing


def test_extreme_terrain_vertical_cliffs():
    """
    Edge case: Vertical cliffs (step function).
    Verify that cliffs aren't smoothed out excessively.
    """
    shape = (100, 100)
    dsm = create_synthetic_array(shape, base_val=100.0)
    # Create a cliff: right half is 150m
    dsm[:, 50:] = 150.0

    dtm = dsm_to_dtm(dsm, resolution=(1.0, 1.0), slope=0.5, nodata=-9999.0)

    # The cliff edge might get smoothed, but the plateau should remain high.
    # Check "ground" on both sides far from edge
    assert abs(dtm[50, 10] - 100.0) < 1.0  # Left side (bottom)
    assert abs(dtm[50, 90] - 150.0) < 1.0  # Right side (top)


def test_extreme_terrain_perfectly_flat():
    """
    Edge case: Perfectly flat terrain with zero variance.
    """
    dsm = np.zeros((100, 100), dtype=np.float32)
    dtm = dsm_to_dtm(dsm, resolution=(1.0, 1.0), nodata=-9999.0)

    # Should remain exactly zero (or extremely close)
    assert np.allclose(dtm, 0.0, atol=1e-5)


def test_high_noise_environment():
    """
    Stress test with high frequency noise added to the DSM.
    """
    shape = (100, 100)
    dsm = create_synthetic_array(shape, base_val=100.0)
    add_noise(dsm, magnitude=2.0)  # High noise

    dtm = dsm_to_dtm(dsm, resolution=(1.0, 1.0), nodata=-9999.0)

    # DTM variance should be lower than DSM variance (smoothing effect)
    dsm_std = np.std(dsm)
    dtm_std = np.std(dtm)
    assert dtm_std < dsm_std


def test_dense_vegetation_simulation():
    """
    Simulate dense vegetation (high frequency objects above ground).
    """
    shape = (100, 100)
    dsm = create_synthetic_array(shape, base_val=100.0)
    add_vegetation(dsm, density=0.5, height_range=(5.0, 15.0))

    dtm = dsm_to_dtm(dsm, resolution=(1.0, 1.0), nodata=-9999.0)

    # Mean of DTM should be close to 100.0, significantly less than mean of DSM
    assert np.mean(dtm) < np.mean(dsm)
    # Should be close to bare earth
    assert abs(np.mean(dtm) - 100.0) < 1.0


def test_sparse_ground_points():
    """
    Edge case: Very few actual ground points visible (mostly buildings).
    This is hard for any algorithm. We check if it finds the 'floor'.
    """
    shape = (100, 100)
    # Start with "building" height everywhere
    dsm = np.full(shape, 120.0, dtype=np.float32)

    # Bore holes to ground (100.0)
    # Only 5% ground
    num_ground = int(shape[0] * shape[1] * 0.05)
    idx = np.random.choice(dsm.size, num_ground, replace=False)
    dsm.ravel()[idx] = 100.0

    # If window size is large enough to bridge the buildings, it should find ground.
    # Default radius is 40m. 100x100 is 100m. 5% ground is scattered.
    # PMF opens from small to large. It should eventually hit the ground points.

    dtm = dsm_to_dtm(dsm, resolution=(1.0, 1.0), kernel_radius_meters=50.0, nodata=-9999.0)

    # The resulting DTM should be closer to 100 than 120
    assert np.mean(dtm) < 110.0


def test_heavy_nodata_coverage():
    """
    Edge case: DSM with a high percentage (>50%) of nodata values.
    """
    shape = (100, 100)
    dsm = create_synthetic_array(shape, base_val=100.0)
    nodata = -9999.0

    # Mask 60%
    mask_idx = np.random.choice(dsm.size, int(dsm.size * 0.6), replace=False)
    dsm.ravel()[mask_idx] = nodata

    dtm = dsm_to_dtm(dsm, resolution=(1.0, 1.0), nodata=nodata)

    # Valid pixels in DTM should match or exceed valid pixels in DSM (gap filling)
    valid_dsm = np.sum(dsm != nodata)
    valid_dtm = np.sum(dtm != nodata)

    # Should have filled some gaps or at least preserved count
    assert valid_dtm >= valid_dsm


def test_checkerboard_nodata_pattern():
    """
    Edge case: Checkerboard pattern of valid/nodata.
    """
    shape = (100, 100)
    dsm = create_synthetic_array(shape, base_val=100.0)
    nodata = -9999.0

    # Create checkerboard
    checker = np.indices(shape).sum(axis=0) % 2
    dsm[checker == 1] = nodata

    # This effectively tests the gap filling and morphological operations on disjoint data
    dtm = dsm_to_dtm(dsm, resolution=(1.0, 1.0), nodata=nodata)

    # Ideally, gap filling fills the checkerboard holes
    # So DTM should be fully valid (or close to it)
    valid_fraction = np.mean(dtm != nodata)
    assert valid_fraction > 0.95


def test_outlier_spikes_and_pits():
    """
    Test handling of single-pixel extreme outliers.
    """
    shape = (50, 50)
    dsm = create_synthetic_array(shape, base_val=100.0)

    # Spike up
    dsm[25, 25] = 10000.0
    # Pit down (but not nodata)
    dsm[10, 10] = 0.0

    dtm = dsm_to_dtm(dsm, resolution=(1.0, 1.0), nodata=-9999.0)

    # Spike should be removed (PMF erodes it)
    assert dtm[25, 25] < 150.0

    # Pit might be preserved as "ground" unless it looks like noise/error?
    # PMF preserves local minima as ground. So the pit will likely remain.
    # Refinement/smoothing might soften it.
    # Let's just assert it doesn't crash or explode.
    assert dtm[10, 10] < 50.0


def test_geographic_reprojection_accuracy():
    """
    Stress test for reprojection. Placeholder since this requires CRS context.
    We skip for now as this is better tested in unit tests with real CRS objects.
    """
    pass


def test_memory_leak_check_repeated_runs():
    """
    Run the pipeline multiple times in a loop to check for memory leaks (rudimentary).
    """
    shape = (500, 500)
    dsm = create_synthetic_array(shape)
    for _ in range(5):
        _ = dsm_to_dtm(dsm, resolution=(1.0, 1.0), nodata=-9999.0)
    # If we got here without OOM or crash, it's a good sign.
