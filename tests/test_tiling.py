import numpy as np
import pytest

from dsm2dtm import algorithm, tiling


def _urban_scene(shape=(900, 700), seed=0):
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0 : shape[0], 0 : shape[1]]
    dsm = (100 + 0.05 * xx + 3 * np.sin(yy / 90)).astype(np.float32)
    for _ in range(60):
        r, c = rng.integers(0, shape[0] - 40), rng.integers(0, shape[1] - 40)
        size = rng.integers(8, 40)
        dsm[r : r + size, c : c + size] += rng.uniform(4, 20)
    dsm[:, :25] = -9999.0
    dsm[400:430, 300:360] = -9999.0
    return dsm


def _run_tiled(dsm, resolution, tile_size, workers=4, **kwargs):
    out = np.full(dsm.shape, np.nan, dtype=np.float32)

    def read(window):
        return dsm[window.row_off : window.row_off + window.height, window.col_off : window.col_off + window.width]

    def write(window, block):
        out[window.row_off : window.row_off + window.height, window.col_off : window.col_off + window.width] = block

    tiling.process_tiled(
        read, write, dsm.shape, resolution, nodata=-9999.0, tile_size=tile_size, workers=workers, **kwargs
    )
    return out


def test_plan_tiles_covers_raster_exactly_once():
    tiles = tiling.plan_tiles((1000, 700), tile_size=256, halo=50)
    coverage = np.zeros((1000, 700), dtype=int)
    for t in tiles:
        w = t.core
        coverage[w.row_off : w.row_off + w.height, w.col_off : w.col_off + w.width] += 1
        assert t.read.row_off >= 0 and t.read.col_off >= 0
        assert t.read.row_off + t.read.height <= 1000 and t.read.col_off + t.read.width <= 700
    assert np.all(coverage == 1)


def test_halo_grows_with_radius_and_finer_resolution():
    assert tiling.halo_pixels(1.0, 80.0) > tiling.halo_pixels(1.0, 40.0)
    assert tiling.halo_pixels(0.5, 40.0) == pytest.approx(2 * tiling.halo_pixels(1.0, 40.0), abs=1)


def test_tiled_matches_untiled():
    dsm = _urban_scene()
    slope = algorithm.calculate_terrain_slope(dsm, 1.0, -9999.0)
    full = algorithm.dsm_to_dtm(dsm, (1.0, 1.0), kernel_radius_meters=20.0, slope=slope, nodata=-9999.0)
    tiled = _run_tiled(dsm, (1.0, 1.0), tile_size=256, kernel_radius_meters=20.0, slope=slope)
    assert not np.any(np.isnan(tiled))
    np.testing.assert_allclose(tiled, full, atol=1e-4)


def test_tiled_auto_slope_close_to_global_estimate():
    dsm = _urban_scene()
    tiled_slope = tiling.estimate_slope_tiled(
        lambda w: dsm[w.row_off : w.row_off + w.height, w.col_off : w.col_off + w.width],
        dsm.shape,
        1.0,
        -9999.0,
        tile_size=256,
    )
    assert tiled_slope == pytest.approx(algorithm.calculate_terrain_slope(dsm, 1.0, -9999.0), rel=0.05)


def test_single_tile_when_raster_is_small():
    assert len(tiling.plan_tiles((300, 200), tile_size=2048, halo=100)) == 1


def test_progress_and_cancel():
    dsm = _urban_scene()
    seen = []
    _run_tiled(dsm, (1.0, 1.0), tile_size=256, workers=1, slope=0.1, progress=lambda done, total: seen.append(done))
    assert seen[-1] == len(tiling.plan_tiles(dsm.shape, 256, tiling.halo_pixels(1.0, None)))

    with pytest.raises(tiling.TilingCancelled):
        _run_tiled(dsm, (1.0, 1.0), tile_size=256, workers=1, slope=0.1, is_cancelled=lambda: True)


def test_tiled_close_to_untiled_at_fine_resolution():
    """Below 0.5 m each tile is resampled on its own grid, so seams may differ slightly."""
    dsm = _urban_scene((1200, 1000))
    slope = algorithm.calculate_terrain_slope(dsm, 0.25, -9999.0)
    full = algorithm.dsm_to_dtm(dsm, (0.25, 0.25), kernel_radius_meters=10.0, slope=slope, nodata=-9999.0)
    tiled = _run_tiled(dsm, (0.25, 0.25), tile_size=256, kernel_radius_meters=10.0, slope=slope)
    np.testing.assert_array_equal(full == -9999.0, tiled == -9999.0)
    diff = np.abs(full - tiled)[full != -9999.0]
    assert np.percentile(diff, 99) < 0.01
    assert np.mean(diff > 0.05) < 0.005
