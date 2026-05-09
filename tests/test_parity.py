"""CLI vs QGIS-plugin parity test.

The library (`src/dsm2dtm/`) and the vendored plugin core
(`qgis_plugin/dsm2dtm/ext_libs/dsm2dtm_core/`) MUST produce numerically
equivalent DTMs from the same input. They have diverged before
(BUG-43); this test is the safety net that catches future drift.
"""

import importlib
import os
import sys

import numpy as np
import pytest
import rasterio

PLUGIN_EXT_LIBS = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "qgis_plugin", "dsm2dtm", "ext_libs"))


@pytest.fixture(scope="module")
def cli_dsm_to_dtm():
    from dsm2dtm.algorithm import dsm_to_dtm

    return dsm_to_dtm


@pytest.fixture(scope="module")
def plugin_dsm_to_dtm():
    if PLUGIN_EXT_LIBS not in sys.path:
        sys.path.insert(0, PLUGIN_EXT_LIBS)
    if "dsm2dtm_core" in sys.modules:
        importlib.reload(sys.modules["dsm2dtm_core.algorithm"])
    from dsm2dtm_core.algorithm import dsm_to_dtm

    return dsm_to_dtm


def _synthetic_dsm(seed: int = 42) -> np.ndarray:
    rng = np.random.default_rng(seed)
    rows, cols = 200, 200
    yy, xx = np.mgrid[0:rows, 0:cols]
    terrain = 100.0 + 0.05 * yy + 0.03 * xx
    terrain += rng.normal(0.0, 0.2, size=terrain.shape)
    dsm = terrain.astype(np.float32)
    dsm[40:70, 40:70] += 18.0
    dsm[120:135, 120:135] += 12.0
    dsm[160:175, 30:55] += 8.0
    return dsm


def _assert_parity(a: np.ndarray, b: np.ndarray, label: str) -> None:
    assert a.shape == b.shape, f"{label}: shape mismatch {a.shape} vs {b.shape}"
    diff = a - b
    finite = np.isfinite(diff)
    assert finite.any(), f"{label}: no finite values to compare"
    max_abs = float(np.max(np.abs(diff[finite])))
    if max_abs > 1e-3:
        rms = float(np.sqrt(np.mean(diff[finite] ** 2)))
        pct_off = float(np.mean(np.abs(diff[finite]) > 1e-3) * 100.0)
        pytest.fail(
            f"{label}: CLI vs plugin diverge — max|Δ|={max_abs:.4f}m, "
            f"rms={rms:.4f}m, {pct_off:.2f}% of cells differ by >1mm"
        )


def test_parity_synthetic_smooth(cli_dsm_to_dtm, plugin_dsm_to_dtm):
    dsm = _synthetic_dsm()
    resolution = (1.0, 1.0)
    cli_dtm = cli_dsm_to_dtm(dsm, resolution, kernel_radius_meters=20.0, slope=0.1)
    plugin_dtm = plugin_dsm_to_dtm(dsm, resolution, kernel_radius_meters=20.0, slope=0.1)
    _assert_parity(cli_dtm, plugin_dtm, "synthetic-smooth")


@pytest.mark.xfail(strict=True, reason="BUG-43: vendored plugin core diverges from src/ — flip xfail off when unified")
def test_parity_synthetic_auto_slope(cli_dsm_to_dtm, plugin_dsm_to_dtm):
    dsm = _synthetic_dsm(seed=7)
    resolution = (1.0, 1.0)
    cli_dtm = cli_dsm_to_dtm(dsm, resolution, kernel_radius_meters=15.0)
    plugin_dtm = plugin_dsm_to_dtm(dsm, resolution, kernel_radius_meters=15.0)
    _assert_parity(cli_dtm, plugin_dtm, "synthetic-auto-slope")


@pytest.mark.xfail(strict=True, reason="BUG-43: vendored plugin core diverges from src/ — flip xfail off when unified")
def test_parity_with_nodata(cli_dsm_to_dtm, plugin_dsm_to_dtm):
    dsm = _synthetic_dsm(seed=11)
    dsm[80:90, :] = -9999.0
    dsm[:, 150:155] = -9999.0
    resolution = (1.0, 1.0)
    cli_dtm = cli_dsm_to_dtm(dsm, resolution, kernel_radius_meters=20.0, slope=0.1, nodata=-9999.0)
    plugin_dtm = plugin_dsm_to_dtm(dsm, resolution, kernel_radius_meters=20.0, slope=0.1, nodata=-9999.0)
    valid = (cli_dtm != -9999.0) & (plugin_dtm != -9999.0)
    _assert_parity(cli_dtm[valid], plugin_dtm[valid], "with-nodata (valid cells)")


@pytest.mark.xfail(strict=True, reason="BUG-43: vendored plugin core diverges from src/ — flip xfail off when unified")
def test_parity_release_fixture(cli_dsm_to_dtm, plugin_dsm_to_dtm, test_data_dir):
    """Both implementations must match on a real DSM tile from the release fixtures."""
    dsm_path = test_data_dir / "dsm_1m_istanbul_hilly_urban.tif"
    if not dsm_path.exists():
        pytest.skip(f"fixture not available: {dsm_path}")

    with rasterio.open(dsm_path) as src:
        full = src.read(1)
        nodata = src.nodata if src.nodata is not None else -9999.0
        x_res = abs(src.transform.a)
        y_res = abs(src.transform.e)

    crop = full[:300, :300].astype(np.float32)
    resolution = (x_res, y_res)
    cli_dtm = cli_dsm_to_dtm(crop, resolution, kernel_radius_meters=40.0, slope=0.0, nodata=float(nodata))
    plugin_dtm = plugin_dsm_to_dtm(crop, resolution, kernel_radius_meters=40.0, slope=0.0, nodata=float(nodata))
    valid = (cli_dtm != float(nodata)) & (plugin_dtm != float(nodata))
    _assert_parity(cli_dtm[valid], plugin_dtm[valid], "release-fixture-istanbul")
