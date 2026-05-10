"""Tests for the QGIS plugin's vendored core.

Vendored modules MUST stay rasterio-free — QGIS does not ship rasterio. We
prove this by importing fresh after blocking `rasterio` in `sys.modules`.
"""

import importlib
import os
import sys

import numpy as np
import pytest

PLUGIN_EXT_LIBS = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "qgis_plugin", "dsm2dtm", "ext_libs"))


def _purge_vendored_imports() -> None:
    for name in list(sys.modules):
        if name.startswith("dsm2dtm_core"):
            del sys.modules[name]


@pytest.fixture
def vendored_no_rasterio(monkeypatch):
    """Hide rasterio + force a fresh import of the vendored modules."""
    _purge_vendored_imports()
    monkeypatch.setitem(sys.modules, "rasterio", None)
    if PLUGIN_EXT_LIBS not in sys.path:
        sys.path.insert(0, PLUGIN_EXT_LIBS)
    yield
    _purge_vendored_imports()


def test_vendored_algorithm_imports_without_rasterio(vendored_no_rasterio):
    """Importing the vendored algorithm must not pull in rasterio."""
    importlib.import_module("dsm2dtm_core.algorithm")
    importlib.import_module("dsm2dtm_core.constants")
    importlib.import_module("dsm2dtm_core.utm_utils")
    assert sys.modules["rasterio"] is None


def test_vendored_algorithm_runs_without_rasterio(vendored_no_rasterio):
    """The vendored DSM→DTM pipeline must produce a sensible DTM in a rasterio-free env."""
    from dsm2dtm_core.algorithm import dsm_to_dtm

    dsm = np.full((50, 50), 100.0, dtype=np.float32)
    dsm[20:30, 20:30] = 120.0  # 10x10 building
    dtm = dsm_to_dtm(dsm, (1.0, 1.0), kernel_radius_meters=10.0, slope=0.1)

    assert dtm[25, 25] < 110.0, "Building was not removed"
    assert abs(dtm[0, 0] - 100.0) < 0.1, "Flat ground drifted"


def test_vendored_utm_lookup_matches_library(vendored_no_rasterio):
    """Vendored utm_utils must agree with the library on real-world coordinates."""
    from dsm2dtm_core.utm_utils import estimate_utm_crs as vendored_lookup

    from dsm2dtm.utm_utils import estimate_utm_crs as library_lookup

    for lon, lat in [(-74.0, 40.7), (151.2, -33.8), (5.3, 60.4), (15.6, 78.2), (179.9, 20.0)]:
        assert vendored_lookup(lon, lat) == library_lookup(lon, lat), f"Drift at ({lon}, {lat})"
