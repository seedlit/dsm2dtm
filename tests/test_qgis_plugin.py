import os
import sys

import numpy as np

# Add the qgis_plugin to sys.path to test its vendored version
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "qgis_plugin")))


def test_qgis_algorithm_no_rasterio():
    """Verify that the plugin's algorithm can be imported and run without rasterio."""
    # Temporarily hide rasterio to simulate QGIS environment
    import sys

    original_rasterio = sys.modules.pop("rasterio", None)
    sys.modules["rasterio"] = None

    try:
        from dsm2dtm.ext_libs.dsm2dtm_core.algorithm import dsm_to_dtm

        # Run a small test to ensure it works
        dsm = np.full((50, 50), 100.0, dtype=np.float32)
        dsm[20:30, 20:30] = 120.0  # building
        resolution = (1.0, 1.0)

        dtm = dsm_to_dtm(dsm, resolution, kernel_radius_meters=10.0, slope=0.1)

        assert dtm[25, 25] < 110.0
        assert abs(dtm[0, 0] - 100.0) < 0.1
    finally:
        if original_rasterio is not None:
            sys.modules["rasterio"] = original_rasterio
        else:
            sys.modules.pop("rasterio", None)
