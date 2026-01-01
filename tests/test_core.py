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
        resolution = src.res
        nodata = src.nodata if src.nodata is not None else -99999.0

    dtm_path = core.main(dsm_path, "temp_data")
    with rasterio.open(dtm_path) as src:
        dtm = src.read(1)

    assert np.mean(dtm) <= np.mean(dsm)
    assert np.max(dtm) <= np.max(dsm)
    assert round(np.mean(dtm),1) == 58.8
    assert round(np.max(dtm),1) == 60.6
    assert round(np.min(dtm),1) == 0.0
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
