import os

import pytest
import rasterio
from dsm2dtm import core


@pytest.fixture(scope="module")
def dsm_path():
    path = "data/test_dsm.tif"
    if not os.path.exists(path):
        pytest.fail(f"Test data not found at {path}")
    return path


def test_dsm_to_dtm_array(dsm_path):
    """Test the core dsm_to_dtm function with array input."""
    with rasterio.open(dsm_path) as src:
        dsm = src.read(1)
        resolution = src.res
        nodata = src.nodata if src.nodata is not None else -99999.0

    dtm = core.dsm_to_dtm(dsm, resolution, nodata=nodata)

    assert dtm.shape == dsm.shape
    # DTM should be <= DSM (ground is below surface)
    valid_mask = (dsm != nodata) & (dtm != nodata)
    assert dtm[valid_mask].mean() <= dsm[valid_mask].mean()


def test_main_function(dsm_path, tmp_path):
    """Test the main entry point function end-to-end."""
    dtm_path = core.main(dsm_path, str(tmp_path))

    assert os.path.isfile(dtm_path)
    assert dtm_path.endswith(".tif")

    with rasterio.open(dtm_path) as src:
        assert src.count == 1
        dtm_array = src.read(1)
        # Verify DTM has reasonable dimensions and data
        assert dtm_array.shape[0] > 0
        assert dtm_array.shape[1] > 0
