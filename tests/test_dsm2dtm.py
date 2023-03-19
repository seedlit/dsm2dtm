import pytest
from src import dsm2dtm


TEST_DSM1 = "data/sample_dsm.tif"
TEST_DSM2 = "data/sample_hiilside_dsm_30cm.tif"


@pytest.mark.parametrize(
    "test_dsm, expected_x_res, expected_y_res",
    [
        (TEST_DSM1, 2.513751187083714e-07, 2.51398813677825e-07),
        (TEST_DSM2, 0.30014781965999754, 0.29999999999989213),
    ],
)
def test_get_raster_resolution(test_dsm, expected_x_res, expected_y_res):
    x_res, y_res = dsm2dtm.get_raster_resolution(test_dsm)
    assert x_res == expected_x_res
    assert y_res == expected_y_res


@pytest.mark.parametrize(
    "test_dsm, expected_epsg",
    [
        (TEST_DSM1, 4326),
        (TEST_DSM2, 32610),
    ],
)
def test_get_raster_crs(test_dsm, expected_epsg):
    epsg = dsm2dtm.get_raster_crs(test_dsm)
    assert epsg == expected_epsg


@pytest.mark.parametrize(
    "x_res, y_res, crs, expected_result",
    [
        (2.513751187083714e-07, 2.51398813677825e-07, 4326, 10.000989807255246),
        (0.30014781965999754, 0.29999999999989213, 32610, 1),
    ],
)
def test_get_downsampling_factor(x_res, y_res, crs, expected_result):
    downsampling_factor = dsm2dtm.get_downsampling_factor(x_res, y_res, crs)
    assert downsampling_factor == expected_result
