import tempfile
from pathlib import Path
import pytest
import rasterio as rio
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


def test_downsample_raster():
    with tempfile.TemporaryDirectory() as tmpdir:
        resampled_dsm_path = dsm2dtm.resample_raster(
            TEST_DSM1, f"{tmpdir}/downsampled_dsm.tif", 0.1
        )
        assert Path(resampled_dsm_path).is_file()
        with rio.open(resampled_dsm_path) as raster:
            assert raster.transform[0] == 2.5209333333325248e-06
            assert raster.transform[4] == -2.5192622377644984e-06


@pytest.mark.parametrize(
    "x_res, y_res, crs, search_radius, smoothen_radius, expected_search_radius, expected_smoothen_radius",
    [
        (2.513751187083714e-07, 2.51398813677825e-07, 4326, 40, 45, 40, 45),
        (0.30014781965999754, 0.29999999999989213, 32610, 40, 45, 39, 44),
    ],
)
def test_get_updated_params(
    x_res,
    y_res,
    crs,
    search_radius,
    smoothen_radius,
    expected_search_radius,
    expected_smoothen_radius,
):
    assert dsm2dtm.get_updated_params(
        x_res, y_res, crs, search_radius, smoothen_radius
    ) == (expected_search_radius, expected_smoothen_radius)
