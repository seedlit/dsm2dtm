import tempfile
from pathlib import Path

import numpy as np
import pytest
import rasterio as rio

from src import dsm2dtm

TEST_DSM1 = "data/sample_dsm.tif"
# TEST_DSM1 = "/Users/naman.jain/Desktop/personal/side_projects/temp/temp/downsampled_dsm.tif"
TEST_DSM2 = "data/sample_hiilside_dsm_30cm.tif"
# TODO: add tests for .sdat files


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
        (2.513751187083714e-07, 2.51398813677825e-07, 4326, 0.1),
        (0.30014781965999754, 0.29999999999989213, 32610, 1),
    ],
)
def test_get_downsampling_factor(x_res, y_res, crs, expected_result):
    downsampling_factor = dsm2dtm.get_downsampling_factor(x_res, y_res, crs)
    assert downsampling_factor == expected_result


def test_resample_raster():
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


@pytest.mark.parametrize(
    "dsm_path, expected_height, expected_width, expected_mean_value",
    [
        (TEST_DSM1, 2866, 3159, 0.0980370044708252),
        (TEST_DSM2, 1726, 1353, 3.7407186031341553),
    ],
)
def test_generate_slope_array(
    dsm_path, expected_height, expected_width, expected_mean_value
):
    slope_array = dsm2dtm.generate_slope_array(dsm_path)
    assert slope_array.shape == (expected_height, expected_width)
    assert slope_array.mean().item() == expected_mean_value


def test_subtract_rasters():
    with tempfile.TemporaryDirectory() as tmpdir:
        subtracted_raster = dsm2dtm.subtract_rasters(
            TEST_DSM1, TEST_DSM1, f"{tmpdir}/subtracted.tif"
        )
        assert Path(subtracted_raster).is_file()
        with rio.open(subtracted_raster) as raster:
            assert raster.read().mean() == 0


def test_replace_values():
    new_array = dsm2dtm.replace_values(TEST_DSM1, TEST_DSM1, 0, 60)
    assert new_array.shape == (2866, 3159)
    assert new_array[1256, 786] == 0
    assert new_array.mean() == 24.193619974344173


def test_remove_noise():
    with rio.open(TEST_DSM1) as raster:
        test_array = raster.read()
        new_array = np.squeeze(dsm2dtm.remove_noise(test_array))
        assert new_array.shape == (2866, 3159)
        assert float(new_array.mean()) == -23043.646484375


def test_expand_holes_in_array():
    with rio.open(TEST_DSM1) as raster:
        test_array = raster.read()
        new_array = np.squeeze(dsm2dtm.expand_holes_in_array(test_array))
        assert new_array.shape == (2866, 3159)
        assert float(new_array.mean()) == 63.69542694091797


def test_extract_dtm():
    # TODO: local workaround
    # export PATH=$PATH:/Applications/SAGA.app/Contents/MacOS
    # export PATH=$PATH:.
    with tempfile.TemporaryDirectory() as tmpdir:
        out_ground_path = f"{tmpdir}/ground.tif"
        dsm2dtm.extract_dtm(
            dsm_path=TEST_DSM1,
            out_ground_path=out_ground_path,
            radius=5,
            terrain_slope=5,
        )
        assert Path(out_ground_path).is_file()
        with rio.open(out_ground_path) as ground:
            ground_array = ground.read()
            assert ground_array.shape == (1, 2866, 3159)
            assert float(ground_array.mean()) == -21370.341796875


def test_array_to_geotiff():
    with tempfile.TemporaryDirectory() as tmpdir:
        with rio.open(TEST_DSM1) as src:
            src_array = np.squeeze(src.read())
            out_path = f"{tmpdir}/new_tif.tif"
            dsm2dtm.array_to_geotif(
                array=src_array, ref_tif_path=TEST_DSM1, out_path=out_path
            )
            assert Path(out_path).is_file()
            with rio.open(out_path) as out:
                out_array = out.read()
                assert out_array.shape == (1, 2866, 3159)
                assert float(out_array.mean()) == 63.69542694091797


def test_close_gaps():
    with tempfile.TemporaryDirectory() as tmpdir:
        out_path = f"{tmpdir}/closed.tif"
        dsm2dtm.close_gaps(
            in_path=TEST_DSM1,
            out_path=out_path,
        )
        assert Path(out_path).is_file()
        with rio.open(out_path) as out:
            out_array = out.read()
            assert out_array.shape == (1, 2866, 3159)
            assert float(out_array.mean()) == 63.69542694091797


def test_smoothen_raster():
    with tempfile.TemporaryDirectory() as tmpdir:
        out_path = f"{tmpdir}/smoothened.tif"
        dsm2dtm.smoothen_raster(
            in_path=TEST_DSM1,
            out_path=out_path,
        )
        assert Path(out_path).is_file()
        with rio.open(out_path) as out:
            out_array = out.read()
            assert out_array.shape == (1, 2866, 3159)
            assert float(out_array.mean()) == 63.69527053833008


def test_main():
    # TODO: run memeory profiler
    with tempfile.TemporaryDirectory() as tmpdir:
        dtm_path = dsm2dtm.main(TEST_DSM1, tmpdir, search_radius=5, smoothen_radius=5)
        assert Path(dtm_path).is_file()
        with rio.open(dtm_path) as dtm:
            dtm_array = dtm.read()
            assert dtm_array.shape == (1, 286, 315)
            assert float(dtm_array.mean()) == 59.982666015625
