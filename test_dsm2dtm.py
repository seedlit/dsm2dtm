import os
import shutil

import pytest
import rasterio

from src.dsm2dtm import core as dsm2dtm

DSM_PATH = "data/sample_dsm.tif"


@pytest.fixture(scope="module")
def temp_dir():
    out_dir = "test_results"
    tmp_dir = os.path.join(out_dir, "temp_files")
    os.makedirs(tmp_dir, exist_ok=True)
    yield tmp_dir
    # Cleanup after tests
    if os.path.exists(out_dir):
        shutil.rmtree(out_dir)


@pytest.fixture(scope="module")
def dsm_ds_path(temp_dir):
    """Fixture to provide the downsampled DSM path, as it's used by subsequent tests."""
    return dsm2dtm.get_res_and_downsample(DSM_PATH, temp_dir)


def test_downsampling(temp_dir, dsm_ds_path):
    dsm_name = DSM_PATH.split("/")[-1].split(".")[0]
    expected_path = os.path.join(temp_dir, "{}_ds.tif".format(dsm_name))
    assert dsm_ds_path == expected_path
    assert os.path.isfile(expected_path)


def test_get_updated_params(dsm_ds_path):
    assert dsm2dtm.get_updated_params(dsm_ds_path, 40, 45) == (40, 45)


def test_generate_slope_raster(temp_dir, dsm_ds_path):
    dsm_name = DSM_PATH.split("/")[-1].split(".")[0]
    dsm_slp_path = os.path.join(temp_dir, dsm_name + "_slp.tif")
    dsm2dtm.generate_slope_raster(dsm_ds_path, dsm_slp_path)
    assert os.path.isfile(dsm_slp_path)
    assert int(dsm2dtm.get_mean(dsm_slp_path)) == 89


def test_extract_dtm(temp_dir, dsm_ds_path):
    dsm_name = DSM_PATH.split("/")[-1].split(".")[0]
    ground_dem_path = os.path.join(temp_dir, dsm_name + "_ground.tif")  # Expect .tif
    non_ground_dem_path = os.path.join(temp_dir, dsm_name + "_non_ground.tif")  # Expect .tif
    # Slope is hardcoded to 89 based on original test expectation, may need adjustment with new algorithm
    dsm2dtm.extract_dtm(dsm_ds_path, ground_dem_path, non_ground_dem_path, 40, 13)
    assert os.path.isfile(ground_dem_path)
    assert os.path.isfile(non_ground_dem_path)


def test_pipeline_integration(temp_dir, dsm_ds_path):
    """
    Runs the rest of the pipeline steps that depend heavily on each other's outputs.
    """
    dsm_name = DSM_PATH.split("/")[-1].split(".")[0]
    ground_dem_path = os.path.join(temp_dir, dsm_name + "_ground.tif")
    non_ground_dem_path = os.path.join(temp_dir, dsm_name + "_non_ground.tif")
    dsm2dtm.extract_dtm(dsm_ds_path, ground_dem_path, non_ground_dem_path, 40, 13)

    # Ensure previous step created the file (redundant but safe)
    assert os.path.isfile(ground_dem_path)

    # Smoothen
    smoothened_ground_path = os.path.join(temp_dir, dsm_name + "_ground_smth.tif")
    dsm2dtm.smoothen_raster(ground_dem_path, smoothened_ground_path, 45)
    assert os.path.isfile(smoothened_ground_path)

    # Subtract
    diff_raster_path = os.path.join(temp_dir, dsm_name + "_ground_diff.tif")
    dsm2dtm.subtract_rasters(ground_dem_path, smoothened_ground_path, diff_raster_path)
    assert os.path.isfile(diff_raster_path)

    # Replace values
    thresholded_ground_path = os.path.join(temp_dir, dsm_name + "_ground_thresholded.tif")
    dsm2dtm.replace_values(ground_dem_path, diff_raster_path, thresholded_ground_path, 0.98)
    assert os.path.isfile(thresholded_ground_path)

    # Remove noise
    ground_dem_filtered_path = os.path.join(temp_dir, dsm_name + "_ground_filtered.tif")
    dsm2dtm.remove_noise(thresholded_ground_path, ground_dem_filtered_path)
    assert os.path.isfile(ground_dem_filtered_path)

    # Expand holes
    bigger_holes_ground_path = os.path.join(temp_dir, dsm_name + "_ground_bigger_holes.tif")
    temp_array = dsm2dtm.expand_holes_in_raster(ground_dem_filtered_path)
    assert temp_array.shape == (286, 315)

    dsm2dtm.save_array_as_geotif(temp_array, ground_dem_filtered_path, bigger_holes_ground_path)
    assert os.path.isfile(bigger_holes_ground_path)

    # Close gaps
    dtm_path = os.path.join(temp_dir, dsm_name + "_dtm.tif")
    dsm2dtm.close_gaps(bigger_holes_ground_path, dtm_path)
    assert os.path.isfile(dtm_path)

    # Final verification
    # For this test, the output is directly dtm_path, not a separate dtm_tif_path from sdat_to_gtiff
    # No separate sdat_to_gtiff call or assertion needed here.

    with rasterio.open(dtm_path) as src:
        dtm_array = src.read(1)
    assert dtm_array.shape == (286, 315)
    res = dsm2dtm.get_raster_resolution(dtm_path)
    # Allow small float tolerance
    assert abs(res[0] - 2.5193e-06) < 1e-6
    assert abs(res[1] - 2.5193e-06) < 1e-6
    assert dsm2dtm.get_raster_crs(dtm_path) == 4326


def test_main_function():
    """Test the main entry point function end-to-end"""
    out_dir = "test_results_main"
    if os.path.exists(out_dir):
        shutil.rmtree(out_dir)

    try:
        dtm_path = dsm2dtm.main(DSM_PATH, out_dir)
        assert os.path.isfile(dtm_path)
        assert dtm_path.endswith(".tif")
        # Verify it's a valid raster
        with rasterio.open(dtm_path) as src:
            assert src.count == 1
            assert src.read(1).shape == (286, 315)
    finally:
        if os.path.exists(out_dir):
            shutil.rmtree(out_dir)
