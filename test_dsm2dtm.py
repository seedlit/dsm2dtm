import os
import pytest
import rasterio
from src.dsm2dtm import core as dsm2dtm


@pytest.fixture(scope="module")
def dsm_path():
    path = "data/test_dsm.tif"
    if not os.path.exists(path):
        pytest.fail(f"Test data not found at {path}")
    return path


@pytest.fixture(scope="module")
def shared_temp_dir(tmp_path_factory):
    """
    Creates a temporary directory for tests sharing intermediate files.
    """
    return tmp_path_factory.mktemp("dsm2dtm_shared")


@pytest.fixture(scope="module")
def dsm_ds_path(dsm_path, shared_temp_dir):
    """Fixture to provide the downsampled DSM path, as it's used by subsequent tests."""
    # Convert Path object to string for core logic
    return dsm2dtm.get_res_and_downsample(dsm_path, str(shared_temp_dir))


def test_downsampling(shared_temp_dir, dsm_ds_path, dsm_path):
    dsm_name = os.path.basename(dsm_path).split(".")[0]
    expected_path = shared_temp_dir / f"{dsm_name}_ds.tif"
    assert dsm_ds_path == str(expected_path)
    assert expected_path.is_file()


def test_get_updated_params(dsm_ds_path):
    assert dsm2dtm.get_updated_params(dsm_ds_path, 40, 45) == (40, 45)


def test_generate_slope_raster(shared_temp_dir, dsm_ds_path, dsm_path):
    dsm_name = os.path.basename(dsm_path).split(".")[0]
    dsm_slp_path = shared_temp_dir / f"{dsm_name}_slp.tif"
    dsm2dtm.generate_slope_raster(dsm_ds_path, str(dsm_slp_path))
    assert dsm_slp_path.is_file()
    assert int(dsm2dtm.get_mean(str(dsm_slp_path))) == 89


def test_extract_dtm(shared_temp_dir, dsm_ds_path, dsm_path):
    dsm_name = os.path.basename(dsm_path).split(".")[0]
    ground_dem_path = shared_temp_dir / f"{dsm_name}_ground.tif"
    non_ground_dem_path = shared_temp_dir / f"{dsm_name}_non_ground.tif"
    
    dsm2dtm.extract_dtm(dsm_ds_path, str(ground_dem_path), str(non_ground_dem_path), 40, 13)
    assert ground_dem_path.is_file()
    assert non_ground_dem_path.is_file()


def test_pipeline_integration(shared_temp_dir, dsm_ds_path, dsm_path):
    """
    Runs the rest of the pipeline steps that depend heavily on each other's outputs.
    """
    dsm_name = os.path.basename(dsm_path).split(".")[0]
    
    ground_dem_path = shared_temp_dir / f"{dsm_name}_ground.tif"
    
    # Ensure extract_dtm was run
    non_ground_dem_path = shared_temp_dir / f"{dsm_name}_non_ground.tif"
    dsm2dtm.extract_dtm(dsm_ds_path, str(ground_dem_path), str(non_ground_dem_path), 40, 13)
    
    # Smoothen
    smoothened_ground_path = shared_temp_dir / f"{dsm_name}_ground_smth.tif"
    dsm2dtm.smoothen_raster(str(ground_dem_path), str(smoothened_ground_path), 45)
    assert smoothened_ground_path.is_file()

    # Subtract
    diff_raster_path = shared_temp_dir / f"{dsm_name}_ground_diff.tif"
    dsm2dtm.subtract_rasters(str(ground_dem_path), str(smoothened_ground_path), str(diff_raster_path))
    assert diff_raster_path.is_file()

    # Replace values
    thresholded_ground_path = shared_temp_dir / f"{dsm_name}_ground_thresholded.tif"
    dsm2dtm.replace_values(str(ground_dem_path), str(diff_raster_path), str(thresholded_ground_path), 0.98)
    assert thresholded_ground_path.is_file()

    # Remove noise
    ground_dem_filtered_path = shared_temp_dir / f"{dsm_name}_ground_filtered.tif"
    dsm2dtm.remove_noise(str(thresholded_ground_path), str(ground_dem_filtered_path))
    assert ground_dem_filtered_path.is_file()

    # Expand holes
    bigger_holes_ground_path = shared_temp_dir / f"{dsm_name}_ground_bigger_holes.tif"
    temp_array = dsm2dtm.expand_holes_in_raster(str(ground_dem_filtered_path))
    
    dsm2dtm.save_array_as_geotif(temp_array, str(ground_dem_filtered_path), str(bigger_holes_ground_path))
    assert bigger_holes_ground_path.is_file()

    # Close gaps
    dtm_path = shared_temp_dir / f"{dsm_name}_dtm.tif"
    dsm2dtm.close_gaps(str(bigger_holes_ground_path), str(dtm_path))
    assert dtm_path.is_file()

    # Verification
    with rasterio.open(dtm_path) as src:
        dtm_array = src.read(1)
        
    assert dtm_array.shape == (56, 69)
    res = dsm2dtm.get_raster_resolution(str(dtm_path))
    assert abs(res[0] - 2.5193e-06) < 1e-6
    assert abs(res[1] - 2.5193e-06) < 1e-6
    assert dsm2dtm.get_raster_crs(str(dtm_path)) == 4326


def test_main_function(dsm_path, tmp_path):
    """Test the main entry point function end-to-end"""
    # tmp_path is a unique temporary directory for this function invocation provided by pytest
    dtm_path = dsm2dtm.main(dsm_path, str(tmp_path))
    
    assert os.path.isfile(dtm_path)
    assert dtm_path.endswith(".tif")
    
    with rasterio.open(dtm_path) as src:
        assert src.count == 1
        assert src.read(1).shape == (56, 69)
