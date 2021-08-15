import os
from dsm2dtm import dsm2dtm
import shutil
import gdal


if __name__ == "__main__":

    dsm_path = "data/sample_dsm.tif"
    out_dir = "test_results"

    dsm_name = dsm_path.split("/")[-1].split(".")[0]
    os.makedirs(out_dir, exist_ok=True)
    temp_dir = os.path.join(out_dir, "temp_files")
    os.makedirs(temp_dir, exist_ok=True)

    # test each function
    dsm_path = dsm2dtm.get_res_and_downsample(dsm_path, temp_dir)
    assert dsm_path == os.path.join(temp_dir, "{}_ds.tif".format(dsm_name))
    assert dsm2dtm.get_updated_params(dsm_path, 40, 45) == (40, 45)

    dsm_slp_path = os.path.join(temp_dir, dsm_name + "_slp.tif")
    dsm2dtm.generate_slope_raster(dsm_path, dsm_slp_path)
    assert os.path.isfile(dsm_slp_path)
    assert int(dsm2dtm.get_mean(dsm_slp_path)) == 89

    ground_dem_path = os.path.join(temp_dir, dsm_name + "_ground.sdat")
    non_ground_dem_path = os.path.join(temp_dir, dsm_name + "_non_ground.sdat")
    dsm2dtm.extract_dtm(dsm_path, ground_dem_path, non_ground_dem_path, 40, 89)
    assert os.path.isfile(ground_dem_path)
    assert os.path.isfile(non_ground_dem_path)

    smoothened_ground_path = os.path.join(temp_dir, dsm_name + "_ground_smth.sdat")
    dsm2dtm.smoothen_raster(ground_dem_path, smoothened_ground_path, 45)
    assert os.path.isfile(smoothened_ground_path)

    diff_raster_path = os.path.join(temp_dir, dsm_name + "_ground_diff.sdat")
    dsm2dtm.subtract_rasters(ground_dem_path, smoothened_ground_path, diff_raster_path)
    assert os.path.isfile(diff_raster_path)

    thresholded_ground_path = os.path.join(
        temp_dir, dsm_name + "_ground_thresholded.sdat"
    )
    dsm2dtm.replace_values(
        ground_dem_path, diff_raster_path, thresholded_ground_path, 0.98
    )
    assert os.path.isfile(thresholded_ground_path)

    ground_dem_filtered_path = os.path.join(temp_dir, dsm_name + "_ground_filtered.tif")
    dsm2dtm.remove_noise(thresholded_ground_path, ground_dem_filtered_path)
    assert os.path.isfile(ground_dem_filtered_path)

    bigger_holes_ground_path = os.path.join(
        temp_dir, dsm_name + "_ground_bigger_holes.sdat"
    )
    temp_array = dsm2dtm.expand_holes_in_raster(ground_dem_filtered_path)
    assert temp_array.shape == (286, 315)

    dsm2dtm.save_array_as_geotif(
        temp_array, ground_dem_filtered_path, bigger_holes_ground_path
    )
    assert os.path.isfile(bigger_holes_ground_path)

    dtm_path = os.path.join(temp_dir, dsm_name + "_dtm.sdat")
    dsm2dtm.close_gaps(bigger_holes_ground_path, dtm_path)
    assert os.path.isfile(dtm_path)

    dtm_tif_path = os.path.join(out_dir, dsm_name + "_dtm.tif")
    dsm2dtm.sdat_to_gtiff(dtm_path, dtm_tif_path)
    assert os.path.isfile(dtm_tif_path)

    dtm_array = gdal.Open(dtm_tif_path).ReadAsArray()
    assert dtm_array.shape == (286, 315)
    assert dsm2dtm.get_raster_resolution(dtm_tif_path) == (2.5193e-06, 2.5193e-06)
    assert dsm2dtm.get_raster_crs(dtm_tif_path) == 4326

    print("All tests passed!")
    shutil.rmtree(out_dir)
