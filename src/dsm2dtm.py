"""
dsm2dtm - Generate DTM (Digital Terrain Model) from DSM (Digital Surface Model)
Author: Naman Jain
        naman.jain@btech2015.iitgn.ac.in
        www.namanji.wixsite.com/naman/
"""

import os
import numpy as np
import rasterio as rio
from rasterio.enums import Resampling
import argparse
from typing import Tuple
from src.constants import TARGET_RES_UTM, TARGET_RES_WGS


def resample_raster(
    src_raster_path: str, resampled_raster_path: str, resampling_factor: float
) -> str:
    """
    Generates a new resampled raster.

    Parameters:
        src_raster_path: Path to the source raster.
        resampled_raster_path: Path where the resampled raster will be generated.
        resampling_factor: Factor by which the source raster will be resampled.

    Returns:
        resampled_raster_path: Path where the resampled raster will be generated.
    """
    with rio.open(src_raster_path) as raster:
        # resample data to target shape
        data = raster.read(
            out_shape=(
                raster.count,
                int(raster.height * resampling_factor),
                int(raster.width * resampling_factor),
            ),
            resampling=Resampling.bilinear,
        )
        # scale image transform
        transform = raster.transform * raster.transform.scale(
            (raster.width / data.shape[-1]), (raster.height / data.shape[-2])
        )
        profile = raster.profile
        profile.update(
            transform=transform,
            height=int(raster.height * resampling_factor),
            width=(raster.width * resampling_factor),
        )
        # import ipdb
        # ipdb.set_trace()
        with rio.open(resampled_raster_path, "w", **profile) as dst:
            dst.write(data)
    return resampled_raster_path


def upsample_raster(in_path, out_path, target_height, target_width):
    pass


def generate_slope_raster(in_path, out_path):
    """
    Generates a slope raster from the input DEM raster.
    Input:
        in_path: {string} path to the DEM raster
    Output:
        out_path: {string} path to the generated slope image
    """
    pass


def get_mean(raster_path, ignore_value=-9999.0):
    pass


def extract_dtm(dsm_path, ground_dem_path, non_ground_dem_path, radius, terrain_slope):
    """
    Generates a ground DEM and non-ground DEM raster from the input DSM raster.
    Input:
        dsm_path: {string} path to the DSM raster
        radius: {int} Search radius of kernel in cells.
        terrain_slope: {float} average slope of the input terrain
    Output:
        ground_dem_path: {string} path to the generated ground DEM raster
        non_ground_dem_path: {string} path to the generated non-ground DEM raster
    """
    cmd = "saga_cmd grid_filter 7 -INPUT {} -RADIUS {} -TERRAINSLOPE {} -GROUND {} -NONGROUND {}".format(
        dsm_path, radius, terrain_slope, ground_dem_path, non_ground_dem_path
    )
    os.system(cmd)


def remove_noise(ground_dem_path, out_path, ignore_value=-99999.0):
    """
    Removes noise (high elevation data points like roofs, etc.) from the ground DEM raster.
    Replaces values in those pixels with No data Value (-99999.0)
    Input:
        ground_dem_path: {string} path to the generated ground DEM raster
        no_data_value: {float} replacing value in the ground raster (to be treated as No Data Value)
    Output:
        out_path: {string} path to the filtered ground DEM raster
    """
    pass


def save_array_as_geotif(array, source_tif_path, out_path):
    """
    Generates a geotiff raster from the input numpy array (height * width * depth)
    Input:
        array: {numpy array} numpy array to be saved as geotiff
        source_tif_path: {string} path to the geotiff from which projection and geotransformation information will be extracted.
    Output:
        out_path: {string} path to the generated Geotiff raster
    """
    pass


def sdat_to_gtiff(sdat_raster_path, out_gtiff_path):
    pass


def close_gaps(in_path, out_path, threshold=0.1):
    """
    Interpolates the holes (no data value) in the input raster.
    Input:
        in_path: {string} path to the input raster with holes
        threshold: {float} Tension Threshold
    Output:
        out_path: {string} path to the generated raster with closed holes.
    """
    cmd = "saga_cmd grid_tools 7 -INPUT {} -THRESHOLD {} -RESULT {}".format(
        in_path, threshold, out_path
    )
    os.system(cmd)


def smoothen_raster(in_path, out_path, radius=2):
    """
    Applies gaussian filter to the input raster.
    Input:
        in_path: {string} path to the input raster
        radius: {int} kernel radius to be used for smoothing
    Output:
        out_path: {string} path to the generated smoothened raster
    """
    cmd = "saga_cmd grid_filter 1 -INPUT {} -RESULT {} -KERNEL_TYPE 0 -KERNEL_RADIUS {}".format(
        in_path, out_path, radius
    )
    os.system(cmd)


def subtract_rasters(rasterA_path, rasterB_path, out_path, no_data_value=-99999.0):
    pass


def replace_values(
    rasterA_path, rasterB_path, out_path, no_data_value=-99999.0, threshold=0.98
):
    """
    Replaces values in input rasterA with no_data_value where cell value >= threshold in rasterB
    Input:
        rasterA_path: {string} path to the input rasterA
        rasterB_path: {string} path to the input rasterB
    Output:
        out_path: {string} path to the generated raster
    """
    pass


def expand_holes_in_raster(
    in_path, search_window=7, no_data_value=-99999.0, threshold=50
):
    """
    Expands holes (cells with no_data_value) in the input raster.
    Input:
        in_path: {string} path to the input raster
        search_window: {int} kernel size to be used as window
        threshold: {float} threshold on percentage of cells with no_data_value
    Output:
        np_raster: {numpy array} Returns the modified input raster's array
    """
    pass


def get_raster_crs(raster_path: str) -> int:
    """
    Returns the CRS (Coordinate Reference System) of the source raster

    Parameters:
        raster_path: Path to the source raster.

    Returns:
        crs: EPSG value of the Coordinate Reference System.
    """
    with rio.open(raster_path) as raster:
        crs = raster.crs.to_epsg()
        return crs


def get_raster_resolution(raster_path: str) -> Tuple[float, float]:
    """
    Returns the absolute values of x and y resolution of the raster.

    Paramteres:
        raster_path: Path to the source raster.

    Returns:
        x_res: Resolution of the cell in X direction.
        y_res: Resolution of the cell in Y direction.
    """
    with rio.open(raster_path) as raster:
        x_res = raster.transform[0]
        y_res = -(raster.transform[4])
        # taking a negative for y_res as we are interested in the abslute value
        return x_res, y_res


def get_res_and_downsample(dsm_path, temp_dir):
    # check DSM resolution. Downsample if DSM is of very high resolution to save processing time.
    pass


def get_updated_params(
    x_res: float, y_res: float, dsm_crs: int, search_radius: int, smoothen_radius: int
) -> Tuple[int, int]:
    """
    Parameters search_radius and smoothen_radius are set wrt to 30cm DSM.
    This function returns updated parameters if DSM is of coarser resolution.

    Parameters:
        x_res: Resolution of the cell in X direction.
        y_res: Resolution of the cell in Y direction.
        dsm_crs: EPSG value of the Coordinate Reference System.
        search_radius: Search radius of kernel (unit: no. of cells).
        smoothen_radius: Kernel radius to be used for smoothing (unit: no. of cells)

    Returns:
        search_radius: Updated Search radius of kernel
        smoothen_radius: Updated Kernel radius to be used for smoothing
    """
    if dsm_crs != 4326:
        if x_res > TARGET_RES_UTM or y_res > TARGET_RES_UTM:
            search_radius = int((min(x_res, y_res) * search_radius) / TARGET_RES_UTM)
            smoothen_radius = int(
                (min(x_res, y_res) * smoothen_radius) / TARGET_RES_UTM
            )
    else:
        if x_res > TARGET_RES_WGS or y_res > TARGET_RES_WGS:
            search_radius = int((min(x_res, y_res) * search_radius) / TARGET_RES_WGS)
            smoothen_radius = int(
                (min(x_res, y_res) * smoothen_radius) / TARGET_RES_WGS
            )
    return search_radius, smoothen_radius


def get_downsampling_factor(x_res: float, y_res: float, raster_crs: int) -> float:
    """
    Returns the downsampling factor based on raster CRS.

    Parameters:
        x_res: Resolution of the cell in X direction.
        y_res: Resolution of the cell in Y direction.
        raster_crs: EPSG value of the Coordinate Reference System.

    Returns:
        downsampling_factor: Factor by which the source raster needs to be downsampled.
    """
    downsampling_factor = 1
    if raster_crs != 4326:
        target_res = TARGET_RES_UTM
        # rounding in case of metres because of too much precision like 8 decimal places, eg: 0.29999997
        x_res = round(x_res, 2)
        y_res = round(x_res, 2)
    else:
        target_res = TARGET_RES_WGS
    if x_res < target_res or y_res < target_res:
        downsampling_factor = target_res / min(x_res, y_res)
    return downsampling_factor


def main(
    dsm_path,
    out_dir,
    search_radius=40,
    smoothen_radius=45,
    dsm_replace_threshold_val=0.98,
):
    # master function that calls all other functions
    os.makedirs(out_dir, exist_ok=True)
    temp_dir = os.path.join(out_dir, "temp_files")
    os.makedirs(temp_dir, exist_ok=True)
    x_res, y_res = get_raster_resolution(dsm_path)
    dsm_crs = get_raster_crs(dsm_path)
    downsampling_factor = get_downsampling_factor(x_res, y_res, dsm_crs)
    dsm_path = resample_raster(dsm_path, temp_dir, downsampling_factor)
    # get updated params wrt to DSM resolution
    search_radius, smoothen_radius = get_updated_params(
        dsm_path, search_radius, smoothen_radius
    )
    # Generate DTM
    # STEP 1: Generate slope raster from dsm to get average slope value
    dsm_name = dsm_path.split("/")[-1].split(".")[0]
    dsm_slp_path = os.path.join(temp_dir, dsm_name + "_slp.tif")
    generate_slope_raster(dsm_path, dsm_slp_path)
    avg_slp = int(get_mean(dsm_slp_path))
    # STEP 2: Split DSM into ground and non-ground surface rasters
    ground_dem_path = os.path.join(temp_dir, dsm_name + "_ground.sdat")
    non_ground_dem_path = os.path.join(temp_dir, dsm_name + "_non_ground.sdat")
    extract_dtm(
        dsm_path,
        ground_dem_path,
        non_ground_dem_path,
        search_radius,
        avg_slp,
    )
    # STEP 3: Applying Gaussian Filter on the generated ground raster (parameters: radius = 45, mode = Circle)
    smoothened_ground_path = os.path.join(temp_dir, dsm_name + "_ground_smth.sdat")
    smoothen_raster(ground_dem_path, smoothened_ground_path, smoothen_radius)
    # STEP 4: Generating a difference raster (ground DEM - smoothened ground DEM)
    diff_raster_path = os.path.join(temp_dir, dsm_name + "_ground_diff.sdat")
    subtract_rasters(ground_dem_path, smoothened_ground_path, diff_raster_path)
    # STEP 5: Thresholding on the difference raster to replace values in Ground DEM by no-data values (threshold = 0.98)
    thresholded_ground_path = os.path.join(
        temp_dir, dsm_name + "_ground_thresholded.sdat"
    )
    replace_values(
        ground_dem_path,
        diff_raster_path,
        thresholded_ground_path,
        threshold=dsm_replace_threshold_val,
    )
    # STEP 6: Removing noisy spikes from the generated DTM
    ground_dem_filtered_path = os.path.join(temp_dir, dsm_name + "_ground_filtered.tif")
    remove_noise(thresholded_ground_path, ground_dem_filtered_path)
    # STEP 7: Expanding holes in the thresholded ground raster
    bigger_holes_ground_path = os.path.join(
        temp_dir, dsm_name + "_ground_bigger_holes.sdat"
    )
    temp = expand_holes_in_raster(ground_dem_filtered_path)
    save_array_as_geotif(temp, ground_dem_filtered_path, bigger_holes_ground_path)
    # STEP 8: Close gaps in the DTM
    dtm_path = os.path.join(temp_dir, dsm_name + "_dtm.sdat")
    close_gaps(bigger_holes_ground_path, dtm_path)
    # STEP 9: Convert to GeoTiff
    dtm_array = gdal.Open(dtm_path).ReadAsArray()
    dtm_tif_path = os.path.join(out_dir, dsm_name + "_dtm.tif")
    # save_array_as_geotif(dtm_array, dsm_path, dtm_tif_path)
    sdat_to_gtiff(dtm_path, dtm_tif_path)
    return dtm_tif_path


# -----------------------------------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate DTM from DSM")
    parser.add_argument("--dsm", help="dsm path string")
    args = parser.parse_args()
    dsm_path = args.dsm
    out_dir = "generated_dtm"
    dtm_path = main(dsm_path, out_dir)
    print("######### DTM generated at: ", dtm_path)
