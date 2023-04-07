"""
dsm2dtm - Generate DTM (Digital Terrain Model) from DSM (Digital Surface Model)
Author: Naman Jain
        naman.jain@btech2015.iitgn.ac.in
        www.namanji.wixsite.com/naman/
"""

import argparse
import os
from typing import Tuple

import numpy as np
import rasterio as rio
import richdem as rd
from rasterio.enums import Resampling

from src.constants import (NOISE_DEVIATION_FACTOR, TARGET_RES_UTM,
                           TARGET_RES_WGS)


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


def generate_slope_array(src_raster_path: str, no_data_value: int = -99999.0):
    """
    Generates slope array from the input DSM raster.

    Parameters:
        src_raster_path: Path to the source DSM.
        no_data_value: No data value in the source DSM.
                        Default value is -99999.0

    Returns:
        slope_array: Array representing the slope values for the source DSM.
    """
    with rio.open(src_raster_path) as raster:
        array = np.squeeze(raster.read())
        array = rd.rdarray(array, no_data=no_data_value)
        slope_array = rd.TerrainAttribute(array, attrib="slope_riserun")
        return slope_array


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


def remove_noise(src_array: np.ndarray, no_data_value: float = -99999.0):
    """
    Replaces noise (high elevation data points like roofs, etc.) from the ground DEM array, with no_data_value.

    Arguments:
        src_array: Numpy array representing ground DEM
        no_data_value: Float value that will be treated as no-data-value by GIS softwares (like QGIS).

    Returns:
        Numpy array with replaced noise values.
    """
    std = src_array[src_array != no_data_value].std()
    mean = src_array[src_array != no_data_value].mean()
    threshold_value = mean + NOISE_DEVIATION_FACTOR * std
    src_array[src_array >= threshold_value] = no_data_value
    return src_array


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


def subtract_rasters(rasterA_path: str, rasterB_path: str, out_path: str) -> str:
    """
    This function writes a new subtracted raster.

    Parameters:
        rasterA_path: Raster from which other raster will be subtracted.
        rasterB_path: Raster which will be subtracted from the other raster.
        out_path: Path where the subtracted raster will be generated.

    Returns:
        out_path: Path where the subtracted raster will be generated.
    """
    # TODO: catch exception in case rasters are of different shape and raise
    # proper error message
    with rio.open(rasterA_path) as raster_a, rio.open(rasterB_path) as raster_b:
        array_a = raster_a.read(masked=True)
        array_b = raster_b.read(masked=True)
        out_array = array_a - array_b
        profile = raster_a.profile
        with rio.open(out_path, "w", **profile) as dst:
            dst.write(out_array)
            return out_path


def replace_values(
    raster_a_path: str,
    raster_b_path: str,
    no_data_value: int = -99999.0,
    threshold: float = 0.98,
) -> np.ndarray:
    """
    Replaces values in input rasterA with no_data_value where cell value >= threshold in rasterB.

    Parameters:
        raster_a_path: Raster in which the values will be replaced.
        raster_b_path: Raster based on which the values will be replaced.
        no_data_value: Float value that will be treated as no-data-value by GIS softwares (like QGIS).
        threshold: Float value in rasterB based on which valued will be replaced in rasterA.

    Returns:
        updated_array: Numpy array with replaced values in raster A.
    """
    with rio.open(raster_a_path) as raster_a, rio.open(raster_b_path) as raster_b:
        array_a = raster_a.read(masked=True)
        array_b = raster_b.read(masked=True)
        np.place(array_a, array_b >= threshold, [no_data_value])
        updated_array = np.squeeze(array_a)
        return updated_array


def expand_holes_in_array(
    src_array: np.ndarray,
    search_window: int = 7,
    no_data_value: float = -99999.0,
    threshold: float = 50,
):
    """
    Expands holes (cells with no_data_value) in the input array.

    Arguments:
        src_array: Numpy arrray representing ground DEM.
        search_window: kernel size to be used as window
        threshold: threshold on percentage of cells with no_data_value

    Returns:
        Numpy array with expanded holes.
    """
    # TODO: refactor
    height, width = src_array.shape[0], src_array.shape[1]
    for i in range(int((search_window - 1) / 2), width, 1):
        for j in range(int((search_window - 1) / 2), height, 1):
            window = src_array[
                int(i - (search_window - 1) / 2) : int(i - (search_window - 1) / 2)
                + search_window,
                int(j - (search_window - 1) / 2) : int(j - (search_window - 1) / 2)
                + search_window,
            ]
            if (
                np.count_nonzero(window == no_data_value)
                >= (threshold * search_window**2) / 100
            ):
                try:
                    src_array[i, j] = no_data_value
                except:
                    pass
    return src_array


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
    no_data_value=-99999.0,
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
        x_res, y_res, dsm_crs, search_radius, smoothen_radius
    )
    # Generate DTM
    # STEP 1: Generate slope raster from dsm to get average slope value
    dsm_name = dsm_path.split("/")[-1].split(".")[0]
    slope_array = generate_slope_array(dsm_path, no_data_value=no_data_value)
    avg_slp = slope_array.mean().item()
    # STEP 2: Split DSM into ground and non-ground surface rasters
    ground_dem_path = os.path.join(temp_dir, dsm_name + "_ground.tif")
    non_ground_dem_path = os.path.join(temp_dir, dsm_name + "_non_ground.tif")
    extract_dtm(
        dsm_path,
        ground_dem_path,
        non_ground_dem_path,
        search_radius,
        avg_slp,
    )
    # STEP 3: Applying Gaussian Filter on the generated ground raster (parameters: radius = 45, mode = Circle)
    smoothened_ground_path = os.path.join(temp_dir, dsm_name + "_ground_smth.tif")
    smoothen_raster(ground_dem_path, smoothened_ground_path, smoothen_radius)
    # STEP 4: Generating a difference raster (ground DEM - smoothened ground DEM)
    diff_raster_path = os.path.join(temp_dir, dsm_name + "_ground_diff.tif")
    diff_raster_path = subtract_rasters(
        ground_dem_path, smoothened_ground_path, diff_raster_path
    )
    # STEP 5: Thresholding on the difference raster to replace values in Ground DEM by no-data values (threshold = 0.98)
    ground_array = replace_values(
        ground_dem_path,
        diff_raster_path,
        no_data_value=no_data_value,
        threshold=dsm_replace_threshold_val,
    )
    # STEP 6: Removing noisy spikes from the generated DTM
    ground_array = remove_noise(ground_array, no_data_value=no_data_value)
    # STEP 7: Expanding holes in the thresholded ground raster
    bigger_holes_ground_path = os.path.join(
        temp_dir, dsm_name + "_ground_bigger_holes.tif"
    )
    ground_array = expand_holes_in_array(ground_array, no_data_value=no_data_value)
    save_array_as_geotif(ground_array, diff_raster_path, bigger_holes_ground_path)
    # STEP 8: Close gaps in the DTM
    dtm_path = os.path.join(temp_dir, dsm_name + "_dtm.tif")
    close_gaps(bigger_holes_ground_path, dtm_path)
    return dtm_path


# -----------------------------------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate DTM from DSM")
    parser.add_argument("--dsm", help="dsm path string")
    args = parser.parse_args()
    dsm_path = args.dsm
    out_dir = "generated_dtm"
    dtm_path = main(dsm_path, out_dir)
    print("######### DTM generated at: ", dtm_path)
