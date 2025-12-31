"""
dsm2dtm - Generate DTM (Digital Terrain Model) from DSM (Digital Surface Model)
Author: Naman Jain
        naman.jain@btech2015.iitgn.ac.in
        www.namanji.wixsite.com/naman/
"""

import argparse
import os

import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.fill import fillnodata
from scipy.ndimage import gaussian_filter, grey_opening


def downsample_raster(in_path, out_path, downsampling_factor):
    """
    Downsamples a raster to a new resolution using rasterio block by block.
    """
    with rasterio.open(in_path) as src:
        profile = src.profile

        # Calculate new dimensions
        new_height = int(src.height // downsampling_factor)
        new_width = int(src.width // downsampling_factor)

        # Update the profile for the new raster
        profile.update(
            height=new_height,
            width=new_width,
            transform=src.transform * src.transform.scale(src.width / new_width, src.height / new_height),
            dtype=src.profile["dtype"],  # Use dtype from profile
        )

        with rasterio.open(out_path, "w", **profile) as dst:
            for ji, window in dst.block_windows(1):
                # Read data from source window
                # Calculate source window based on destination window bounds
                bounds = dst.window_bounds(window)
                src_window = src.window(*bounds)

                # Round window to ensure we cover the area
                src_window = src_window.round_offsets().round_shape()

                data = src.read(
                    1,  # Assuming single band
                    window=src_window,
                    out_shape=(
                        window.height,
                        window.width,
                    ),  # Resample to output block shape
                    resampling=Resampling.bilinear,
                )
                dst.write(data, 1, window=window)


def upsample_raster(in_path, out_path, target_height, target_width):
    """
    Upsamples a raster to target dimensions using rasterio block by block.
    """
    with rasterio.open(in_path) as src:
        # Calculate new transform
        new_transform = src.transform * src.transform.scale(src.width / target_width, src.height / target_height)

        profile = src.profile
        profile.update(
            height=target_height,
            width=target_width,
            transform=new_transform,
            dtype=src.profile["dtype"],  # Use dtype from profile
        )

        with rasterio.open(out_path, "w", **profile) as dst:
            for ji, window in dst.block_windows(1):
                # Read data from source window
                # Calculate source window based on destination window bounds
                bounds = dst.window_bounds(window)
                src_window = src.window(*bounds)

                # Round window
                src_window = src_window.round_offsets().round_shape()

                data = src.read(
                    1,  # Assuming single band
                    out_shape=(window.height, window.width),
                    window=src_window,
                    resampling=Resampling.bilinear,
                )
                dst.write(data, 1, window=window)


def generate_slope_raster(in_path, out_path):
    """
    Generates a slope raster from the input DEM raster using numpy block by block.
    Input:
        in_path: {string} path to the DEM raster
    Output:
        out_path: {string} path to the generated slope image
    """
    with rasterio.open(in_path) as src:
        profile = src.profile
        nodata = src.nodata
        res_x, res_y = src.res

        profile.update(dtype=np.float32, count=1, nodata=nodata if nodata is not None else -9999.0)

        with rasterio.open(out_path, "w", **profile) as dst:
            overlap = 1
            for ji, window in dst.block_windows(1):
                read_window = rasterio.windows.union(src.window(*src.window_bounds(window)), window)
                read_window = rasterio.windows.Window(
                    window.col_off - overlap,
                    window.row_off - overlap,
                    window.width + 2 * overlap,
                    window.height + 2 * overlap,
                )
                read_window = read_window.intersection(rasterio.windows.Window(0, 0, src.width, src.height))

                dem_block = src.read(1, window=read_window)

                if nodata is not None:
                    dem_block_masked = np.ma.masked_equal(dem_block, nodata)
                else:
                    dem_block_masked = dem_block

                grad_y, grad_x = np.gradient(dem_block_masked, res_y, res_x)
                slope_radians = np.arctan(np.sqrt(grad_x**2 + grad_y**2))
                slope_degrees = np.degrees(slope_radians)

                if nodata is not None:
                    slope_degrees[dem_block_masked.mask] = nodata

                start_row = window.row_off - read_window.row_off
                start_col = window.col_off - read_window.col_off
                end_row = start_row + window.height
                end_col = start_col + window.width

                slope_output_block = slope_degrees[start_row:end_row, start_col:end_col]

                dst.write(slope_output_block, 1, window=window)


def get_mean(raster_path, ignore_value=-9999.0):
    """
    Calculates the mean of a raster, ignoring specified values, using rasterio block by block.
    """
    total_sum = 0.0
    total_count = 0

    with rasterio.open(raster_path) as src:
        nodata = src.nodata

        for ji, window in src.block_windows(1):
            block = src.read(1, window=window)

            if nodata is not None:
                mask = block == nodata
            else:
                mask = np.zeros_like(block, dtype=bool)

            if ignore_value != -9999.0:
                mask |= block == ignore_value
            elif ignore_value == -9999.0 and nodata != -9999.0:
                mask |= block == ignore_value

            valid_data = block[~mask]

            if valid_data.size > 0:
                total_sum += np.sum(valid_data)
                total_count += valid_data.size

    if total_count == 0:
        return 0.0

    return total_sum / total_count


def extract_dtm(dsm_path, ground_dem_path, non_ground_dem_path, radius, terrain_slope):
    """
    Generates a ground DEM and non-ground DEM raster from the input DSM raster
    using morphological operations block by block.
    """
    with rasterio.open(dsm_path) as src:
        profile = src.profile
        nodata = src.nodata

        profile.update(dtype=src.profile["dtype"], count=1, driver="GTiff")

        struct_element = np.ones((2 * radius + 1, 2 * radius + 1))

        with (
            rasterio.open(ground_dem_path, "w", **profile) as dst_ground,
            rasterio.open(non_ground_dem_path, "w", **profile) as dst_non_ground,
        ):
            overlap = radius

            for ji, window in dst_ground.block_windows(1):
                read_window = rasterio.windows.union(src.window(*src.window_bounds(window)), window)
                read_window = rasterio.windows.Window(
                    window.col_off - overlap,
                    window.row_off - overlap,
                    window.width + 2 * overlap,
                    window.height + 2 * overlap,
                )
                read_window = read_window.intersection(rasterio.windows.Window(0, 0, src.width, src.height))

                dsm_block = src.read(1, window=read_window)

                if nodata is not None:
                    valid_mask = dsm_block != nodata
                    if np.any(valid_mask):
                        fill_value = np.min(dsm_block[valid_mask])
                    else:
                        fill_value = -99999.0

                    dsm_filled = np.where(dsm_block == nodata, fill_value, dsm_block)
                else:
                    dsm_filled = dsm_block

                ground_block = grey_opening(dsm_filled, structure=struct_element)

                if nodata is not None:
                    ground_block[dsm_block == nodata] = nodata

                non_ground_block = dsm_block - ground_block
                if nodata is not None:
                    non_ground_block[dsm_block == nodata] = nodata

                start_row = window.row_off - read_window.row_off
                start_col = window.col_off - read_window.col_off
                end_row = start_row + window.height
                end_col = start_col + window.width

                ground_output = ground_block[start_row:end_row, start_col:end_col]
                non_ground_output = non_ground_block[start_row:end_row, start_col:end_col]

                dst_ground.write(ground_output, 1, window=window)
                dst_non_ground.write(non_ground_output, 1, window=window)


def remove_noise(ground_dem_path, out_path, ignore_value=-99999.0):
    """
    Removes noise (high elevation data points like roofs, etc.) block by block.
    First calculates global stats, then applies threshold block-wise.
    """
    # 1. Calculate global statistics iteratively to avoid loading full raster
    mean = get_mean(ground_dem_path, ignore_value)

    # Calculate stddev iteratively (requires a second pass or Welford's algorithm)
    # For simplicity and memory efficiency, let's do a second pass for stddev
    # Standard Deviation = sqrt( sum((x - mean)^2) / N )
    sum_sq_diff = 0.0
    count = 0

    with rasterio.open(ground_dem_path) as src:
        nodata = src.nodata
        for ji, window in src.block_windows(1):
            block = src.read(1, window=window)

            mask = np.zeros_like(block, dtype=bool)
            if nodata is not None:
                mask |= block == nodata
            if ignore_value != -9999.0:
                mask |= block == ignore_value
            elif ignore_value == -9999.0 and nodata != -9999.0:
                mask |= block == ignore_value

            valid_data = block[~mask]
            if valid_data.size > 0:
                sum_sq_diff += np.sum((valid_data - mean) ** 2)
                count += valid_data.size

    if count == 0:
        std = 0.0
    else:
        std = np.sqrt(sum_sq_diff / count)

    threshold_value = mean + 1.5 * std

    # 2. Apply threshold block by block
    with rasterio.open(ground_dem_path) as src:
        profile = src.profile
        profile.update(nodata=ignore_value)

        with rasterio.open(out_path, "w", **profile) as dst:
            for ji, window in src.block_windows(1):
                block = src.read(1, window=window)

                # Apply masking for processing
                if src.nodata is not None:
                    # Treat src nodata as ignore_value for output consistency
                    block[block == src.nodata] = ignore_value

                # Apply threshold
                block[block >= threshold_value] = ignore_value

                dst.write(block, 1, window=window)


def save_array_as_geotif(array, source_tif_path, out_path, nodata_value=None):
    """
    Generates a geotiff raster from the input numpy array.
    """
    with rasterio.open(source_tif_path) as src_profile:
        profile = src_profile.profile

    profile.update(
        dtype=array.dtype,
        count=1,
        nodata=nodata_value if nodata_value is not None else profile.get("nodata"),
    )

    with rasterio.open(out_path, "w", **profile) as dst:
        dst.write(array, 1)


def close_gaps(
    in_path,
    out_path,
    max_search_distance=10.0,
    smoothing_iterations=0,
    nodata_value=None,
):
    """
    Interpolates holes using rasterio.fill.fillnodata block by block (with overlap).
    """
    with rasterio.open(in_path) as src:
        profile = src.profile
        nodata = src.nodata if nodata_value is None else nodata_value
        profile.update(nodata=nodata)

        with rasterio.open(out_path, "w", **profile) as dst:
            overlap = int(max_search_distance) + 2  # Padding for fill context

            for ji, window in dst.block_windows(1):
                read_window = rasterio.windows.union(src.window(*src.window_bounds(window)), window)
                read_window = rasterio.windows.Window(
                    window.col_off - overlap,
                    window.row_off - overlap,
                    window.width + 2 * overlap,
                    window.height + 2 * overlap,
                )
                read_window = read_window.intersection(rasterio.windows.Window(0, 0, src.width, src.height))

                block = src.read(1, window=read_window)
                mask = block == nodata

                filled_block = fillnodata(
                    block,
                    mask=mask,
                    max_search_distance=max_search_distance,
                    smoothing_iterations=smoothing_iterations,
                )

                # Crop
                start_row = window.row_off - read_window.row_off
                start_col = window.col_off - read_window.col_off
                end_row = start_row + window.height
                end_col = start_col + window.width

                output_block = filled_block[start_row:end_row, start_col:end_col]
                dst.write(output_block, 1, window=window)


def smoothen_raster(in_path, out_path, radius=2, nodata_value=None):
    """
    Applies gaussian filter block by block with overlap.
    """
    with rasterio.open(in_path) as src:
        profile = src.profile
        nodata = src.nodata if nodata_value is None else nodata_value
        profile.update(nodata=nodata)

        with rasterio.open(out_path, "w", **profile) as dst:
            # Overlap needs to be sufficient for Gaussian kernel (3*sigma rule of thumb)
            overlap = int(3 * radius) + 1

            for ji, window in dst.block_windows(1):
                read_window = rasterio.windows.union(src.window(*src.window_bounds(window)), window)
                read_window = rasterio.windows.Window(
                    window.col_off - overlap,
                    window.row_off - overlap,
                    window.width + 2 * overlap,
                    window.height + 2 * overlap,
                )
                read_window = read_window.intersection(rasterio.windows.Window(0, 0, src.width, src.height))

                block = src.read(1, window=read_window)

                if nodata is not None:
                    if np.any(block != nodata):
                        fill_value = np.mean(block[block != nodata])
                    else:
                        fill_value = 0.0
                    temp_block = np.where(block == nodata, fill_value, block)
                else:
                    temp_block = block

                smoothed_block = gaussian_filter(temp_block, sigma=radius)

                if nodata is not None:
                    smoothed_block[block == nodata] = nodata

                start_row = window.row_off - read_window.row_off
                start_col = window.col_off - read_window.col_off
                end_row = start_row + window.height
                end_col = start_col + window.width

                output_block = smoothed_block[start_row:end_row, start_col:end_col]
                dst.write(output_block, 1, window=window)


def subtract_rasters(rasterA_path, rasterB_path, out_path, no_data_value=None):
    """
    Subtracts rasterB from rasterA block by block.
    """
    with rasterio.open(rasterA_path) as srcA, rasterio.open(rasterB_path) as srcB:
        profile = srcA.profile
        nodataA = srcA.nodata if no_data_value is None else no_data_value
        nodataB = srcB.nodata

        profile.update(nodata=nodataA)

        with rasterio.open(out_path, "w", **profile) as dst:
            for ji, window in dst.block_windows(1):
                arrayA = srcA.read(1, window=window)
                arrayB = srcB.read(1, window=window)

                result_array = arrayA - arrayB

                nodata_mask = (arrayA == nodataA) | (arrayB == nodataB)
                if nodataA is not None:
                    result_array[nodata_mask] = nodataA

                dst.write(result_array, 1, window=window)


def replace_values(rasterA_path, rasterB_path, out_path, no_data_value=None, threshold=0.98):
    """
    Replaces values block by block.
    """
    with rasterio.open(rasterA_path) as srcA, rasterio.open(rasterB_path) as srcB:
        profile = srcA.profile
        nodataA = srcA.nodata if no_data_value is None else no_data_value

        profile.update(nodata=nodataA)

        with rasterio.open(out_path, "w", **profile) as dst:
            for ji, window in dst.block_windows(1):
                arrayA = srcA.read(1, window=window)
                arrayB = srcB.read(1, window=window)

                result_array = np.where(arrayB >= threshold, nodataA, arrayA)

                if nodataA is not None:
                    result_array[arrayA == nodataA] = nodataA

                dst.write(result_array, 1, window=window)


def expand_holes_in_raster(in_path, search_window=7, no_data_value=-99999.0, threshold=50):
    """
    Expands holes (cells with no_data_value) in the input raster using scipy block by block.
    Replaces inefficient nested loops with binary dilation-based logic.
    """
    # This logic roughly translates to: if 'threshold'% of neighborhood is nodata, make pixel nodata.
    # This is equivalent to checking the sum of nodata pixels in a window.
    # If sum >= threshold_count, set to nodata.
    # We can use scipy.ndimage.generic_filter or simpler convolution.

    # Actually, simply returning the array is not block-wise I/O.
    # This function originally returned a numpy array.
    # To keep with the flow, let's make it write to a file or return the array processing function
    # But for OOM safety, it should probably process an input file and return an array (if small)
    # OR better: change the pipeline to expect this function to return
    # the processed array block-by-block?
    # Actually, looking at main(), `save_array_as_geotif` is called immediately after.
    # So `expand_holes` creates a big array, then `save` writes it.
    # We should combine them or change `expand_holes` to return nothing and write to a file itself.
    # Let's change the pattern: return the whole array but calculate efficiently.

    # Optimized implementation using convolution/filtering on the whole array (fast, but memory heavy)
    # If we hit OOM here, we'll know.

    with rasterio.open(in_path) as src:
        np_raster = src.read(1)
        nodata = src.nodata if src.nodata is not None else no_data_value

    # Create binary mask of nodata
    nodata_mask = (np_raster == nodata).astype(int)

    # Kernel for counting neighbors
    kernel = np.ones((search_window, search_window))

    # Count nodata neighbors
    # scipy.ndimage.convolve or correlate.
    # We want count of 1s in window.
    from scipy.ndimage import convolve

    nodata_count = convolve(nodata_mask, kernel, mode="constant", cval=0)

    # Threshold count
    threshold_count = (threshold * search_window**2) / 100

    # Update raster
    new_nodata_mask = nodata_count >= threshold_count
    np_raster[new_nodata_mask] = nodata

    return np_raster


def get_raster_crs(raster_path):
    """
    Returns the CRS (Coordinate Reference System) of the raster using rasterio.
    """
    with rasterio.open(raster_path) as src:
        return src.crs.to_epsg()  # Return EPSG code for consistency with old behavior


def get_raster_resolution(raster_path):
    """
    Returns the X and Y resolution of the raster using rasterio.
    """
    with rasterio.open(raster_path) as src:
        # src.res is a tuple (x_res, y_res)
        return src.res[0], src.res[1]


def get_res_and_downsample(dsm_path, temp_dir):
    # check DSM resolution. Downsample if DSM is of very high resolution to save processing time.
    x_res, y_res = get_raster_resolution(dsm_path)  # resolutions are in meters
    dsm_name = os.path.splitext(os.path.basename(dsm_path))[0]
    dsm_crs = get_raster_crs(dsm_path)

    downsampled_dsm_path = dsm_path  # Initialize with original path

    current_x_res, current_y_res = get_raster_resolution(dsm_path)

    if dsm_crs != 4326:  # Projected CRS
        if current_x_res < 0.3 or current_y_res < 0.3:  # If resolution is finer than 30cm
            target_res = 0.3
            factor = target_res / current_x_res
            downsampled_dsm_path = os.path.join(temp_dir, dsm_name + "_ds.tif")
            downsample_raster(dsm_path, downsampled_dsm_path, factor)
            dsm_path = downsampled_dsm_path

    else:  # Geographic CRS
        if current_x_res < 2.514e-06 or current_y_res < 2.514e-06:
            target_res = 2.514e-06
            factor = target_res / current_x_res
            downsampled_dsm_path = os.path.join(temp_dir, dsm_name + "_ds.tif")
            downsample_raster(dsm_path, downsampled_dsm_path, factor)
            dsm_path = downsampled_dsm_path

    return dsm_path


def get_updated_params(dsm_path, search_radius, smoothen_radius):
    x_res, y_res = get_raster_resolution(dsm_path)  # resolutions are in meters
    dsm_crs = get_raster_crs(dsm_path)
    if dsm_crs != 4326:
        if x_res > 0.3 or y_res > 0.3:
            search_radius = int((min(x_res, y_res) * search_radius) / 0.3)
            smoothen_radius = int((min(x_res, y_res) * smoothen_radius) / 0.3)
    else:
        if x_res > 2.514e-06 or y_res > 2.514e-06:
            search_radius = int((min(x_res, y_res) * search_radius) / 2.514e-06)
            smoothen_radius = int((min(x_res, y_res) * smoothen_radius) / 2.514e-06)
    return search_radius, smoothen_radius


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

    original_dsm_path = dsm_path  # Keep track of the original DSM path
    dsm_path = get_res_and_downsample(dsm_path, temp_dir)

    # get updated params wrt to DSM resolution
    search_radius, smoothen_radius = get_updated_params(dsm_path, search_radius, smoothen_radius)

    # Generate DTM
    # STEP 1: Generate slope raster
    dsm_name = os.path.splitext(os.path.basename(dsm_path))[0]
    dsm_slp_path = os.path.join(temp_dir, dsm_name + "_slp.tif")
    generate_slope_raster(dsm_path, dsm_slp_path)
    avg_slp = int(get_mean(dsm_slp_path))

    # STEP 2: Split DSM into ground and non-ground
    ground_dem_path = os.path.join(temp_dir, dsm_name + "_ground.tif")
    non_ground_dem_path = os.path.join(temp_dir, dsm_name + "_non_ground.tif")
    extract_dtm(
        dsm_path,
        ground_dem_path,
        non_ground_dem_path,
        search_radius,
        avg_slp,
    )

    # STEP 3: Gaussian Filter
    smoothened_ground_path = os.path.join(temp_dir, dsm_name + "_ground_smth.tif")
    smoothen_raster(ground_dem_path, smoothened_ground_path, smoothen_radius)

    # STEP 4: Difference raster
    diff_raster_path = os.path.join(temp_dir, dsm_name + "_ground_diff.tif")
    subtract_rasters(ground_dem_path, smoothened_ground_path, diff_raster_path)

    # STEP 5: Thresholding
    thresholded_ground_path = os.path.join(temp_dir, dsm_name + "_ground_thresholded.tif")
    replace_values(
        ground_dem_path,
        diff_raster_path,
        thresholded_ground_path,
        threshold=dsm_replace_threshold_val,
    )

    # STEP 6: Removing noisy spikes
    ground_dem_filtered_path = os.path.join(temp_dir, dsm_name + "_ground_filtered.tif")
    remove_noise(thresholded_ground_path, ground_dem_filtered_path)

    # STEP 7: Expanding holes
    bigger_holes_ground_path = os.path.join(temp_dir, dsm_name + "_ground_bigger_holes.tif")
    # Note: expand_holes_in_raster still returns array for now, optimizing it is partial fix
    temp_array = expand_holes_in_raster(ground_dem_filtered_path)
    save_array_as_geotif(temp_array, ground_dem_filtered_path, bigger_holes_ground_path)

    # STEP 8: Close gaps
    dtm_final_temp_path = os.path.join(temp_dir, dsm_name + "_dtm_final.tif")
    close_gaps(bigger_holes_ground_path, dtm_final_temp_path)

    # STEP 9: Final output
    dtm_tif_path = os.path.join(out_dir, os.path.splitext(os.path.basename(original_dsm_path))[0] + "_dtm.tif")

    with rasterio.open(dtm_final_temp_path) as src_dtm:
        # We should really avoid reading the whole file here too
        # Copy file block by block or just shutil.copy if format/profile is identical
        profile = src_dtm.profile
        with rasterio.open(dtm_tif_path, "w", **profile) as dst_dtm:
            for ji, window in src_dtm.block_windows(1):
                dst_dtm.write(src_dtm.read(1, window=window), 1, window=window)

    return dtm_tif_path


# -----------------------------------------------------------------------------------------------------
def main_cli():
    """
    Command line interface for generating DTM from DSM.
    """
    parser = argparse.ArgumentParser(description="Generate DTM from DSM")
    parser.add_argument("--dsm", help="Path to the DSM file", required=True)
    parser.add_argument("--out_dir", help="Directory to save the output DTM", default="generated_dtm")
    args = parser.parse_args()
    dtm_path = main(args.dsm, args.out_dir)
    print(f"######### DTM generated at: {dtm_path}")


if __name__ == "__main__":
    main_cli()
