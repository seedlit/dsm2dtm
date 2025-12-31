"""
dsm2dtm - Generate DTM (Digital Terrain Model) from DSM (Digital Surface Model)
Author: Naman Jain
        naman.jain@btech2015.iitgn.ac.in
"""

import os
import numpy as np
import rasterio
import argparse
from rasterio.enums import Resampling
from scipy.ndimage import grey_opening, gaussian_filter
from rasterio.fill import fillnodata

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
            transform=src.transform * src.transform.scale(
                src.width / new_width, src.height / new_height
            ),
            dtype=src.dtype # Keep original dtype unless explicitly changing
        )

        with rasterio.open(out_path, 'w', **profile) as dst:
            # Read and write block by block
            # This is a simplified block-wise approach, for complex resampling
            # more sophisticated iteration over blocks might be needed.
            for ji, window in dst.block_windows(1):
                # Read data from source window
                # Calculate the window in the source raster that corresponds to the output block
                src_window = rasterio.windows.get_parent(window, src.transform, dst.transform)
                
                data = src.read(
                    1, # Assuming single band
                    window=src_window,
                    out_shape=(window.height, window.width), # Resample to output block shape
                    resampling=Resampling.bilinear
                )
                dst.write(data, 1, window=window)

def upsample_raster(in_path, out_path, target_height, target_width):
    """
    Upsamples a raster to target dimensions using rasterio block by block.
    """
    with rasterio.open(in_path) as src:
        # Calculate new transform
        new_transform = src.transform * src.transform.scale(
            src.width / target_width, src.height / target_height
        )

        profile = src.profile
        profile.update(
            height=target_height,
            width=target_width,
            transform=new_transform,
            dtype=src.dtype # Keep original dtype unless explicitly changing
        )

        with rasterio.open(out_path, 'w', **profile) as dst:
            for ji, window in dst.block_windows(1):
                # Read data from source that covers the output window,
                # resampling it to the output window's size
                data = src.read(
                    1, # Assuming single band
                    out_shape=(window.height, window.width),
                    window=rasterio.windows.get_parent(window, src.transform, dst.transform),
                    resampling=Resampling.bilinear
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
        res_x, res_y = src.res # Get resolution (assuming same units as Z, or degrees)
        
        # Update profile for output slope raster
        profile.update(
            dtype=np.float32, # Slope is typically float
            count=1,
            nodata=nodata if nodata is not None else -9999.0 # Ensure nodata is set
        )

        with rasterio.open(out_path, 'w', **profile) as dst:
            # Iterate over blocks, reading a slightly larger window for gradient calculation
            # to handle edge effects, then writing the central part.
            
            # Define an overlap for gradient calculation (e.g., 1 pixel on each side)
            overlap = 1
            
            for ji, window in dst.block_windows(1):
                # Calculate a larger window for reading from source to handle overlap
                read_window = rasterio.windows.union(
                    src.window(*src.window_bounds(window)), # window_bounds returns (left, bottom, right, top)
                    window # Ensure the output window itself is covered
                )
                # Ideally, just expanding the window is simpler if we assume aligned pixels
                read_window = rasterio.windows.Window(
                    window.col_off - overlap,
                    window.row_off - overlap,
                    window.width + 2 * overlap,
                    window.height + 2 * overlap
                )
                # intersect with dataset bounds
                read_window = read_window.intersection(rasterio.windows.Window(0, 0, src.width, src.height))

                # Read data with overlap
                dem_block = src.read(1, window=read_window)

                # Handle nodata within the block for gradient calculation
                if nodata is not None:
                    dem_block_masked = np.ma.masked_equal(dem_block, nodata)
                else:
                    dem_block_masked = dem_block

                # Calculate gradient (slope) for the block
                # Dividing by resolution to get gradient in correct units (dZ/dX, dZ/dY)
                # Note: np.gradient returns (gradient_axis_0, gradient_axis_1) -> (dy, dx)
                grad_y, grad_x = np.gradient(dem_block_masked, res_y, res_x)
                
                slope_radians = np.arctan(np.sqrt(grad_x**2 + grad_y**2))
                slope_degrees = np.degrees(slope_radians)

                # Apply nodata from masked array or original if needed
                if nodata is not None:
                    slope_degrees[dem_block_masked.mask] = nodata
                
                # Extract the central part corresponding to the output block
                # We need to map the output window coordinates back to the read_block coordinates
                # Calculate offsets relative to the read_window
                start_row = window.row_off - read_window.row_off
                start_col = window.col_off - read_window.col_off
                end_row = start_row + window.height
                end_col = start_col + window.width

                slope_output_block = slope_degrees[
                    start_row:end_row,
                    start_col:end_col
                ]

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
            # Read block
            block = src.read(1, window=window)
            
            # Mask nodata and ignore_value
            if nodata is not None:
                mask = (block == nodata)
            else:
                mask = np.zeros_like(block, dtype=bool)
                
            if ignore_value != -9999.0: # Check if explicit ignore value is provided/different
                 mask |= (block == ignore_value)
            # Also handle the default ignore_value if it's not the nodata
            elif ignore_value == -9999.0 and nodata != -9999.0:
                 mask |= (block == ignore_value)

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
    Input:
        dsm_path: {string} path to the DSM raster
        radius: {int} Search radius of kernel in cells (used for morphological filter).
        terrain_slope: {float} average slope of the input terrain (currently unused but kept for compatibility).
    Output:
        ground_dem_path: {string} path to the generated ground DEM (GeoTIFF) raster
        non_ground_dem_path: {string} path to the generated non-ground DEM (GeoTIFF) raster
    """
    with rasterio.open(dsm_path) as src:
        profile = src.profile
        nodata = src.nodata
        
        # Update profile for output rasters
        profile.update(dtype=src.dtype, count=1, driver='GTiff') # Maintain original dtype

        # Create overlapping structure element once
        struct_element = np.ones((2 * radius + 1, 2 * radius + 1))

        with rasterio.open(ground_dem_path, 'w', **profile) as dst_ground, \
             rasterio.open(non_ground_dem_path, 'w', **profile) as dst_non_ground:
            
            overlap = radius # Overlap must cover the kernel radius
            
            for ji, window in dst_ground.block_windows(1):
                # Calculate read window with padding
                read_window = rasterio.windows.union(
                    src.window(*src.window_bounds(window)),
                    window
                )
                read_window = rasterio.windows.Window(
                    window.col_off - overlap,
                    window.row_off - overlap,
                    window.width + 2 * overlap,
                    window.height + 2 * overlap
                )
                # Clip to dataset bounds
                read_window = read_window.intersection(rasterio.windows.Window(0, 0, src.width, src.height))

                # Read data
                dsm_block = src.read(1, window=read_window)

                # Handle nodata
                if nodata is not None:
                    # Fill nodata with a very low value for opening (erosion then dilation)
                    # Ideally, should be handled more gracefully, but for opening, 
                    # minimum value is safe if it's below any valid data.
                    valid_mask = (dsm_block != nodata)
                    if np.any(valid_mask):
                        fill_value = np.min(dsm_block[valid_mask])
                    else:
                        fill_value = -99999.0 # arbitrary low value if all nodata
                        
                    dsm_filled = np.where(dsm_block == nodata, fill_value, dsm_block)
                else:
                    dsm_filled = dsm_block

                # Apply morphological opening
                ground_block = grey_opening(dsm_filled, structure=struct_element)

                # Restore nodata
                if nodata is not None:
                    ground_block[dsm_block == nodata] = nodata
                
                # Calculate non-ground
                non_ground_block = dsm_block - ground_block
                if nodata is not None:
                    non_ground_block[dsm_block == nodata] = nodata

                # Crop to output window
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
    Removes noise (high elevation data points like roofs, etc.) from the ground DEM raster.
    Replaces values in those pixels with No data Value (`ignore_value`).
    """
    with rasterio.open(ground_dem_path) as src:
        ground_np = src.read(1, masked=True) # Read as masked array to handle nodata
        
    # Calculate std and mean, ignoring nodata values
    valid_values = ground_np[~ground_np.mask]

    if valid_values.size == 0:
        # If no valid data, just write an empty raster or handle as appropriate
        # For now, let's return without modification, or create an empty raster
        # For now, create an empty raster with nodata
        profile = src.profile
        profile.update(nodata=ignore_value)
        with rasterio.open(out_path, 'w', **profile) as dst:
            dst.write(np.full(src.shape, ignore_value, dtype=src.dtype), 1)
        return

    std = valid_values.std()
    mean = valid_values.mean()
    threshold_value = mean + 1.5 * std
    
    # Create a copy to modify without affecting the original masked array
    modified_ground_np = ground_np.filled(fill_value=ignore_value) # Fill masked areas with ignore_value

    # Apply thresholding
    modified_ground_np[modified_ground_np >= threshold_value] = ignore_value
    
    save_array_as_geotif(modified_ground_np, ground_dem_path, out_path, nodata_value=ignore_value)


def save_array_as_geotif(array, source_tif_path, out_path, nodata_value=None):
    """
    Generates a geotiff raster from the input numpy array.
    """
    with rasterio.open(source_tif_path) as src_profile:
        profile = src_profile.profile

    # Update profile with array's dtype and nodata
    profile.update(
        dtype=array.dtype,
        count=1, # Assuming single band output
        nodata=nodata_value if nodata_value is not None else profile.get('nodata')
    )

    with rasterio.open(out_path, 'w', **profile) as dst:
        dst.write(array, 1)


def close_gaps(in_path, out_path, max_search_distance=10.0, smoothing_iterations=0, nodata_value=None):
    """
    Interpolates the holes (no data value) in the input raster using rasterio.fill.fillnodata.
    Input:
        in_path: {string} path to the input raster with holes
        max_search_distance: {float} The maximum distance in pixels to search for valid pixels.
        smoothing_iterations: {int} The number of smoothing iterations to apply.
    Output:
        out_path: {string} path to the generated raster with closed holes (GeoTIFF).
    """
    with rasterio.open(in_path) as src:
        in_array = src.read(1)
        profile = src.profile
        nodata = src.nodata if nodata_value is None else nodata_value

    # Create a mask for nodata values
    mask = in_array == nodata

    # Fill nodata values
    filled_array = fillnodata(
        in_array,
        mask=mask,
        max_search_distance=max_search_distance,
        smoothing_iterations=smoothing_iterations
    )

    profile.update(dtype=filled_array.dtype, count=1, nodata=nodata, driver='GTiff')

    with rasterio.open(out_path, 'w', **profile) as dst:
        dst.write(filled_array, 1)


def smoothen_raster(in_path, out_path, radius=2, nodata_value=None):
    """
    Applies gaussian filter to the input raster using scipy.ndimage.
    Input:
        in_path: {string} path to the input raster
        radius: {int} kernel radius to be used for smoothing (standard deviation for Gaussian).
    Output:
        out_path: {string} path to the generated smoothened raster (GeoTIFF).
    """
    with rasterio.open(in_path) as src:
        in_array = src.read(1)
        profile = src.profile
        nodata = src.nodata if nodata_value is None else nodata_value

    # Handle nodata values
    if nodata is not None:
        # Fill nodata values for smoothing, then re-mask
        # Using a conservative fill, e.g., mean of valid data or a constant
        if np.any(in_array != nodata):
            fill_value = np.mean(in_array[in_array != nodata])
        else:
            fill_value = 0.0 # Default if all are nodata

        temp_array = np.where(in_array == nodata, fill_value, in_array)
    else:
        temp_array = in_array

    # Apply Gaussian filter
    smoothed_array = gaussian_filter(temp_array, sigma=radius)

    # Restore nodata values
    if nodata is not None:
        smoothed_array[in_array == nodata] = nodata

    profile.update(dtype=smoothed_array.dtype, count=1, nodata=nodata, driver='GTiff')

    with rasterio.open(out_path, 'w', **profile) as dst:
        dst.write(smoothed_array, 1)


def subtract_rasters(rasterA_path, rasterB_path, out_path, no_data_value=None):
    """
    Subtracts rasterB from rasterA using numpy, saving the result as a GeoTIFF.
    """
    with rasterio.open(rasterA_path) as srcA:
        arrayA = srcA.read(1)
        profile = srcA.profile
        nodataA = srcA.nodata if no_data_value is None else no_data_value

    with rasterio.open(rasterB_path) as srcB:
        arrayB = srcB.read(1)
        nodataB = srcB.nodata

    # Perform subtraction
    result_array = arrayA - arrayB

    # Combine nodata masks and apply to result
    # Consider nodata from either input as nodata in output
    nodata_mask = (arrayA == nodataA) | (arrayB == nodataB)
    if nodataA is not None:
        result_array[nodata_mask] = nodataA

    profile.update(dtype=result_array.dtype, count=1, nodata=nodataA, driver='GTiff')

    with rasterio.open(out_path, 'w', **profile) as dst:
        dst.write(result_array, 1)


def replace_values(
    rasterA_path, rasterB_path, out_path, no_data_value=None, threshold=0.98
):
    """
    Replaces values in input rasterA with no_data_value where cell value >= threshold in rasterB.
    """
    with rasterio.open(rasterA_path) as srcA:
        arrayA = srcA.read(1)
        profile = srcA.profile
        nodataA = srcA.nodata if no_data_value is None else no_data_value

    with rasterio.open(rasterB_path) as srcB:
        arrayB = srcB.read(1)

    # Apply replacement logic using numpy where
    # Where arrayB >= threshold, use nodataA, else use arrayA
    result_array = np.where(arrayB >= threshold, nodataA, arrayA)
    
    # Ensure nodata values from original rasterA are propagated
    result_array[arrayA == nodataA] = nodataA

    profile.update(dtype=result_array.dtype, count=1, nodata=nodataA, driver='GTiff')

    with rasterio.open(out_path, 'w', **profile) as dst:
        dst.write(result_array, 1)


def expand_holes_in_raster(
    in_path, search_window=7, no_data_value=-99999.0, threshold=50
):
    """
    Expands holes (cells with no_data_value) in the input raster using numpy.
    Input:
        in_path: {string} path to the input raster
        search_window: {int} kernel size to be used as window
        threshold: {float} threshold on percentage of cells with no_data_value
    Output:
        np_raster: {numpy array} Returns the modified input raster's array
    """
    with rasterio.open(in_path) as src:
        np_raster = src.read(1)
        nodata = src.nodata if src.nodata is not None else no_data_value
    
    # Ensure nodata is consistent for operations
    np_raster_modified = np_raster.copy()
    height, width = np_raster.shape
    
    # Iterate through the array with a sliding window
    # This can be optimized with more advanced numpy/scipy operations
    half_window = (search_window - 1) // 2

    # Create a padded array to handle windowing at edges without conditional logic inside loop
    padded_raster = np.pad(np_raster_modified, half_window, mode='edge')
    
    for i in range(height):
        for j in range(width):
            window = padded_raster[i:i + search_window, j:j + search_window]
            
            # Calculate percentage of nodata values in the window
            nodata_count = np.count_nonzero(window == nodata)
            total_cells = search_window ** 2
            
            if (nodata_count / total_cells) * 100 >= threshold:
                np_raster_modified[i, j] = nodata
                
    return np_raster_modified


def get_raster_crs(raster_path):
    """
    Returns the CRS (Coordinate Reference System) of the raster using rasterio.
    """
    with rasterio.open(raster_path) as src:
        return src.crs.to_epsg() # Return EPSG code for consistency with old behavior


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

    downsampled_dsm_path = dsm_path # Initialize with original path

    # Use a common reference resolution (e.g., 30cm or similar for projected, degrees for geographic)
    # The original logic used gdal.Open(dsm_path).GetGeoTransform()[1] directly which is problematic
    # as it implies downsampling factor is directly tied to target_res / current_x_res.
    # Let's simplify for initial refactor and assume downsampling is only applied if original is too high-res.

    current_x_res, current_y_res = get_raster_resolution(dsm_path)

    if dsm_crs != 4326: # Projected CRS
        if current_x_res < 0.3 or current_y_res < 0.3: # If resolution is finer than 30cm
            target_res = 0.3
            # Calculate scale factor for rasterio
            x_scale = current_x_res / target_res
            y_scale = current_y_res / target_res
            
            downsampled_dsm_path = os.path.join(temp_dir, dsm_name + "_ds.tif")
            with rasterio.open(dsm_path) as src:
                profile = src.profile
                # Calculate new dimensions based on scale
                new_height = int(src.height / y_scale)
                new_width = int(src.width / x_scale)
                
                # Read and resample
                resampled_data = src.read(
                    out_shape=(src.count, new_height, new_width),
                    resampling=Resampling.bilinear
                )
                # Update profile for resampled data
                profile.update(
                    transform=src.transform * src.transform.scale(1/x_scale, 1/y_scale),
                    width=new_width,
                    height=new_height
                )
                with rasterio.open(downsampled_dsm_path, 'w', **profile) as dst:
                    dst.write(resampled_data)
            dsm_path = downsampled_dsm_path
    else: # Geographic CRS (WGS84, etc.)
        # Example: if resolution is finer than ~2.5e-6 degrees (approx 30cm at equator)
        if current_x_res < 2.514e-06 or current_y_res < 2.514e-06:
            target_res = 2.514e-06
            x_scale = current_x_res / target_res
            y_scale = current_y_res / target_res

            downsampled_dsm_path = os.path.join(temp_dir, dsm_name + "_ds.tif")
            with rasterio.open(dsm_path) as src:
                profile = src.profile
                new_height = int(src.height / y_scale)
                new_width = int(src.width / x_scale)
                
                resampled_data = src.read(
                    out_shape=(src.count, new_height, new_width),
                    resampling=Resampling.bilinear
                )
                profile.update(
                    transform=src.transform * src.transform.scale(1/x_scale, 1/y_scale),
                    width=new_width,
                    height=new_height
                )
                with rasterio.open(downsampled_dsm_path, 'w', **profile) as dst:
                    dst.write(resampled_data)
            dsm_path = downsampled_dsm_path

    return dsm_path


def get_updated_params(dsm_path, search_radius, smoothen_radius):
    # search_radius and smoothen_radius are set wrt to 30cm DSM
    # returns updated parameters if DSM is of coarser resolution
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
    
    original_dsm_path = dsm_path # Keep track of the original DSM path
    dsm_path = get_res_and_downsample(dsm_path, temp_dir)
    
    # get updated params wrt to DSM resolution
    search_radius, smoothen_radius = get_updated_params(
        dsm_path, search_radius, smoothen_radius
    )
    
    # Generate DTM
    # STEP 1: Generate slope raster from dsm to get average slope value
    dsm_name = os.path.splitext(os.path.basename(dsm_path))[0]
    dsm_slp_path = os.path.join(temp_dir, dsm_name + "_slp.tif")
    generate_slope_raster(dsm_path, dsm_slp_path)
    avg_slp = int(get_mean(dsm_slp_path))
    
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
    subtract_rasters(ground_dem_path, smoothened_ground_path, diff_raster_path)
    
    # STEP 5: Thresholding on the difference raster to replace values in Ground DEM by no-data values (threshold = 0.98)
    thresholded_ground_path = os.path.join(
        temp_dir, dsm_name + "_ground_thresholded.tif"
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
        temp_dir, dsm_name + "_ground_bigger_holes.tif"
    )
    temp_array = expand_holes_in_raster(ground_dem_filtered_path)
    save_array_as_geotif(temp_array, ground_dem_filtered_path, bigger_holes_ground_path)
    
    # STEP 8: Close gaps in the DTM
    dtm_final_temp_path = os.path.join(temp_dir, dsm_name + "_dtm_final.tif")
    close_gaps(bigger_holes_ground_path, dtm_final_temp_path)
    
    # STEP 9: Final output DTM to the specified out_dir
    dtm_tif_path = os.path.join(out_dir, os.path.splitext(os.path.basename(original_dsm_path))[0] + "_dtm.tif")

    # We need to copy or translate the final temporary DTM to its permanent location
    # For simplicity, let's just rename/move it for now if it's the last step.
    # A robust solution might involve resampling back to original resolution/extent if downsampled initially.

    # For now, simply move the final temp DTM to the expected output path
    # This assumes no upsampling or further transformation to original DSM extent/resolution is needed at the end.
    # If original DSM was downsampled, ideally we'd upsample the DTM back to original resolution here.

    # Let's copy the content to handle cases where dtm_final_temp_path is not directly movable
    with rasterio.open(dtm_final_temp_path) as src_dtm:
        dtm_array = src_dtm.read()
        dtm_profile = src_dtm.profile

    with rasterio.open(dtm_tif_path, 'w', **dtm_profile) as dst_dtm:
        dst_dtm.write(dtm_array)

    return dtm_tif_path


# -----------------------------------------------------------------------------------------------------
def main_cli():
    """
    Command line interface for generating DTM from DSM.
    """
    parser = argparse.ArgumentParser(description="Generate DTM from DSM")
    parser.add_argument("--dsm", help="Path to the DSM file", required=True)
    parser.add_argument(
        "--out_dir", help="Directory to save the output DTM", default="generated_dtm"
    )
    args = parser.parse_args()
    dtm_path = main(args.dsm, args.out_dir)
    print(f"######### DTM generated at: {dtm_path}")


if __name__ == "__main__":
    main_cli()