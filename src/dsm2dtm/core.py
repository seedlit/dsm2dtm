"""
dsm2dtm - Generate DTM (Digital Terrain Model) from DSM (Digital Surface Model)
Author: Naman Jain
        naman.jain@btech2015.iitgn.ac.in
"""

import argparse
import os
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import rasterio
from numpy.typing import NDArray
from rasterio.crs import CRS
from rasterio.fill import fillnodata
from rasterio.transform import Affine
from rasterio.warp import Resampling, calculate_default_transform, reproject
from scipy.ndimage import gaussian_filter, grey_opening, zoom

from dsm2dtm.constants import (
    DEFAULT_KERNEL_RADIUS_METERS,
    DEFAULT_NODATA,
    FINAL_SMOOTH_SIGMA_METERS,
    GAP_FILL_MAX_SEARCH_DISTANCE_METERS,
    GAP_FILL_SMOOTHING_ITERATIONS,
    MIN_PROCESSING_RESOLUTION_METERS,
    PMF_INITIAL_THRESHOLD,
    PMF_INITIAL_WINDOW_METERS,
    PMF_MAX_THRESHOLD,
    PMF_MAX_WINDOW_METERS,
    PMF_SLOPE,
    REFINEMENT_ELEVATION_THRESHOLD,
    REFINEMENT_SMOOTH_SIGMA_METERS,
)
from dsm2dtm.utm_utils import estimate_utm_crs


@dataclass
class AdaptiveParameters:
    """Parameters adapted to the specific DSM resolution."""

    pmf_initial_window: int
    pmf_max_window: int
    pmf_slope: float
    refinement_smooth_sigma: float
    final_smooth_sigma: float
    gap_fill_search_dist: float


@dataclass
class DSMContext:
    """Holds DSM data and metadata, including reprojection details."""

    dsm: NDArray[np.floating]
    profile: dict
    nodata: float
    resolution: Tuple[float, float]
    is_reprojected: bool = False
    original_crs: Optional[CRS] = None
    original_transform: Optional[Affine] = None
    original_shape: Optional[Tuple[int, int]] = None  # (height, width)
    utm_crs: Optional[CRS] = None
    utm_transform: Optional[Affine] = None


def calculate_terrain_slope(dsm: NDArray[np.floating], resolution: float, nodata: float) -> float:
    """
    Calculate the median terrain slope (rise/run) of the DSM.

    This function estimates the general slope of the terrain to automatically tune the
    Progressive Morphological Filter (PMF) slope parameter. It calculates the gradient magnitude
    of the DSM and takes the median of the valid slopes.

    Args:
        dsm (NDArray[np.floating]): The input Digital Surface Model (DSM) as a 2D numpy array.
        resolution (float): The spatial resolution of the DSM (in meters per pixel).
        nodata (float): The value representing no data in the DSM.

    Returns:
        float: The calculated median slope of the terrain (clamped between 0.01 and 1.0).
    """
    # Decimate if resolution is very fine (e.g. < 0.5m)
    # Target ~1.0m resolution for slope estimation
    target_res = 1.0
    current_res = max(resolution, 0.001)

    if current_res < (target_res * 0.5):
        scale_factor = current_res / target_res
        dsm_for_slope = zoom(dsm, scale_factor, order=1)
        res_for_slope = target_res
    else:
        dsm_for_slope = dsm
        res_for_slope = current_res

    valid_mask = dsm_for_slope != nodata
    if not np.any(valid_mask):
        return PMF_SLOPE

    # Check for sufficient size for gradient calculation
    if dsm_for_slope.shape[0] < 2 or dsm_for_slope.shape[1] < 2:
        return PMF_SLOPE

    dy, dx = np.gradient(dsm_for_slope)
    slope_per_pixel = np.sqrt(dy**2 + dx**2)
    slope_dimensionless = slope_per_pixel / res_for_slope
    valid_slopes = slope_dimensionless[valid_mask]
    if len(valid_slopes) == 0:
        return PMF_SLOPE
    # Use Median instead of Mean for robustness against outliers (vertical walls)
    median_slope = np.nanmedian(valid_slopes)
    # Clamp to reasonable bounds
    median_slope = max(0.01, min(median_slope, 1.0))
    return float(median_slope)


def get_adaptive_parameters(
    resolution: float, max_image_dimension: int = 10000, base_slope: float = PMF_SLOPE
) -> AdaptiveParameters:
    """
    Calculate algorithm parameters based on input resolution in meters.

    This function scales various parameters (like window sizes and search distances)
    based on the resolution of the input data to ensure consistent behavior across
    datasets with different resolutions.

    Args:
        resolution (float): The spatial resolution of the DSM (in meters per pixel).
        max_image_dimension (int, optional): The maximum dimension of the image to clamp window sizes.
            Defaults to 10000.
        base_slope (float, optional): The base slope value to use for PMF. Defaults to PMF_SLOPE.

    Returns:
        AdaptiveParameters: A data class containing the adapted parameters.
    """
    # Avoid division by zero
    res_meters = max(resolution, 0.001)
    # 1. PMF Window Sizes (Pixels)
    init_window = int(PMF_INITIAL_WINDOW_METERS / res_meters)
    if init_window % 2 == 0:
        init_window += 1
    init_window = max(3, init_window)
    max_window_target = int(PMF_MAX_WINDOW_METERS / res_meters)

    limit = max_image_dimension
    if limit % 2 == 0:
        limit -= 1

    max_window = min(max_window_target, limit)
    if max_window % 2 == 0:
        max_window -= 1
    max_window = max(init_window, max_window)
    pmf_slope = base_slope * res_meters
    refine_sigma = REFINEMENT_SMOOTH_SIGMA_METERS / res_meters
    final_sigma = FINAL_SMOOTH_SIGMA_METERS / res_meters
    gap_dist = GAP_FILL_MAX_SEARCH_DISTANCE_METERS / res_meters

    return AdaptiveParameters(
        pmf_initial_window=init_window,
        pmf_max_window=max_window,
        pmf_slope=pmf_slope,
        refinement_smooth_sigma=refine_sigma,
        final_smooth_sigma=final_sigma,
        gap_fill_search_dist=gap_dist,
    )


def progressive_morphological_filter(
    surface: NDArray[np.floating],
    nodata: float,
    initial_window: int,
    max_window: int,
    slope: float,
    initial_threshold: float = PMF_INITIAL_THRESHOLD,
    max_threshold: float = PMF_MAX_THRESHOLD,
) -> NDArray[np.floating]:
    """
    Apply the Progressive Morphological Filter (PMF) to separate ground from non-ground points.

    PMF works by iteratively opening the surface with increasing window sizes. At each step,
    pixels that rise above the opened surface by more than a dynamic threshold are classified as non-ground.

    Args:
        surface (NDArray[np.floating]): The input DSM surface array.
        nodata (float): The nodata value.
        initial_window (int): The initial size of the morphological window (in pixels).
        max_window (int): The maximum size of the morphological window (in pixels).
        slope (float): The slope parameter used to calculate the elevation threshold.
        initial_threshold (float, optional): The initial elevation difference threshold.
            Defaults to PMF_INITIAL_THRESHOLD.
        max_threshold (float, optional): The maximum elevation difference threshold. Defaults to PMF_MAX_THRESHOLD.

    Returns:
        NDArray[np.floating]: An array representing the estimated ground surface,
            with non-ground points replaced by the opened surface value.
    """
    valid_mask = surface != nodata
    if not np.any(valid_mask):
        return surface.copy()
    min_val = np.min(surface[valid_mask])
    working = np.where(valid_mask, surface, min_val)
    # Use adapted initial window size
    window_size = initial_window
    while window_size <= max_window:
        window_radius = (window_size - 1) // 2
        dh_threshold = min(initial_threshold + slope * window_radius, max_threshold)

        struct = np.ones((window_size, window_size))
        opened = grey_opening(working, footprint=struct)

        diff = working - opened
        non_ground_mask = diff > dh_threshold
        working[non_ground_mask] = opened[non_ground_mask]

        window_size = 2 * window_size - 1
        # Stop if we exceed max window
        if window_size > max_window:
            break

    return np.where(valid_mask, working, nodata)


def refine_ground_surface(
    ground: NDArray[np.floating],
    nodata: float,
    smoothen_radius: float,
    elevation_threshold: float = REFINEMENT_ELEVATION_THRESHOLD,
) -> NDArray[np.floating]:
    """
    Refine the ground surface by removing points that deviate significantly from a smoothed version.

    This step helps to remove residual non-ground objects (like small spikes) that might have been
    missed by the PMF. It compares the current ground estimate with a Gaussian-smoothed version
    and removes points that are higher than the smoothed surface by a specified threshold.

    Args:
        ground (NDArray[np.floating]): The estimated ground surface array.
        nodata (float): The nodata value.
        smoothen_radius (float): The sigma (standard deviation) for the Gaussian smoothing kernel (in pixels).
        elevation_threshold (float, optional): The elevation difference threshold for outlier removal.
            Defaults to REFINEMENT_ELEVATION_THRESHOLD.

    Returns:
        NDArray[np.floating]: The refined ground surface array with outliers marked as nodata.
    """
    valid_mask = ground != nodata
    if not np.any(valid_mask):
        return ground.copy()
    min_val = np.min(ground[valid_mask])
    smooth_input = np.where(ground == nodata, min_val, ground)
    smoothed = gaussian_filter(smooth_input, sigma=smoothen_radius)
    diff = ground - smoothed
    refined = ground.copy()
    refined[(diff >= elevation_threshold) & valid_mask] = nodata
    return refined


def _process_coarse_dsm(
    dsm: NDArray[np.floating],
    cell_size: float,
    nodata: float,
    kernel_radius_meters: Optional[float],
    slope: Optional[float],
    initial_threshold: float,
    max_threshold: float,
) -> NDArray[np.floating]:
    """
    Downsamples high-resolution DSM, processes at coarse resolution, and upsamples the result.

    This function is a helper for `dsm_to_dtm` to handle very high-resolution data efficiently
    and stably. It downsamples the input DSM to a coarser resolution, runs the DTM generation
    process on the coarse data, and then upsamples the result back to the original resolution.

    Args:
        dsm (NDArray[np.floating]): The high-resolution input DSM.
        cell_size (float): The pixel size of the input DSM (in meters).
        nodata (float): The nodata value.
        kernel_radius_meters (Optional[float]): The radius for the PMF window in meters.
        slope (Optional[float]): The terrain slope parameter.
        initial_threshold (float): The initial elevation threshold for PMF.
        max_threshold (float): The maximum elevation threshold for PMF.

    Returns:
        NDArray[np.floating]: The generated DTM at the original resolution.
    """
    target_res = MIN_PROCESSING_RESOLUTION_METERS

    # Calculate new dimensions
    h, w = dsm.shape
    scale = cell_size / target_res
    new_h = int(h * scale)
    new_w = int(w * scale)

    # Guard against over-reduction for small chips
    if new_h < 10 or new_w < 10:
        # Too small to downsample, process as is
        return _process_standard_dsm(
            dsm, cell_size, nodata, kernel_radius_meters, slope, initial_threshold, max_threshold
        )

    print(
        f"High resolution input ({cell_size:.4f}m). "
        f"Downsampling to {MIN_PROCESSING_RESOLUTION_METERS}m for processing stability..."
    )

    # Transforms (Dummy, 0,0 origin)
    src_transform = Affine.scale(cell_size, cell_size)
    dst_transform = Affine.scale(target_res, target_res)
    dummy_crs = CRS.from_epsg(3857)

    dsm_coarse = np.empty((new_h, new_w), dtype=np.float32)

    reproject(
        source=dsm,
        destination=dsm_coarse,
        src_transform=src_transform,
        src_crs=dummy_crs,
        dst_transform=dst_transform,
        dst_crs=dummy_crs,
        resampling=Resampling.bilinear,
        src_nodata=nodata,
        dst_nodata=nodata,
    )

    # Recursive call with coarser resolution
    # We pass slope=None to let it re-calculate/adapt to coarse DTM
    dtm_coarse = dsm_to_dtm(
        dsm_coarse,
        (target_res, target_res),
        kernel_radius_meters=kernel_radius_meters,
        slope=slope,
        initial_threshold=initial_threshold,
        max_threshold=max_threshold,
        nodata=nodata,
    )

    # Upsample Result back to original size
    dtm_fine = np.empty((h, w), dtype=np.float32)

    reproject(
        source=dtm_coarse,
        destination=dtm_fine,
        src_transform=dst_transform,  # Coarse
        src_crs=dummy_crs,
        dst_transform=src_transform,  # Original
        dst_crs=dummy_crs,
        resampling=Resampling.bilinear,
        src_nodata=nodata,
        dst_nodata=nodata,
    )

    return dtm_fine


def _process_standard_dsm(
    dsm: NDArray[np.floating],
    cell_size: float,
    nodata: float,
    kernel_radius_meters: Optional[float],
    slope: Optional[float],
    initial_threshold: float,
    max_threshold: float,
) -> NDArray[np.floating]:
    """
    Standard DTM generation pipeline: Parameters -> PMF -> Refine -> Smooth -> GapFill.

    This pipeline consists of:
    1. Parameter adaptation based on resolution.
    2. Progressive Morphological Filtering (PMF) to identify ground points.
    3. Refinement to remove outliers based on local smoothing.
    4. Final light Gaussian smoothing.
    5. Gap interpolation to fill nodata areas.

    Args:
        dsm (NDArray[np.floating]): The input DSM array.
        cell_size (float): The pixel size (in meters).
        nodata (float): The nodata value.
        kernel_radius_meters (Optional[float]): User-specified kernel radius in meters.
        slope (Optional[float]): User-specified terrain slope.
        initial_threshold (float): Initial elevation threshold for PMF.
        max_threshold (float): Maximum elevation threshold for PMF.

    Returns:
        NDArray[np.floating]: The resulting DTM array.
    """
    # Calculate Slope dynamically if not provided
    if slope is None:
        slope = calculate_terrain_slope(dsm, cell_size, nodata)

    # Get max image dimension for clamping
    max_dim = max(dsm.shape[0], dsm.shape[1])

    # Adapt parameters to current resolution and image size
    params = get_adaptive_parameters(cell_size, max_image_dimension=max_dim, base_slope=slope)

    # Override max window if kernel_radius_meters provided
    if kernel_radius_meters is not None:
        if cell_size < 0.01:
            # Fallback for un-projected Degree data (Lat/Lon).
            res_meters = cell_size * 111320.0
        else:
            res_meters = cell_size
        res_meters = max(res_meters, 0.001)

        max_window_px = int(kernel_radius_meters / res_meters) * 2 + 1
        limit = max_dim
        if limit % 2 == 0:
            limit -= 1
        max_window_px = min(max_window_px, limit)

        params.pmf_max_window = max(max_window_px, params.pmf_initial_window)

    # Step 1: Progressive morphological filtering
    ground = progressive_morphological_filter(
        dsm,
        nodata=nodata,
        initial_window=params.pmf_initial_window,
        max_window=params.pmf_max_window,
        slope=params.pmf_slope,
        initial_threshold=initial_threshold,
        max_threshold=max_threshold,
    )

    # Step 2: Smooth-based refinement to remove remaining outliers
    ground = refine_ground_surface(
        ground,
        nodata=nodata,
        smoothen_radius=params.refinement_smooth_sigma,
        elevation_threshold=REFINEMENT_ELEVATION_THRESHOLD,
    )

    # Step 3: Light gaussian smoothing
    valid_mask = ground != nodata
    if np.any(valid_mask):
        min_val = np.min(ground[valid_mask])
        smooth_input = np.where(ground == nodata, min_val, ground)
        smoothed = gaussian_filter(smooth_input, sigma=params.final_smooth_sigma)
        ground = np.where(valid_mask, smoothed, nodata)

    # Step 4: Gap interpolation
    mask = (ground != nodata).astype(np.uint8)
    dtm = ground.copy().astype(np.float32)
    fillnodata(
        dtm,
        mask=mask,
        max_search_distance=params.gap_fill_search_dist,
        smoothing_iterations=GAP_FILL_SMOOTHING_ITERATIONS,
    )

    return dtm


def dsm_to_dtm(
    dsm: NDArray[np.floating],
    resolution: Tuple[float, float],
    kernel_radius_meters: Optional[float] = None,
    slope: Optional[float] = None,
    initial_threshold: float = PMF_INITIAL_THRESHOLD,
    max_threshold: float = PMF_MAX_THRESHOLD,
    nodata: float = DEFAULT_NODATA,
) -> NDArray[np.floating]:
    """
    Generate DTM from DSM using Progressive Morphological Filter with refinement.
    Handles high-resolution data by optionally processing at a coarser scale.

    Args:
        dsm (NDArray[np.floating]): The input DSM array.
        resolution (Tuple[float, float]): The resolution of the DSM (x_res, y_res).
        kernel_radius_meters (Optional[float], optional): The maximum window radius for filtering in meters.
            Defaults to None.
        slope (Optional[float], optional): The terrain slope. If None, it is calculated automatically.
            Defaults to None.
        initial_threshold (float, optional): The initial elevation threshold for PMF.
            Defaults to PMF_INITIAL_THRESHOLD.
        max_threshold (float, optional): The maximum elevation threshold for PMF. Defaults to PMF_MAX_THRESHOLD.
        nodata (float, optional): The nodata value. Defaults to DEFAULT_NODATA.

    Returns:
        NDArray[np.floating]: The generated DTM as a numpy array.
    """
    # Calculate cell size (degrees or meters)
    cell_size = (abs(resolution[0]) + abs(resolution[1])) / 2.0
    cell_size = max(cell_size, 0.001)  # Avoid zero

    # Check if we should process at a coarser resolution
    if cell_size < (MIN_PROCESSING_RESOLUTION_METERS * 0.9):
        return _process_coarse_dsm(
            dsm, cell_size, nodata, kernel_radius_meters, slope, initial_threshold, max_threshold
        )

    # --- Standard Processing (Full Resolution) ---
    return _process_standard_dsm(dsm, cell_size, nodata, kernel_radius_meters, slope, initial_threshold, max_threshold)


def _load_dsm(dsm_path: str) -> DSMContext:
    """
    Loads the DSM file. If Geographic, reprojects to UTM in memory.

    Args:
        dsm_path (str): The file path to the DSM.

    Returns:
        DSMContext: A context object containing the loaded DSM data and metadata.
    """
    with rasterio.open(dsm_path) as src:
        is_geographic = src.crs and src.crs.is_geographic
        nodata = src.nodata if src.nodata is not None else DEFAULT_NODATA

        if not is_geographic:
            return DSMContext(
                dsm=src.read(1),
                profile=src.profile,
                nodata=nodata,
                resolution=src.res,
                is_reprojected=False,
                original_shape=(src.height, src.width),
            )

        print(f"Input is Geographic ({src.crs}). Reprojecting to UTM for processing...")
        left, bottom, right, top = src.bounds
        center_lon = (left + right) / 2
        center_lat = (bottom + top) / 2
        utm_crs = estimate_utm_crs(center_lon, center_lat)
        print(f"Selected CRS: {utm_crs}")

        transform, width, height = calculate_default_transform(src.crs, utm_crs, src.width, src.height, *src.bounds)

        dsm_utm = np.zeros((height, width), dtype=np.float32)
        reproject(
            source=rasterio.band(src, 1),
            destination=dsm_utm,
            src_transform=src.transform,
            src_crs=src.crs,
            dst_transform=transform,
            dst_crs=utm_crs,
            resampling=Resampling.bilinear,
            dst_nodata=nodata,
        )

        profile = src.profile.copy()
        profile.update(
            {
                "crs": utm_crs,
                "transform": transform,
                "width": width,
                "height": height,
                "nodata": nodata,
            }
        )

        return DSMContext(
            dsm=dsm_utm,
            profile=profile,
            nodata=nodata,
            resolution=(transform[0], -transform[4]),
            is_reprojected=True,
            original_crs=src.crs,
            original_transform=src.transform,
            original_shape=(src.height, src.width),
            utm_crs=utm_crs,
            utm_transform=transform,
        )


def _save_dtm(dtm: NDArray[np.floating], output_path: str, context: DSMContext) -> None:
    """
    Saves the DTM. Reprojects back to original CRS if the input was reprojected.

    Args:
        dtm (NDArray[np.floating]): The generated DTM array.
        output_path (str): The file path where the DTM will be saved.
        context (DSMContext): The context object containing original metadata and CRS info.
    """
    if not context.is_reprojected:
        profile = context.profile.copy()
        profile.update(dtype=dtm.dtype, nodata=context.nodata)
        with rasterio.open(output_path, "w", **profile) as dst:
            dst.write(dtm, 1)
        return

    print("Reprojecting DTM back to original CRS...")

    # Destination array (Original dimensions)
    orig_h, orig_w = context.original_shape
    dtm_out = np.zeros((orig_h, orig_w), dtype=dtm.dtype)

    reproject(
        source=dtm,
        destination=dtm_out,
        src_transform=context.utm_transform,
        src_crs=context.utm_crs,
        dst_transform=context.original_transform,
        dst_crs=context.original_crs,
        resampling=Resampling.bilinear,
        src_nodata=context.nodata,
        dst_nodata=context.nodata,
    )

    out_profile = {
        "driver": "GTiff",  # Default
        "dtype": dtm.dtype,
        "nodata": context.nodata,
        "width": orig_w,
        "height": orig_h,
        "count": 1,
        "crs": context.original_crs,
        "transform": context.original_transform,
    }

    with rasterio.open(output_path, "w", **out_profile) as dst:
        dst.write(dtm_out, 1)


def main(
    dsm_path: str,
    out_dir: str,
    kernel_radius_meters: float = DEFAULT_KERNEL_RADIUS_METERS,
    slope: Optional[float] = None,
    initial_threshold: float = PMF_INITIAL_THRESHOLD,
    max_threshold: float = PMF_MAX_THRESHOLD,
) -> str:
    """
    Generate DTM from DSM file.
    Automatically handles Geographic CRS by reprojecting to local UTM.

    Args:
        dsm_path (str): Path to the input DSM file.
        out_dir (str): Directory where the output DTM will be saved.
        kernel_radius_meters (float, optional): Kernel radius for PMF in meters.
            Defaults to DEFAULT_KERNEL_RADIUS_METERS.
        slope (Optional[float], optional): Terrain slope. If None, calculated from data. Defaults to None.
        initial_threshold (float, optional): Initial elevation threshold for PMF.
            Defaults to PMF_INITIAL_THRESHOLD.
        max_threshold (float, optional): Max elevation threshold for PMF. Defaults to PMF_MAX_THRESHOLD.

    Returns:
        str: The path to the generated DTM file.
    """
    os.makedirs(out_dir, exist_ok=True)
    # Ensure temp dir exists if used internally
    # Keeping behavior consistent, though explicit temp file usage is reduced.

    output_name = os.path.splitext(os.path.basename(dsm_path))[0] + "_dtm.tif"
    dtm_path = os.path.join(out_dir, output_name)

    # 1. Load and Prepare
    ctx = _load_dsm(dsm_path)

    # 2. Process
    dtm = dsm_to_dtm(
        ctx.dsm,
        ctx.resolution,
        kernel_radius_meters=kernel_radius_meters,
        slope=slope,
        initial_threshold=initial_threshold,
        max_threshold=max_threshold,
        nodata=ctx.nodata,
    )

    # 3. Save
    _save_dtm(dtm, dtm_path, ctx)

    print(f"DTM generated at: {dtm_path}")
    return dtm_path


def main_cli() -> None:
    """Command line interface for generating DTM from DSM."""
    parser = argparse.ArgumentParser(description="Generate DTM from DSM")
    parser.add_argument("--dsm", help="Path to the DSM file", required=True)
    parser.add_argument("--out_dir", help="Directory to save the output DTM", default="generated_dtm")
    parser.add_argument(
        "--radius", type=float, default=DEFAULT_KERNEL_RADIUS_METERS, help="Window radius in meters (default: 40.0)"
    )
    parser.add_argument(
        "--slope", type=float, default=None, help="Terrain slope (0-1). If not provided, computed from DSM."
    )
    parser.add_argument(
        "--init_threshold",
        type=float,
        default=PMF_INITIAL_THRESHOLD,
        help="Initial elevation threshold in meters (default: 0.1)",
    )
    parser.add_argument(
        "--max_threshold",
        type=float,
        default=PMF_MAX_THRESHOLD,
        help="Max elevation threshold in meters (default: 0.5)",
    )
    args = parser.parse_args()

    main(
        args.dsm,
        args.out_dir,
        kernel_radius_meters=args.radius,
        slope=args.slope,
        initial_threshold=args.init_threshold,
        max_threshold=args.max_threshold,
    )


if __name__ == "__main__":
    main_cli()
