"""
dsm2dtm - Generate DTM (Digital Terrain Model) from DSM (Digital Surface Model)
Author: Naman Jain
        naman.jain@btech2015.iitgn.ac.in
"""

import argparse
import os
from dataclasses import dataclass
from typing import Optional

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


def calculate_terrain_slope(dsm: NDArray[np.floating], resolution: float, nodata: float) -> float:
    """
    Calculate the median terrain slope (rise/run) of the DSM.
    Used to automatically tune the PMF slope parameter.
    For high resolution data, we decimate to ~1m to avoid noise/objects dominating the slope.
    """
    # Decimate if resolution is very fine (e.g. < 0.5m)
    # Target ~1.0m resolution for slope estimation
    target_res = 1.0
    current_res = max(resolution, 0.001)

    if current_res < (target_res * 0.5):
        scale_factor = current_res / target_res
        # Zoom using order 0 (nearest) or 1 (bilinear) - 1 is smoother
        # We handle nodata by masking
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

    # Use numpy gradient
    dy, dx = np.gradient(dsm_for_slope)

    # Calculate magnitude of slope per pixel
    slope_per_pixel = np.sqrt(dy**2 + dx**2)

    # Convert to rise/run (dimensionless)
    slope_dimensionless = slope_per_pixel / res_for_slope

    # Filter valid
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

    # 2. PMF Slope (Meters per Pixel)
    # base_slope is rise/run.
    pmf_slope = base_slope * res_meters

    # 3. Smoothing Sigmas (Pixels)
    # We maintain constant physical smoothing size (meters)
    refine_sigma = REFINEMENT_SMOOTH_SIGMA_METERS / res_meters
    final_sigma = FINAL_SMOOTH_SIGMA_METERS / res_meters

    # 4. Gap Fill Distance (Pixels)
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
    Progressive Morphological Filter (PMF).
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
    Refines ground surface by removing points that deviate significantly
    from a smoothed version of the surface.
    """
    valid_mask = ground != nodata

    if not np.any(valid_mask):
        return ground.copy()

    min_val = np.min(ground[valid_mask])

    # Create smoothed surface
    smooth_input = np.where(ground == nodata, min_val, ground)
    smoothed = gaussian_filter(smooth_input, sigma=smoothen_radius)

    # Calculate difference and apply threshold
    diff = ground - smoothed
    refined = ground.copy()
    refined[(diff >= elevation_threshold) & valid_mask] = nodata

    return refined


def dsm_to_dtm(
    dsm: NDArray[np.floating],
    resolution: tuple[float, float],
    kernel_radius_meters: Optional[float] = None,
    slope: Optional[float] = None,
    initial_threshold: float = PMF_INITIAL_THRESHOLD,
    max_threshold: float = PMF_MAX_THRESHOLD,
    nodata: float = DEFAULT_NODATA,
) -> NDArray[np.floating]:
    """
    Generate DTM from DSM using Progressive Morphological Filter with refinement.
    Handles high-resolution data by optionally processing at a coarser scale.
    """
    # Calculate cell size (degrees or meters)
    cell_size = (abs(resolution[0]) + abs(resolution[1])) / 2.0
    cell_size = max(cell_size, 0.001)  # Avoid zero

    # Check if we should process at a coarser resolution
    # If cell_size is significantly smaller than MIN_PROCESSING_RESOLUTION_METERS
    if cell_size < (MIN_PROCESSING_RESOLUTION_METERS * 0.9):
        target_res = MIN_PROCESSING_RESOLUTION_METERS

        # Calculate new dimensions
        h, w = dsm.shape
        scale = cell_size / target_res
        new_h = int(h * scale)
        new_w = int(w * scale)

        # Guard against over-reduction for small chips
        if new_h >= 10 and new_w >= 10:
            print(
                f"High resolution input ({cell_size:.4f}m). "
                f"Downsampling to {MIN_PROCESSING_RESOLUTION_METERS}m for processing stability..."
            )

            # Transforms (Dummy, 0,0 origin)
            src_transform = Affine.scale(cell_size, cell_size)
            dst_transform = Affine.scale(target_res, target_res)

            # Dummy CRS (Arbitrary but valid)
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
                slope=slope,  # Pass slope if user provided, else None
                initial_threshold=initial_threshold,
                max_threshold=max_threshold,
                nodata=nodata,
            )

            # Upsample Result back to original size
            dtm_fine = np.empty((h, w), dtype=np.float32)

            # Upsample using Bilinear (works well for continuous terrain)
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

    # --- Standard Processing (Full Resolution) ---

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
    """
    os.makedirs(out_dir, exist_ok=True)
    temp_dir = os.path.join(out_dir, "temp_files")
    os.makedirs(temp_dir, exist_ok=True)

    output_name = os.path.splitext(os.path.basename(dsm_path))[0] + "_dtm.tif"
    dtm_path = os.path.join(out_dir, output_name)

    with rasterio.open(dsm_path) as src:
        # Check if reprojection is needed
        reproject_needed = src.crs and src.crs.is_geographic

        if reproject_needed:
            print(f"Input is Geographic ({src.crs}). Reprojecting to UTM for processing...")
            # Estimate UTM bucket from center of image
            left, bottom, right, top = src.bounds
            center_lon = (left + right) / 2
            center_lat = (bottom + top) / 2
            utm_crs = estimate_utm_crs(center_lon, center_lat)
            print(f"Selected CRS: {utm_crs}")

            # Calculate transform for UTM
            transform, width, height = calculate_default_transform(src.crs, utm_crs, src.width, src.height, *src.bounds)

            # Metadata for UTM in-memory raster
            kwargs = src.meta.copy()
            kwargs.update(
                {
                    "crs": utm_crs,
                    "transform": transform,
                    "width": width,
                    "height": height,
                    "nodata": src.nodata if src.nodata is not None else DEFAULT_NODATA,
                }
            )

            # Reproject DSM to memory/temp array
            # We can allocation numpy array destination
            dsm_utm = np.zeros((height, width), dtype=np.float32)

            reproject(
                source=rasterio.band(src, 1),
                destination=dsm_utm,
                src_transform=src.transform,
                src_crs=src.crs,
                dst_transform=transform,
                dst_crs=utm_crs,
                resampling=Resampling.bilinear,
                dst_nodata=kwargs["nodata"],
            )

            dsm = dsm_utm
            # Effective resolution in UTM meters
            # x_res = transform[0], y_res = -transform[4] (usually equal)
            resolution = (transform[0], -transform[4])
            nodata = kwargs["nodata"]

        else:
            dsm = src.read(1)
            resolution = src.res
            nodata = src.nodata if src.nodata is not None else DEFAULT_NODATA

    # Generate DTM (Processing happening in Meters now)
    dtm = dsm_to_dtm(
        dsm,
        resolution,
        kernel_radius_meters=kernel_radius_meters,
        slope=slope,
        initial_threshold=initial_threshold,
        max_threshold=max_threshold,
        nodata=nodata,
    )

    if reproject_needed:
        # Reproject DTM back to original CRS
        print("Reprojecting DTM back to original CRS...")
        with rasterio.open(dsm_path) as src:
            src_profile = src.profile.copy()
            src_profile.update(dtype=dtm.dtype, nodata=nodata)

            # We need to reproject dtm (UTM) -> Output (Geo)
            # define temporary UTM profile for DTM
            dtm_utm_height, dtm_utm_width = dsm_utm.shape

            # Destination array (Original dimensions)
            dtm_out = np.zeros((src.height, src.width), dtype=dtm.dtype)

            reproject(
                source=dtm,
                destination=dtm_out,
                src_transform=transform,  # UTM transform
                src_crs=utm_crs,
                dst_transform=src.transform,  # Original transform
                dst_crs=src.crs,
                resampling=Resampling.bilinear,
                src_nodata=nodata,
                dst_nodata=nodata,
            )

        # Write Result
        with rasterio.open(dtm_path, "w", **src_profile) as dst:
            dst.write(dtm_out, 1)

    else:
        # standard write
        with rasterio.open(dsm_path) as src:
            profile = src.profile
            profile.update(dtype=dtm.dtype, nodata=nodata)

        with rasterio.open(dtm_path, "w", **profile) as dst:
            dst.write(dtm, 1)

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
