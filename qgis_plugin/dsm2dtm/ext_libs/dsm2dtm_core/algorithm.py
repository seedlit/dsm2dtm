"""
Core algorithms for DSM to DTM conversion.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional, Tuple

import numpy as np
from scipy.ndimage import distance_transform_edt, gaussian_filter, grey_opening, zoom

# Backward compatibility for numpy < 1.21 (QGIS uses 1.20)
if TYPE_CHECKING:
    from numpy.typing import NDArray
else:
    NDArray = np.ndarray

from dsm2dtm_core.constants import (
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
    """
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

    if dsm_for_slope.shape[0] < 2 or dsm_for_slope.shape[1] < 2:
        return PMF_SLOPE

    dsm_nan = dsm_for_slope.copy()
    dsm_nan[~valid_mask] = np.nan

    dy, dx = np.gradient(dsm_nan)
    slope_per_pixel = np.sqrt(dy**2 + dx**2)
    slope_dimensionless = slope_per_pixel / res_for_slope

    valid_slopes = slope_dimensionless[valid_mask]

    if len(valid_slopes) == 0 or np.all(np.isnan(valid_slopes)):
        return PMF_SLOPE

    median_slope = np.nanmedian(valid_slopes)
    median_slope = max(0.01, min(median_slope, 1.0))
    return float(median_slope)


def get_adaptive_parameters(
    resolution: float, max_image_dimension: int = 10000, base_slope: float = PMF_SLOPE
) -> AdaptiveParameters:
    """Calculate algorithm parameters based on input resolution in meters."""
    res_meters = max(resolution, 0.001)
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
    """Apply the Progressive Morphological Filter (PMF)."""
    valid_mask = surface != nodata
    if not np.any(valid_mask):
        return surface.copy()
    min_val = np.min(surface[valid_mask])
    working = np.where(valid_mask, surface, min_val)
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
        if window_size > max_window:
            break

    return np.where(valid_mask, working, nodata)


def refine_ground_surface(
    ground: NDArray[np.floating],
    nodata: float,
    smoothen_radius: float,
    elevation_threshold: float = REFINEMENT_ELEVATION_THRESHOLD,
) -> NDArray[np.floating]:
    """Refine the ground surface by removing points that deviate significantly."""
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
    """Downsamples high-resolution DSM, processes at coarse resolution, and upsamples."""
    target_res = MIN_PROCESSING_RESOLUTION_METERS

    h, w = dsm.shape
    scale = cell_size / target_res
    new_h = int(h * scale)
    new_w = int(w * scale)

    if new_h < 10 or new_w < 10:
        return _process_standard_dsm(
            dsm, cell_size, nodata, kernel_radius_meters, slope, initial_threshold, max_threshold
        )

    print(
        f"High resolution input ({cell_size:.4f}m). "
        f"Downsampling to {MIN_PROCESSING_RESOLUTION_METERS}m for processing stability..."
    )

    valid_mask = dsm != nodata
    invalid_mask = ~valid_mask
    if not np.any(valid_mask):
        return dsm.copy()

    # Nearest neighbor fill before zooming to avoid interpolation artifacts with nodata
    dsm_filled = dsm.copy()
    if np.any(invalid_mask):
        _, ind = distance_transform_edt(invalid_mask, return_distances=True, return_indices=True)
        dsm_filled = dsm_filled[tuple(ind)]

    dsm_coarse = zoom(dsm_filled, scale, order=1)

    dtm_coarse = dsm_to_dtm(
        dsm_coarse,
        (target_res, target_res),
        kernel_radius_meters=kernel_radius_meters,
        slope=slope,
        initial_threshold=initial_threshold,
        max_threshold=max_threshold,
        nodata=nodata,
    )

    # Upsample
    dtm_fine = zoom(dtm_coarse, (h / new_h, w / new_w), order=1)

    # Restore original nodata mask
    dtm_fine[invalid_mask] = nodata

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
    """Standard DTM generation pipeline."""
    if slope is None:
        slope = calculate_terrain_slope(dsm, cell_size, nodata)

    max_dim = max(dsm.shape[0], dsm.shape[1])
    params = get_adaptive_parameters(cell_size, max_image_dimension=max_dim, base_slope=slope)

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

    ground = progressive_morphological_filter(
        dsm,
        nodata=nodata,
        initial_window=params.pmf_initial_window,
        max_window=params.pmf_max_window,
        slope=params.pmf_slope,
        initial_threshold=initial_threshold,
        max_threshold=max_threshold,
    )

    ground = refine_ground_surface(
        ground,
        nodata=nodata,
        smoothen_radius=params.refinement_smooth_sigma,
        elevation_threshold=REFINEMENT_ELEVATION_THRESHOLD,
    )

    valid_mask = ground != nodata
    if np.any(valid_mask):
        min_val = np.min(ground[valid_mask])
        smooth_input = np.where(ground == nodata, min_val, ground)
        smoothed = gaussian_filter(smooth_input, sigma=params.final_smooth_sigma)
        ground = np.where(valid_mask, smoothed, nodata)

    # Gap interpolation
    invalid_mask = ground == nodata
    if np.any(invalid_mask) and np.any(valid_mask):
        _, ind = distance_transform_edt(invalid_mask, return_distances=True, return_indices=True)
        filled_ground = ground[tuple(ind)]

        if GAP_FILL_SMOOTHING_ITERATIONS > 0:
            # Apply some smoothing specifically to the filled boundaries
            smoothed_filled = gaussian_filter(filled_ground, sigma=1.0)
            ground[invalid_mask] = smoothed_filled[invalid_mask]
        else:
            ground[invalid_mask] = filled_ground[invalid_mask]

    return ground


def dsm_to_dtm(
    dsm: NDArray[np.floating],
    resolution: Tuple[float, float],
    kernel_radius_meters: Optional[float] = None,
    slope: Optional[float] = None,
    initial_threshold: float = PMF_INITIAL_THRESHOLD,
    max_threshold: float = PMF_MAX_THRESHOLD,
    nodata: float = DEFAULT_NODATA,
) -> NDArray[np.floating]:
    """Generate DTM from DSM."""
    cell_size = (abs(resolution[0]) + abs(resolution[1])) / 2.0
    cell_size = max(cell_size, 0.001)

    if cell_size < (MIN_PROCESSING_RESOLUTION_METERS * 0.9):
        return _process_coarse_dsm(
            dsm, cell_size, nodata, kernel_radius_meters, slope, initial_threshold, max_threshold
        )

    return _process_standard_dsm(dsm, cell_size, nodata, kernel_radius_meters, slope, initial_threshold, max_threshold)
