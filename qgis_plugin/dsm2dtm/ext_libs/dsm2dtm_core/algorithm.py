"""
Core algorithms for DSM to DTM conversion.

Pure numpy + scipy: no rasterio dependency. The QGIS plugin vendors this module
verbatim, so any I/O (raster open, reproject, write) must live in callers
(`core.py` for the library, `processing_algorithm.py` for the plugin).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Tuple

import numpy as np
from scipy.ndimage import distance_transform_edt, gaussian_filter, grey_opening, uniform_filter, zoom

from dsm2dtm_core.constants import (
    DEFAULT_NODATA,
    FINAL_SMOOTH_SIGMA_METERS,
    GAP_FILL_MAX_SEARCH_DISTANCE_METERS,
    MAX_AUTO_SLOPE,
    MIN_PROCESSING_RESOLUTION_METERS,
    PMF_INITIAL_THRESHOLD,
    PMF_INITIAL_WINDOW_METERS,
    PMF_MAX_THRESHOLD,
    PMF_MAX_WINDOW_METERS,
    PMF_SLOPE,
    REFINEMENT_ELEVATION_THRESHOLD,
    REFINEMENT_SMOOTH_SIGMA_METERS,
)

logger = logging.getLogger(__name__)

# Backward compatibility for numpy < 1.21 (QGIS uses 1.20)
if TYPE_CHECKING:
    from numpy.typing import NDArray
else:
    NDArray = np.ndarray


def _fill_nearest(arr: NDArray[np.floating], invalid_mask: NDArray[np.bool_]) -> NDArray[np.floating]:
    """Copy of `arr` with invalid cells replaced by their nearest valid neighbour."""
    if not np.any(invalid_mask):
        return arr.copy()
    ind = distance_transform_edt(invalid_mask, return_distances=False, return_indices=True)
    return arr[tuple(ind)]


@dataclass
class AdaptiveParameters:
    """Parameters adapted to the specific DSM resolution."""

    pmf_initial_window: int
    pmf_max_window: int
    pmf_slope: float
    refinement_smooth_sigma: float
    final_smooth_sigma: float
    gap_fill_search_dist: float


def terrain_slope_samples(dsm: NDArray[np.floating], resolution: float, nodata: float) -> NDArray[np.floating]:
    """
    Gradient magnitudes (rise/run) of the valid DSM cells, computed at >= ~1 m resolution.

    Exposed separately from `calculate_terrain_slope` so tiled processing can pool samples
    across tiles and take one global median.

    Args:
        dsm (NDArray[np.floating]): The input DSM as a 2D numpy array.
        resolution (float): The spatial resolution of the DSM (in meters per pixel).
        nodata (float): The value representing no data in the DSM.

    Returns:
        NDArray[np.floating]: 1D array of finite slopes; empty if none can be computed.
    """
    target_res = 1.0
    current_res = max(resolution, 0.001)

    if current_res < (target_res * 0.5):
        dsm_for_slope = zoom(dsm, current_res / target_res, order=1)
        res_for_slope = target_res
    else:
        dsm_for_slope = dsm
        res_for_slope = current_res

    valid_mask = dsm_for_slope != nodata
    if not np.any(valid_mask) or dsm_for_slope.shape[0] < 2 or dsm_for_slope.shape[1] < 2:
        return np.empty(0, dtype=np.float32)

    # NaN out nodata so gradients at the edge of validity are dropped instead of overflowing.
    dsm_nan = dsm_for_slope.astype(np.float32, copy=True)
    dsm_nan[~valid_mask] = np.nan
    dy, dx = np.gradient(dsm_nan)
    slopes = np.hypot(dy, dx)[valid_mask] / res_for_slope
    return slopes[np.isfinite(slopes)]


def median_slope(samples: NDArray[np.floating]) -> float:
    """Median of slope samples clamped to [0.01, MAX_AUTO_SLOPE]; `PMF_SLOPE` when there are none."""
    if samples.size == 0:
        return PMF_SLOPE
    # Median, not mean: robust against vertical walls.
    return float(min(max(np.median(samples), 0.01), MAX_AUTO_SLOPE))


def calculate_terrain_slope(dsm: NDArray[np.floating], resolution: float, nodata: float) -> float:
    """
    Calculate the median terrain slope (rise/run) of the DSM.

    This function estimates the general slope of the terrain to automatically tune the
    Progressive Morphological Filter (PMF) slope parameter.

    Args:
        dsm (NDArray[np.floating]): The input Digital Surface Model (DSM) as a 2D numpy array.
        resolution (float): The spatial resolution of the DSM (in meters per pixel).
        nodata (float): The value representing no data in the DSM.

    Returns:
        float: The calculated median slope of the terrain (clamped between 0.01 and MAX_AUTO_SLOPE).
    """
    return median_slope(terrain_slope_samples(dsm, resolution, nodata))


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
    pad = max_window // 2
    working = _pad_along_trend(_fill_nearest(surface, ~valid_mask), pad)
    for window_size in _pmf_window_sequence(initial_window, max_window):
        window_radius = (window_size - 1) // 2
        dh_threshold = min(initial_threshold + slope * window_radius, max_threshold)

        opened = grey_opening(working, size=(window_size, window_size))

        diff = working - opened
        non_ground_mask = diff > dh_threshold
        working[non_ground_mask] = opened[non_ground_mask]

    working = working[pad : pad + surface.shape[0], pad : pad + surface.shape[1]]
    return np.where(valid_mask, working, nodata)


def _pad_along_trend(arr: NDArray[np.floating], pad: int) -> NDArray[np.floating]:
    """
    Pad by extrapolating a smoothed trend past the border.

    Morphological opening near an edge otherwise treats uphill terrain as a peak and
    shaves it. Odd reflection alone continues slopes but mirrors edge objects into
    taller ramps; smoothing the reflected margin keeps the slope and drops the objects.
    """
    if pad == 0:
        return arr
    padded = np.pad(arr, pad, mode="reflect", reflect_type="odd")
    # Box filter: O(1) per pixel for any size, and like any linear smoother it preserves planes.
    # Only the margin needs the trend, so smooth strips just deep enough to cover it.
    size = pad | 1
    strip = 2 * pad
    padded[:pad] = uniform_filter(padded[:strip], size)[:pad]
    padded[-pad:] = uniform_filter(padded[-strip:], size)[-pad:]
    padded[:, :pad] = uniform_filter(padded[:, :strip], size)[:, :pad]
    padded[:, -pad:] = uniform_filter(padded[:, -strip:], size)[:, -pad:]
    return padded


def _pmf_window_sequence(initial_window: int, max_window: int) -> list[int]:
    """Exponentially growing odd window sizes, always ending exactly at `max_window`."""
    windows = []
    window_size = initial_window
    while window_size < max_window:
        windows.append(window_size)
        window_size = 2 * window_size - 1
    windows.append(max_window)
    return windows


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
    smoothed = gaussian_filter(_fill_nearest(ground, ~valid_mask), sigma=smoothen_radius)
    diff = ground - smoothed
    refined = ground.copy()
    refined[(diff >= elevation_threshold) & valid_mask] = nodata
    return refined


def _process_coarse_dsm(
    dsm: NDArray[np.floating],
    cell_size: float,
    nodata: float,
    kernel_radius_meters: float | None,
    slope: float | None,
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

    logger.info(
        "High resolution input (%.4fm). Downsampling to %sm for processing stability...",
        cell_size,
        MIN_PROCESSING_RESOLUTION_METERS,
    )

    valid_mask = dsm != nodata
    invalid_mask = ~valid_mask
    if not np.any(valid_mask):
        return dsm.copy()

    # Nearest-neighbor fill before zooming so bilinear interpolation
    # doesn't smear the nodata sentinel into valid neighbors.
    dsm_coarse = zoom(_fill_nearest(dsm, invalid_mask), scale, order=1)

    dtm_coarse = dsm_to_dtm(
        dsm_coarse,
        (target_res, target_res),
        kernel_radius_meters=kernel_radius_meters,
        slope=slope,
        initial_threshold=initial_threshold,
        max_threshold=max_threshold,
        nodata=nodata,
    )

    # Fill nodata in dtm_coarse before bilinear upsample so the sentinel
    # doesn't smear into valid neighbours (BUG-58).
    coarse_invalid = dtm_coarse == nodata
    dtm_coarse_filled = dtm_coarse if np.all(coarse_invalid) else _fill_nearest(dtm_coarse, coarse_invalid)

    dtm_fine = zoom(dtm_coarse_filled, (h / dtm_coarse_filled.shape[0], w / dtm_coarse_filled.shape[1]), order=1)
    # scipy.ndimage.zoom can be off by one — clamp/pad to the expected shape.
    dtm_fine = dtm_fine[:h, :w]
    if dtm_fine.shape != (h, w):
        padded = np.full((h, w), nodata, dtype=dtm_fine.dtype)
        padded[: dtm_fine.shape[0], : dtm_fine.shape[1]] = dtm_fine
        dtm_fine = padded

    dtm_fine[invalid_mask] = nodata
    return dtm_fine


def _process_standard_dsm(
    dsm: NDArray[np.floating],
    cell_size: float,
    nodata: float,
    kernel_radius_meters: float | None,
    slope: float | None,
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
        # Fallback for un-projected Degree data (Lat/Lon): convert via the equator constant.
        res_meters = cell_size * 111320.0 if cell_size < 0.01 else cell_size
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

    # Step 3: Gap fill, then light smoothing. Filling first keeps nodata out of the
    # Gaussian kernel; cells beyond `gap_fill_search_dist` (water bodies, coverage
    # gaps) are returned to nodata rather than stamped with a far-away value.
    invalid_mask = ground == nodata
    if np.all(invalid_mask):
        return ground.astype(np.float32)
    has_gaps = np.any(invalid_mask)
    if has_gaps:
        distances, ind = distance_transform_edt(invalid_mask, return_distances=True, return_indices=True)
        ground = ground[tuple(ind)]
    dtm = gaussian_filter(ground, sigma=params.final_smooth_sigma).astype(np.float32)
    if has_gaps:
        dtm[invalid_mask & (distances > params.gap_fill_search_dist)] = nodata
    return dtm


def dsm_to_dtm(
    dsm: NDArray[np.floating],
    resolution: Tuple[float, float],
    kernel_radius_meters: float | None = None,
    slope: float | None = None,
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

    # Work in float32 with a finite sentinel: integer DSMs can't hold NaN, and NaN
    # nodata never compares equal to itself.
    work_nodata = float(nodata) if np.isfinite(np.float32(nodata)) else DEFAULT_NODATA
    dsm = np.asarray(dsm, dtype=np.float32)
    non_finite = ~np.isfinite(dsm)
    if np.any(non_finite):
        dsm = np.where(non_finite, np.float32(work_nodata), dsm)

    if cell_size < (MIN_PROCESSING_RESOLUTION_METERS * 0.9):
        dtm = _process_coarse_dsm(
            dsm, cell_size, work_nodata, kernel_radius_meters, slope, initial_threshold, max_threshold
        )
    else:
        dtm = _process_standard_dsm(
            dsm, cell_size, work_nodata, kernel_radius_meters, slope, initial_threshold, max_threshold
        )

    # Bare earth can't sit above the surface; smoothing and gap fill occasionally push it there.
    valid = dsm != np.float32(work_nodata)
    dtm[valid] = np.minimum(dtm[valid], dsm[valid])

    if work_nodata != nodata:
        dtm[dtm == np.float32(work_nodata)] = nodata
    return dtm
