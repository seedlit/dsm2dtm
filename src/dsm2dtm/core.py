"""
dsm2dtm - Generate DTM (Digital Terrain Model) from DSM (Digital Surface Model)
Author: Naman Jain
"""

import argparse
import os

import numpy as np
import rasterio
from numpy.typing import NDArray
from rasterio.fill import fillnodata
from scipy.ndimage import gaussian_filter, grey_opening


def progressive_morphological_filter(
    surface: NDArray[np.floating],
    nodata: float,
    max_window: int = 81,
    slope: float = 0.05,
    initial_threshold: float = 0.1,
    max_threshold: float = 0.5,
) -> NDArray[np.floating]:
    """
    Progressive Morphological Filter (PMF) based on Zhang et al. (2003).

    Iteratively applies morphological opening with increasing window sizes.
    Points above the opened surface by more than a threshold are classified as non-ground.
    """
    valid_mask = surface != nodata

    if not np.any(valid_mask):
        return surface.copy()

    min_val = np.min(surface[valid_mask])
    working = np.where(valid_mask, surface, min_val)

    window_size = 3
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
    smoothen_radius: float = 10.0,
    elevation_threshold: float = 0.8,
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
    kernel_radius: int = 40,
    nodata: float = -99999.0,
) -> NDArray[np.floating]:
    """
    Generate DTM from DSM using Progressive Morphological Filter with refinement.

    Parameters
    ----------
    dsm : NDArray[np.floating]
        Input DSM as 2D numpy array
    resolution : tuple[float, float]
        Pixel resolution (x_res, y_res) in CRS units
    kernel_radius : int
        Maximum filter kernel radius in pixels
    nodata : float
        No-data value for the raster

    Returns
    -------
    NDArray[np.floating]
        Generated DTM as 2D numpy array
    """
    # Step 1: Progressive morphological filtering
    max_window = 2 * kernel_radius + 1
    if max_window < 81:
        max_window = 81

    ground = progressive_morphological_filter(
        dsm,
        nodata=nodata,
        max_window=max_window,
        slope=0.05,
        initial_threshold=0.1,
        max_threshold=0.5,
    )

    # Step 2: Smooth-based refinement to remove remaining outliers
    ground = refine_ground_surface(
        ground,
        nodata=nodata,
        smoothen_radius=10.0,
        elevation_threshold=0.8,
    )

    # Step 3: Light gaussian smoothing
    valid_mask = ground != nodata
    if np.any(valid_mask):
        min_val = np.min(ground[valid_mask])
        smooth_input = np.where(ground == nodata, min_val, ground)
        smoothed = gaussian_filter(smooth_input, sigma=0.5)
        ground = np.where(valid_mask, smoothed, nodata)

    # Step 4: Gap interpolation
    mask = (ground != nodata).astype(np.uint8)
    dtm = ground.copy().astype(np.float32)
    fillnodata(dtm, mask=mask, max_search_distance=100.0, smoothing_iterations=0)

    return dtm


def main(
    dsm_path: str,
    out_dir: str,
    kernel_radius: int = 40,
) -> str:
    """
    Generate DTM from DSM file.

    Parameters
    ----------
    dsm_path : str
        Path to input DSM GeoTIFF
    out_dir : str
        Directory for output DTM
    kernel_radius : int
        Maximum morphological filter kernel radius in pixels

    Returns
    -------
    str
        Path to generated DTM file
    """
    os.makedirs(out_dir, exist_ok=True)
    temp_dir = os.path.join(out_dir, "temp_files")
    os.makedirs(temp_dir, exist_ok=True)

    original_dsm_path = dsm_path

    # Read DSM
    with rasterio.open(dsm_path) as src:
        dsm = src.read(1)
        profile = src.profile
        resolution = src.res
        nodata = src.nodata if src.nodata is not None else -99999.0

    # Generate DTM
    dtm = dsm_to_dtm(dsm, resolution, kernel_radius=kernel_radius, nodata=nodata)

    # Write output
    output_name = os.path.splitext(os.path.basename(original_dsm_path))[0] + "_dtm.tif"
    dtm_path = os.path.join(out_dir, output_name)

    profile.update(dtype=dtm.dtype, nodata=nodata)
    with rasterio.open(dtm_path, "w", **profile) as dst:
        dst.write(dtm, 1)

    return dtm_path


def main_cli() -> None:
    """Command line interface for generating DTM from DSM."""
    parser = argparse.ArgumentParser(description="Generate DTM from DSM")
    parser.add_argument("--dsm", help="Path to the DSM file", required=True)
    parser.add_argument("--out_dir", help="Directory to save the output DTM", default="generated_dtm")
    parser.add_argument("--kernel_radius", type=int, default=40, help="Maximum kernel radius")
    args = parser.parse_args()

    dtm_path = main(args.dsm, args.out_dir, kernel_radius=args.kernel_radius)
    print(f"DTM generated at: {dtm_path}")


if __name__ == "__main__":
    main_cli()
