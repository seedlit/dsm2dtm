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
from rasterio.transform import Affine
from rasterio.warp import Resampling, calculate_default_transform, reproject

from dsm2dtm.algorithm import dsm_to_dtm
from dsm2dtm.constants import (
    DEFAULT_KERNEL_RADIUS_METERS,
    DEFAULT_NODATA,
    PMF_INITIAL_THRESHOLD,
    PMF_MAX_THRESHOLD,
)
from dsm2dtm.utm_utils import estimate_utm_crs


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


def _load_dsm(dsm_path: str) -> DSMContext:
    """
    Load a DSM file and prepare it for processing.
    If the input DSM uses a Geographic CRS (e.g., Lat/Lon), it is automatically
    reprojected to an appropriate local UTM zone to ensure accurate metric calculations.

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
    Save the generated DTM to a file.
    If the DSM was reprojected during loading (e.g., from Geographic to UTM),
    this function handles reprojecting the result back to the original CRS before saving.

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
    Main function to generate a DTM from a DSM file.
    This function coordinates the entire process: loading the DSM (and reprojecting if necessary),
    generating the DTM, and saving the result to the specified output directory.

    Args:
        dsm_path (str): Path to the input DSM file.
        out_dir (str): Directory where the output DTM will be saved.
        kernel_radius_meters (float, optional): Kernel radius for PMF in meters.
            Defaults to DEFAULT_KERNEL_RADIUS_METERS.
        slope (Optional[float], optional): Terrain slope. If None, calculated from data. Defaults to None.
        initial_threshold (float, optional): Initial elevation threshold for PMF. Defaults to PMF_INITIAL_THRESHOLD.
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
