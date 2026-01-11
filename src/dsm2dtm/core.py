"""
dsm2dtm - Generate DTM (Digital Terrain Model) from DSM (Digital Surface Model)
Author: Naman Jain
        naman.jain@btech2015.iitgn.ac.in
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Dict, Optional, Tuple, Union

import numpy as np
import rasterio
from rasterio.crs import CRS
from rasterio.io import DatasetReader
from rasterio.transform import Affine
from rasterio.warp import Resampling, calculate_default_transform, reproject

# Backward compatibility for numpy < 1.21 (QGIS uses 1.20)
if TYPE_CHECKING:
    from numpy.typing import NDArray
else:
    NDArray = np.ndarray

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


def _create_context_from_src(src: DatasetReader) -> DSMContext:
    """
    Internal helper to create DSMContext from an open rasterio dataset.
    """
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


def _load_dsm(dsm_input: Union[str, DatasetReader]) -> DSMContext:
    """
    Load DSM context from a file path or an open rasterio dataset.
    """
    # Check if input is a path (str or PathLike)
    if isinstance(dsm_input, (str, os.PathLike)):
        with rasterio.open(dsm_input) as src:
            return _create_context_from_src(src)
    else:
        # Assume it is an open rasterio dataset
        return _create_context_from_src(dsm_input)


def _prepare_output(dtm: NDArray[np.floating], context: DSMContext) -> Tuple[NDArray[np.floating], Dict[str, Any]]:
    """
    Prepare the DTM for output, reprojecting back to original CRS if necessary.

    Args:
        dtm (NDArray[np.floating]): The generated DTM array (potentially in UTM).
        context (DSMContext): The context object containing original metadata.

    Returns:
        Tuple[NDArray, Dict]: The DTM array and its rasterio profile/metadata.
    """
    if not context.is_reprojected:
        profile = context.profile.copy()
        profile.update(dtype=dtm.dtype, nodata=context.nodata)
        return dtm, profile

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

    return dtm_out, out_profile


def save_dtm(dtm: NDArray[np.floating], profile: Dict[str, Any], output_path: str) -> None:
    """
    Write the DTM array to a file using the provided profile.

    Args:
        dtm (NDArray[np.floating]): The DTM array.
        profile (Dict[str, Any]): Rasterio profile metadata.
        output_path (str): Destination file path.
    """
    with rasterio.open(output_path, "w", **profile) as dst:
        dst.write(dtm, 1)


def generate_dtm(
    dsm_input: Union[str, DatasetReader],
    kernel_radius_meters: float = DEFAULT_KERNEL_RADIUS_METERS,
    slope: Optional[float] = None,
    initial_threshold: float = PMF_INITIAL_THRESHOLD,
    max_threshold: float = PMF_MAX_THRESHOLD,
) -> Tuple[NDArray[np.floating], Dict[str, Any]]:
    """
    Generate a DTM from a DSM (file path or rasterio dataset).

    This function handles loading, optional reprojection (to UTM for processing),
    DTM generation, and reprojection back to the original CRS.

    Args:
        dsm_input (Union[str, rasterio.io.DatasetReader]): Input DSM file path or open rasterio dataset.
        kernel_radius_meters (float, optional): Kernel radius for PMF in meters.
            Defaults to DEFAULT_KERNEL_RADIUS_METERS.
        slope (Optional[float], optional): Terrain slope. If None, calculated from data. Defaults to None.
        initial_threshold (float, optional): Initial elevation threshold for PMF. Defaults to PMF_INITIAL_THRESHOLD.
        max_threshold (float, optional): Max elevation threshold for PMF. Defaults to PMF_MAX_THRESHOLD.

    Returns:
        Tuple[NDArray, Dict]: A tuple containing the DTM numpy array and the rasterio profile (metadata).
    """
    # 1. Load and Prepare (Reproject to UTM if needed)
    ctx = _load_dsm(dsm_input)

    # 2. Process
    dtm_utm = dsm_to_dtm(
        ctx.dsm,
        ctx.resolution,
        kernel_radius_meters=kernel_radius_meters,
        slope=slope,
        initial_threshold=initial_threshold,
        max_threshold=max_threshold,
        nodata=ctx.nodata,
    )

    # 3. Prepare Output (Reproject back if needed)
    return _prepare_output(dtm_utm, ctx)


def main_cli() -> None:
    """Command line interface for generating DTM from DSM."""
    parser = argparse.ArgumentParser(description="Generate DTM from DSM")
    parser.add_argument("--dsm", help="Path to the DSM file", required=True)
    parser.add_argument("--out_dir", help="Directory to save the output DTM", default="generated_dtm")
    parser.add_argument(
        "--radius",
        type=float,
        default=DEFAULT_KERNEL_RADIUS_METERS,
        help=(
            "Window radius for the morphological filter in meters. "
            "Objects larger than 2x this radius will NOT be removed. "
            "Set this to slightly larger than half the width of the largest building. "
            "(default: 40.0)"
        ),
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

    # Generate DTM (returns array and profile)
    dtm_array, profile = generate_dtm(
        args.dsm,
        kernel_radius_meters=args.radius,
        slope=args.slope,
        initial_threshold=args.init_threshold,
        max_threshold=args.max_threshold,
    )

    # Construct output path
    os.makedirs(args.out_dir, exist_ok=True)
    output_name = os.path.splitext(os.path.basename(args.dsm))[0] + "_dtm.tif"
    dtm_path = os.path.join(args.out_dir, output_name)

    # Save to file
    save_dtm(dtm_array, profile, dtm_path)

    print(f"DTM generated at: {dtm_path}")


if __name__ == "__main__":
    main_cli()
