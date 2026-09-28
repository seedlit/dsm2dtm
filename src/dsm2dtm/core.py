"""
dsm2dtm - Generate DTM (Digital Terrain Model) from DSM (Digital Surface Model)
Author: Naman Jain
        naman.jain@btech2015.iitgn.ac.in
"""

from __future__ import annotations

import argparse
import logging
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, Tuple, Union

import numpy as np
import rasterio
from rasterio.crs import CRS
from rasterio.io import DatasetReader
from rasterio.transform import Affine
from rasterio.vrt import WarpedVRT
from rasterio.warp import Resampling, calculate_default_transform, reproject
from rasterio.windows import Window as RioWindow

from dsm2dtm import tiling
from dsm2dtm.algorithm import dsm_to_dtm
from dsm2dtm.constants import (
    DEFAULT_KERNEL_RADIUS_METERS,
    DEFAULT_NODATA,
    DEFAULT_TILE_SIZE,
    PMF_INITIAL_THRESHOLD,
    PMF_MAX_THRESHOLD,
)
from dsm2dtm.utm_utils import estimate_utm_crs

logger = logging.getLogger(__name__)

# Backward compatibility for numpy < 1.21 (QGIS uses 1.20)
if TYPE_CHECKING:
    from numpy.typing import NDArray
else:
    NDArray = np.ndarray


@dataclass
class DSMContext:
    """Holds DSM data and metadata, including reprojection details."""

    dsm: NDArray[np.floating]
    profile: dict
    nodata: float
    resolution: Tuple[float, float]
    is_reprojected: bool = False
    original_crs: CRS | None = None
    original_transform: Affine | None = None
    original_shape: Tuple[int, int] | None = None  # (height, width)
    utm_crs: CRS | None = None
    utm_transform: Affine | None = None


def _create_context_from_src(src: DatasetReader) -> DSMContext:
    """
    Internal helper to create DSMContext from an open rasterio dataset.
    """
    if src.crs is None:
        raise ValueError("Input raster has no CRS. Assign a CRS before processing — units cannot be inferred.")

    is_geographic = src.crs.is_geographic
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

    logger.info("Input is Geographic (%s). Reprojecting to UTM for processing...", src.crs)
    left, bottom, right, top = src.bounds
    center_lon = (left + right) / 2
    center_lat = (bottom + top) / 2
    utm_crs = CRS.from_epsg(estimate_utm_crs(center_lon, center_lat))
    logger.info("Selected CRS: %s", utm_crs)

    transform, width, height = calculate_default_transform(src.crs, utm_crs, src.width, src.height, *src.bounds)

    # Pre-fill with nodata so cells outside the warped source extent stay flagged.
    dsm_utm = np.full((height, width), nodata, dtype=np.float32)
    reproject(
        source=rasterio.band(src, 1),
        destination=dsm_utm,
        src_transform=src.transform,
        src_crs=src.crs,
        dst_transform=transform,
        dst_crs=utm_crs,
        resampling=Resampling.bilinear,
        src_nodata=nodata,
        dst_nodata=nodata,
        init_dest_nodata=False,
    )

    profile = src.profile.copy()
    profile.update(
        {
            "crs": utm_crs,
            "transform": transform,
            "width": width,
            "height": height,
            "nodata": nodata,
            "dtype": dsm_utm.dtype,
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

    logger.info("Reprojecting DTM back to original CRS...")

    # Destination array (Original dimensions); pre-fill with nodata so corners
    # outside the warped extent stay flagged instead of defaulting to 0.0.
    if context.original_shape is None:
        raise RuntimeError("DSMContext.original_shape missing — cannot reproject DTM back to source extent.")
    orig_h, orig_w = context.original_shape
    dtm_out = np.full((orig_h, orig_w), context.nodata, dtype=dtm.dtype)

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
        init_dest_nodata=False,
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
    parent = os.path.dirname(output_path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    with rasterio.open(output_path, "w", **profile) as dst:
        dst.write(dtm, 1)


def _validate_params(kernel_radius_meters: float, slope: float | None) -> None:
    if not np.isfinite(kernel_radius_meters) or kernel_radius_meters <= 0:
        raise ValueError(f"kernel_radius_meters must be a positive finite number, got {kernel_radius_meters!r}")
    if slope is not None and (not np.isfinite(slope) or slope <= 0 or slope > 1):
        raise ValueError(f"slope must be in (0, 1] when provided, got {slope!r}")


def _output_profile(crs: CRS, transform: Affine, width: int, height: int, nodata: float) -> Dict[str, Any]:
    """Tiled, DEFLATE-compressed float32 GeoTIFF; 256 px blocks align with DEFAULT_TILE_SIZE writes."""
    return {
        "driver": "GTiff",
        "dtype": "float32",
        "count": 1,
        "crs": crs,
        "transform": transform,
        "width": width,
        "height": height,
        "nodata": nodata,
        "tiled": True,
        "blockxsize": 256,
        "blockysize": 256,
        "compress": "deflate",
        "predictor": 3,
        "BIGTIFF": "IF_SAFER",
    }


def _to_rio(window: tiling.Window) -> RioWindow:
    return RioWindow(window.col_off, window.row_off, window.width, window.height)


def _process_dataset_tiled(
    src: DatasetReader | WarpedVRT, output_path: str | os.PathLike, nodata: float, tile_size: int, **kwargs: Any
) -> float:
    """Stream `src` band 1 through `tiling.process_tiled` into a new GeoTIFF on the same grid."""
    profile = _output_profile(src.crs, src.transform, src.width, src.height, nodata)
    with rasterio.open(output_path, "w", **profile) as dst:
        return tiling.process_tiled(
            read=lambda w: src.read(1, window=_to_rio(w)),
            write=lambda w, block: dst.write(block, 1, window=_to_rio(w)),
            shape=(src.height, src.width),
            resolution=src.res,
            nodata=nodata,
            tile_size=tile_size,
            progress=lambda done, total: logger.info("Tile %d/%d", done, total),
            **kwargs,
        )


def _warp_back_clamped(
    utm_path: str | os.PathLike, src: DatasetReader, output_path: str | os.PathLike, nodata: float, tile_size: int
) -> None:
    """Warp the UTM DTM onto the source grid block by block, clamping to the source DSM."""
    profile = _output_profile(src.crs, src.transform, src.width, src.height, nodata)
    with (
        rasterio.open(utm_path) as utm,
        WarpedVRT(
            utm,
            crs=src.crs,
            transform=src.transform,
            width=src.width,
            height=src.height,
            resampling=Resampling.bilinear,
            src_nodata=nodata,
            nodata=nodata,
        ) as back,
        rasterio.open(output_path, "w", **profile) as dst,
    ):
        for tile in tiling.plan_tiles((src.height, src.width), tile_size, halo=0):
            window = _to_rio(tile.core)
            dtm = back.read(1, window=window).astype(np.float32)
            dsm = src.read(1, window=window).astype(np.float32)
            valid = (dtm != nodata) & (dsm != nodata) & np.isfinite(dtm) & np.isfinite(dsm)
            dtm[valid] = np.minimum(dtm[valid], dsm[valid])
            dst.write(dtm, 1, window=window)


def generate_dtm_file(
    dsm_path: str | os.PathLike,
    output_path: str | os.PathLike,
    kernel_radius_meters: float = DEFAULT_KERNEL_RADIUS_METERS,
    slope: float | None = None,
    initial_threshold: float = PMF_INITIAL_THRESHOLD,
    max_threshold: float = PMF_MAX_THRESHOLD,
    tile_size: int = DEFAULT_TILE_SIZE,
    workers: int | None = None,
) -> None:
    """
    Generate a DTM GeoTIFF from a DSM file, streaming tile by tile so memory use is
    bounded by `workers x tile_size` rather than raster size.

    Geographic inputs are warped to UTM on the fly (no full-size copy in memory), processed,
    written to a temporary file next to `output_path`, and warped back onto the input grid.

    Args:
        dsm_path: Input DSM file.
        output_path: Output DTM GeoTIFF (tiled, DEFLATE-compressed float32).
        kernel_radius_meters, slope, initial_threshold, max_threshold: As for `generate_dtm`.
        tile_size: Core tile edge in pixels.
        workers: Worker threads; defaults to `tiling.default_workers()`.
    """
    _validate_params(kernel_radius_meters, slope)
    if Path(dsm_path).resolve() == Path(output_path).resolve():
        raise ValueError("output_path must differ from dsm_path")
    parent = os.path.dirname(os.fspath(output_path))
    if parent:
        os.makedirs(parent, exist_ok=True)

    params = dict(
        kernel_radius_meters=kernel_radius_meters,
        slope=slope,
        initial_threshold=initial_threshold,
        max_threshold=max_threshold,
        workers=workers,
    )
    with rasterio.open(dsm_path) as src:
        if src.crs is None:
            raise ValueError("Input raster has no CRS. Assign a CRS before processing — units cannot be inferred.")
        nodata = src.nodata if src.nodata is not None else DEFAULT_NODATA

        if not src.crs.is_geographic:
            used_slope = _process_dataset_tiled(src, output_path, nodata, tile_size, **params)
        else:
            left, bottom, right, top = src.bounds
            utm_crs = CRS.from_epsg(estimate_utm_crs((left + right) / 2, (bottom + top) / 2))
            logger.info("Input is Geographic (%s). Reprojecting to %s for processing...", src.crs, utm_crs)
            with (
                WarpedVRT(src, crs=utm_crs, resampling=Resampling.bilinear, src_nodata=nodata, nodata=nodata) as vrt,
                tempfile.TemporaryDirectory(dir=parent or None) as tmp,
            ):
                utm_path = os.path.join(tmp, "dtm_utm.tif")
                used_slope = _process_dataset_tiled(vrt, utm_path, nodata, tile_size, **params)
                logger.info("Reprojecting DTM back to original CRS...")
                _warp_back_clamped(utm_path, src, output_path, nodata, tile_size)
    logger.info("Terrain slope used: %.3f", used_slope)


def generate_dtm(
    dsm_input: Union[str, DatasetReader],
    kernel_radius_meters: float = DEFAULT_KERNEL_RADIUS_METERS,
    slope: float | None = None,
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
    _validate_params(kernel_radius_meters, slope)

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
        help=f"Max elevation threshold in meters (default: {PMF_MAX_THRESHOLD})",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite the output DTM if it already exists.",
    )
    parser.add_argument(
        "--tile_size",
        type=int,
        default=DEFAULT_TILE_SIZE,
        help=f"Tile edge in pixels; memory use scales with workers x tile_size^2 (default: {DEFAULT_TILE_SIZE})",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help=f"Worker threads (default: {tiling.default_workers()})",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    if not os.path.isfile(args.dsm):
        raise SystemExit(f"Input DSM not found: {args.dsm}")

    os.makedirs(args.out_dir, exist_ok=True)
    output_name = os.path.splitext(os.path.basename(args.dsm))[0] + "_dtm.tif"
    dtm_path = os.path.join(args.out_dir, output_name)

    if os.path.exists(dtm_path) and not args.overwrite:
        raise SystemExit(f"Output already exists: {dtm_path} (pass --overwrite to replace)")

    generate_dtm_file(
        args.dsm,
        dtm_path,
        kernel_radius_meters=args.radius,
        slope=args.slope,
        initial_threshold=args.init_threshold,
        max_threshold=args.max_threshold,
        tile_size=args.tile_size,
        workers=args.workers,
    )

    logger.info("DTM generated at: %s", dtm_path)


if __name__ == "__main__":
    main_cli()
