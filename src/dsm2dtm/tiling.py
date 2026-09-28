"""
Tiled, multi-threaded DSM -> DTM processing for rasters too large to hold in memory.

Pure numpy/scipy and I/O-agnostic: callers supply `read(window)` / `write(window, block)`
callbacks (rasterio in `core.py`, GDAL in the QGIS plugin), so this module is vendored
into the plugin alongside `algorithm.py`.

Each tile is read with a halo wide enough that every filter in the pipeline sees the
same neighbourhood it would see untiled, so interior results match `dsm_to_dtm` on the
whole raster. scipy.ndimage releases the GIL, so threads give real parallelism.
"""

from __future__ import annotations

import math
import os
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable, List, Tuple

import numpy as np

from dsm2dtm.algorithm import dsm_to_dtm, median_slope, terrain_slope_samples
from dsm2dtm.constants import (
    DEFAULT_TILE_SIZE,
    FINAL_SMOOTH_SIGMA_METERS,
    GAP_FILL_MAX_SEARCH_DISTANCE_METERS,
    MIN_PROCESSING_RESOLUTION_METERS,
    PMF_INITIAL_THRESHOLD,
    PMF_MAX_THRESHOLD,
    PMF_MAX_WINDOW_METERS,
    REFINEMENT_SMOOTH_SIGMA_METERS,
)

if TYPE_CHECKING:
    from numpy.typing import NDArray
else:
    NDArray = np.ndarray

# Samples kept per tile for the global slope median; bounds memory on huge rasters.
_SLOPE_SAMPLES_PER_TILE = 250_000


class TilingCancelled(Exception):
    """Raised when `is_cancelled()` returns True between tiles."""


@dataclass(frozen=True)
class Window:
    row_off: int
    col_off: int
    height: int
    width: int


@dataclass(frozen=True)
class Tile:
    read: Window
    core: Window

    def crop(self, block: NDArray[np.floating]) -> NDArray[np.floating]:
        """Cut the core region out of a block covering `self.read`."""
        r = self.core.row_off - self.read.row_off
        c = self.core.col_off - self.read.col_off
        return block[r : r + self.core.height, c : c + self.core.width]


def halo_pixels(resolution: float, kernel_radius_meters: float | None) -> int:
    """Halo (pixels) covering the PMF window, both Gaussian kernels and the gap-fill reach."""
    radius = PMF_MAX_WINDOW_METERS / 2 if kernel_radius_meters is None else kernel_radius_meters
    reach_m = radius + 4 * REFINEMENT_SMOOTH_SIGMA_METERS + 4 * FINAL_SMOOTH_SIGMA_METERS
    reach_m += GAP_FILL_MAX_SEARCH_DISTANCE_METERS
    # Fine inputs are processed on a 0.5 m grid; pad one coarse cell for resampling at the seam.
    if resolution < MIN_PROCESSING_RESOLUTION_METERS * 0.9:
        reach_m += 2 * MIN_PROCESSING_RESOLUTION_METERS
    return math.ceil(reach_m / max(resolution, 1e-6))


def plan_tiles(shape: Tuple[int, int], tile_size: int, halo: int) -> List[Tile]:
    """Split `shape` into non-overlapping core windows, each with a halo clipped to the raster."""
    height, width = shape
    tiles = []
    for row in range(0, height, tile_size):
        for col in range(0, width, tile_size):
            core = Window(row, col, min(tile_size, height - row), min(tile_size, width - col))
            r0, c0 = max(row - halo, 0), max(col - halo, 0)
            r1 = min(row + core.height + halo, height)
            c1 = min(col + core.width + halo, width)
            tiles.append(Tile(read=Window(r0, c0, r1 - r0, c1 - c0), core=core))
    return tiles


def default_workers() -> int:
    """Threads to use by default; capped because each worker holds ~10x a tile in memory."""
    return max(1, min(os.cpu_count() or 1, 4))


def estimate_slope_tiled(
    read: Callable[[Window], NDArray[np.floating]],
    shape: Tuple[int, int],
    resolution: float,
    nodata: float,
    tile_size: int = DEFAULT_TILE_SIZE,
) -> float:
    """Global median terrain slope, pooled from tile cores so the whole raster is never in memory."""
    rng = np.random.default_rng(0)
    pooled = []
    for tile in plan_tiles(shape, tile_size, halo=0):
        samples = terrain_slope_samples(_as_float(read(tile.core), nodata), resolution, nodata)
        if samples.size > _SLOPE_SAMPLES_PER_TILE:
            samples = rng.choice(samples, _SLOPE_SAMPLES_PER_TILE, replace=False)
        pooled.append(samples)
    return median_slope(np.concatenate(pooled) if pooled else np.empty(0, dtype=np.float32))


def process_tiled(
    read: Callable[[Window], NDArray[np.floating]],
    write: Callable[[Window, NDArray[np.floating]], None],
    shape: Tuple[int, int],
    resolution: Tuple[float, float],
    nodata: float,
    kernel_radius_meters: float | None = None,
    slope: float | None = None,
    initial_threshold: float = PMF_INITIAL_THRESHOLD,
    max_threshold: float = PMF_MAX_THRESHOLD,
    tile_size: int = DEFAULT_TILE_SIZE,
    workers: int | None = None,
    progress: Callable[[int, int], None] | None = None,
    is_cancelled: Callable[[], bool] | None = None,
) -> float:
    """
    Run `dsm_to_dtm` tile by tile, writing each tile's core through `write`.

    `read` and `write` are serialized with a lock, so they need not be thread-safe.

    Args:
        read: Returns the DSM block for a window.
        write: Stores the DTM block for a (core) window.
        shape: Raster (height, width).
        resolution: (x_res, y_res) in meters.
        nodata: Nodata value of the blocks returned by `read`; also used for output.
        kernel_radius_meters, slope, initial_threshold, max_threshold: As for `dsm_to_dtm`.
            When `slope` is None it is estimated once over the whole raster.
        tile_size: Core tile edge in pixels.
        workers: Thread count; defaults to `default_workers()`.
        progress: Called with (tiles_done, tiles_total) after each tile.
        is_cancelled: Polled before each tile; raises `TilingCancelled` when it returns True.

    Returns:
        float: The slope used, so callers can report it.
    """
    cell_size = (abs(resolution[0]) + abs(resolution[1])) / 2.0
    io_lock = threading.Lock()

    def locked_read(window: Window) -> NDArray[np.floating]:
        with io_lock:
            return _as_float(read(window), nodata)

    if slope is None:
        slope = estimate_slope_tiled(locked_read, shape, cell_size, nodata, tile_size)

    tiles = plan_tiles(shape, tile_size, halo_pixels(cell_size, kernel_radius_meters))
    done = 0

    def run(tile: Tile) -> None:
        nonlocal done
        if is_cancelled is not None and is_cancelled():
            raise TilingCancelled()
        dtm = dsm_to_dtm(
            locked_read(tile.read),
            resolution,
            kernel_radius_meters=kernel_radius_meters,
            slope=slope,
            initial_threshold=initial_threshold,
            max_threshold=max_threshold,
            nodata=nodata,
        )
        with io_lock:
            write(tile.core, tile.crop(dtm))
            done += 1
            if progress is not None:
                progress(done, len(tiles))

    with ThreadPoolExecutor(max_workers=workers or default_workers()) as pool:
        for future in [pool.submit(run, tile) for tile in tiles]:
            future.result()
    return slope


def _as_float(block: NDArray, nodata: float) -> NDArray[np.floating]:
    """float32 view of a block, with NaN mapped to `nodata` so every tile sees one sentinel."""
    block = np.asarray(block, dtype=np.float32)
    if np.isfinite(nodata):
        non_finite = ~np.isfinite(block)
        if np.any(non_finite):
            block = np.where(non_finite, np.float32(nodata), block)
    return block
