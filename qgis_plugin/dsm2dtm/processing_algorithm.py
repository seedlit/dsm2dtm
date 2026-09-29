"""DSM to DTM Processing Algorithm."""

import os

import numpy as np

# Vendored, pure numpy/scipy modules (no rasterio dependency)
from dsm2dtm_core.constants import MIN_PROCESSING_RESOLUTION_METERS
from dsm2dtm_core.tiling import TilingCancelled, default_workers, plan_tiles, process_tiled
from dsm2dtm_core.utm_utils import estimate_utm_crs
from qgis.core import (
    Qgis,
    QgsProcessingAlgorithm,
    QgsProcessingException,
    QgsProcessingParameterNumber,
    QgsProcessingParameterRasterDestination,
    QgsProcessingParameterRasterLayer,
    QgsProcessingUtils,
)

# GeoTIFF creation options for elevation data: lossless DEFLATE, tiled, with the
# float-friendly PREDICTOR=3 — typical 4-5x size reduction on smooth terrain.
_GTIFF_OPTS = ["COMPRESS=DEFLATE", "TILED=YES", "PREDICTOR=3", "BIGTIFF=IF_SAFER"]
_MAX_WINDOW_PX = 5000
# Scoped enum exists from QGIS 3.36; PyQt6 builds (QGIS 4) need it instead of the unscoped alias.
_NUMBER_DOUBLE = getattr(Qgis, "ProcessingNumberParameterType", QgsProcessingParameterNumber).Double


def _utm_epsg_for(lon: float, lat: float) -> int:
    """EPSG of the UTM zone for (lon, lat). Pyproj-backed so the plugin and CLI agree."""
    return estimate_utm_crs(lon, lat)


def _safe_nodata(nodata: float) -> float:
    """Clamp non-finite nodata to -9999 so downstream readers (ArcGIS, older GDAL) cope."""
    return -9999.0 if not np.isfinite(nodata) else float(nodata)


class Dsm2DtmAlgorithm(QgsProcessingAlgorithm):
    """Algorithm to convert DSM (Digital Surface Model) to DTM (Digital Terrain Model).

    Uses Progressive Morphological Filtering to remove buildings, vegetation,
    and other non-ground features from elevation data.
    """

    INPUT = "INPUT"
    RADIUS = "RADIUS"
    SLOPE = "SLOPE"
    OUTPUT = "OUTPUT"

    def name(self):
        """Return the algorithm name used for identification."""
        return "dsm_to_dtm"

    def displayName(self):
        """Return the algorithm name shown in the toolbox."""
        return "DSM to DTM"

    def group(self):
        """Return the group name this algorithm belongs to."""
        return "Terrain"

    def groupId(self):
        """Return the group ID."""
        return "terrain"

    def shortHelpString(self):
        """Return the help text shown in the algorithm dialog."""
        return (
            "Generates a Digital Terrain Model (DTM) from a Digital Surface Model (DSM).\n\n"
            "Removes buildings, vegetation, and other non-ground features using "
            "Progressive Morphological Filtering.\n\n"
            "Parameters:\n"
            "• Input DSM: The input Digital Surface Model raster.\n"
            "• Radius: Kernel radius in meters. Objects larger than 2x this value "
            "will typically NOT be removed. Default: 40m.\n"
            "• Slope: Terrain slope (0-1). Set to 0 for automatic detection.\n\n"
            "Output:\n"
            "• A bare-earth DTM raster with the same extent and resolution as the input."
        )

    def createInstance(self):
        """Return a new instance of the algorithm."""
        return Dsm2DtmAlgorithm()

    def initAlgorithm(self, config=None):
        """Define the algorithm inputs and outputs."""
        self.addParameter(
            QgsProcessingParameterRasterLayer(
                self.INPUT,
                "Input DSM",
            )
        )
        self.addParameter(
            QgsProcessingParameterNumber(
                self.RADIUS,
                "Radius (meters)",
                type=_NUMBER_DOUBLE,
                defaultValue=40.0,
                minValue=1.0,
                maxValue=500.0,
            )
        )
        self.addParameter(
            QgsProcessingParameterNumber(
                self.SLOPE,
                "Slope (0=auto, 0.01-1.0=manual)",
                type=_NUMBER_DOUBLE,
                defaultValue=0.0,
                minValue=0.0,
                maxValue=1.0,
                optional=True,
            )
        )
        self.addParameter(
            QgsProcessingParameterRasterDestination(
                self.OUTPUT,
                "Output DTM",
            )
        )

    def processAlgorithm(self, parameters, context, feedback):
        """Execute the DSM to DTM conversion algorithm.

        Streams the raster tile by tile, so memory use is bounded by
        worker count x tile size rather than raster size.

        Args:
            parameters: Algorithm parameters from the dialog.
            context: Processing context.
            feedback: Feedback object for progress reporting.

        Returns:
            Dictionary with output layer path.
        """
        input_layer = self.parameterAsRasterLayer(parameters, self.INPUT, context)
        radius = self.parameterAsDouble(parameters, self.RADIUS, context)
        slope = self.parameterAsDouble(parameters, self.SLOPE, context)
        output_path = self.parameterAsOutputLayer(parameters, self.OUTPUT, context)

        if input_layer is None:
            raise QgsProcessingException("Invalid input raster layer")

        input_crs = input_layer.crs()
        if not input_crs.isValid():
            raise QgsProcessingException("Input raster has no CRS. Assign a CRS in QGIS before running this algorithm.")

        feedback.pushInfo(f"Processing: {input_layer.source()}")
        feedback.pushInfo(f"Radius: {radius}m, Slope: {'auto' if slope == 0 else slope}")

        extent = input_layer.extent()
        rows = input_layer.height()
        cols = input_layer.width()
        feedback.pushInfo(f"Raster size: {cols}x{rows} pixels")

        nodata = input_layer.dataProvider().sourceNoDataValue(1)
        if nodata is None or not np.isfinite(nodata):
            nodata = -9999.0
            feedback.pushInfo("No usable nodata value found, using -9999.0")

        from osgeo import gdal

        src_ds = gdal.Open(input_layer.source())
        if src_ds is None:
            raise QgsProcessingException(f"Could not open input raster: {input_layer.source()}")
        # Snapshot the file's authoritative geotransform/projection so the
        # output matches the input even when QGIS layer.crs / layer.extent
        # disagree with on-disk metadata or the geotransform has rotation.
        src_geotransform = src_ds.GetGeoTransform()
        src_projection = src_ds.GetProjection() or input_crs.toWkt()

        workers = _worker_count(context)
        run_kwargs = {
            "kernel_radius_meters": radius if radius > 0 else None,
            "slope": slope if slope > 0 else None,
            "workers": workers,
            "is_cancelled": feedback.isCanceled,
        }

        try:
            if input_crs.isGeographic():
                center_lon = (extent.xMinimum() + extent.xMaximum()) / 2.0
                center_lat = (extent.yMinimum() + extent.yMaximum()) / 2.0
                utm_epsg = _utm_epsg_for(center_lon, center_lat)
                feedback.pushInfo(
                    f"Input is geographic ({input_crs.authid()}). Reprojecting to EPSG:{utm_epsg} for processing..."
                )
                # A VRT warps lazily, so each tile is reprojected on read instead of
                # materialising the whole UTM raster in memory.
                utm_ds = gdal.Warp(
                    "",
                    src_ds,
                    format="VRT",
                    dstSRS=f"EPSG:{utm_epsg}",
                    srcNodata=nodata,
                    dstNodata=nodata,
                    resampleAlg=gdal.GRA_Bilinear,
                )
                if utm_ds is None:
                    raise QgsProcessingException(f"Failed to reproject input to EPSG:{utm_epsg}")
                utm_path = QgsProcessingUtils.generateTempFilename("dsm2dtm_utm.tif")
                try:
                    self._run_tiles(gdal, utm_ds, utm_path, nodata, run_kwargs, feedback, progress_span=(0, 80))
                    utm_ds = None
                    if feedback.isCanceled():
                        return {}
                    feedback.pushInfo("Reprojecting DTM back to input CRS...")
                    out_ds = gdal.Warp(
                        output_path,
                        utm_path,
                        format="GTiff",
                        dstSRS=src_projection,
                        outputBounds=(extent.xMinimum(), extent.yMinimum(), extent.xMaximum(), extent.yMaximum()),
                        width=cols,
                        height=rows,
                        srcNodata=nodata,
                        dstNodata=nodata,
                        resampleAlg=gdal.GRA_Bilinear,
                        creationOptions=_GTIFF_OPTS,
                    )
                    if out_ds is None:
                        raise QgsProcessingException(f"Could not create output file: {output_path}")
                    out_ds = None
                finally:
                    if os.path.exists(utm_path):
                        gdal.GetDriverByName("GTiff").Delete(utm_path)
                _clamp_to_dsm(gdal, output_path, src_ds, nodata)
            else:
                self._run_tiles(
                    gdal,
                    src_ds,
                    output_path,
                    nodata,
                    run_kwargs,
                    feedback,
                    progress_span=(0, 100),
                    geotransform=src_geotransform,
                    projection=src_projection,
                )
        except TilingCancelled:
            return {}
        finally:
            src_ds = None

        if feedback.isCanceled():
            return {}
        feedback.setProgress(100)
        feedback.pushInfo("Done!")
        return {self.OUTPUT: output_path}

    def _run_tiles(
        self, gdal, in_ds, out_path, nodata, run_kwargs, feedback, progress_span, geotransform=None, projection=None
    ):
        """Stream band 1 of `in_ds` through the tiled pipeline into a new GeoTIFF on the same grid."""
        cols, rows = in_ds.RasterXSize, in_ds.RasterYSize
        gt = geotransform or in_ds.GetGeoTransform()
        # Pixel sizes from the geotransform (handles rotated/sheared rasters).
        resolution = ((gt[1] ** 2 + gt[2] ** 2) ** 0.5, (gt[4] ** 2 + gt[5] ** 2) ** 0.5)
        feedback.pushInfo(f"Processing resolution: {resolution[0]:.4f}m x {resolution[1]:.4f}m")
        _check_window_size(run_kwargs["kernel_radius_meters"], resolution)

        out_ds = gdal.GetDriverByName("GTiff").Create(out_path, cols, rows, 1, gdal.GDT_Float32, options=_GTIFF_OPTS)
        if out_ds is None:
            raise QgsProcessingException(f"Could not create output file: {out_path}")
        out_ds.SetGeoTransform(gt)
        out_ds.SetProjection(projection or in_ds.GetProjection())
        in_band = in_ds.GetRasterBand(1)
        out_band = out_ds.GetRasterBand(1)
        out_band.SetNoDataValue(_safe_nodata(nodata))

        start, end = progress_span

        def progress(done, total):
            feedback.setProgress(start + (end - start) * done / total)

        feedback.pushInfo(f"Running DSM to DTM conversion with {run_kwargs['workers']} threads...")
        try:
            used_slope = process_tiled(
                read=lambda w: in_band.ReadAsArray(w.col_off, w.row_off, w.width, w.height),
                write=lambda w, block: out_band.WriteArray(block, w.col_off, w.row_off),
                shape=(rows, cols),
                resolution=resolution,
                nodata=nodata,
                progress=progress,
                **run_kwargs,
            )
        except TilingCancelled:
            raise
        except Exception as e:
            raise QgsProcessingException(f"Algorithm failed: {e!s}") from e
        finally:
            out_band.FlushCache()
            out_band = None
            out_ds = None
        feedback.pushInfo(f"Terrain slope used: {used_slope:.3f}")


def _worker_count(context):
    """Honour the Processing 'max threads' setting where QGIS exposes it."""
    max_threads = getattr(context, "maximumThreads", lambda: 0)()
    return min(max_threads, default_workers()) if max_threads and max_threads > 0 else default_workers()


def _processing_resolution(resolution):
    """Inputs finer than ~0.45 m are filtered on a 0.5 m grid (see algorithm.dsm_to_dtm)."""
    cell = max(resolution[0], resolution[1], 1e-6)
    return MIN_PROCESSING_RESOLUTION_METERS if cell < MIN_PROCESSING_RESOLUTION_METERS * 0.9 else cell


def _check_window_size(radius, resolution):
    if radius is None:
        return
    max_window_px = int(2 * radius / _processing_resolution(resolution)) + 1
    if max_window_px > _MAX_WINDOW_PX:
        raise QgsProcessingException(
            f"Radius {radius}m at resolution {min(resolution):.4f}m would build a "
            f"{max_window_px}-pixel kernel — exceeds safety cap of {_MAX_WINDOW_PX}. "
            f"Reduce the radius or downsample first."
        )


def _clamp_to_dsm(gdal, dtm_path, src_ds, nodata):
    """Keep the back-warped DTM at or below the input DSM, block by block."""
    dtm_ds = gdal.Open(dtm_path, gdal.GA_Update)
    dtm_band = dtm_ds.GetRasterBand(1)
    dsm_band = src_ds.GetRasterBand(1)
    write_nodata = _safe_nodata(nodata)
    for tile in plan_tiles((dtm_ds.RasterYSize, dtm_ds.RasterXSize), 2048, halo=0):
        w = tile.core
        dtm = dtm_band.ReadAsArray(w.col_off, w.row_off, w.width, w.height).astype(np.float32)
        dsm = dsm_band.ReadAsArray(w.col_off, w.row_off, w.width, w.height).astype(np.float32)
        valid = (dtm != write_nodata) & (dsm != nodata) & np.isfinite(dsm)
        dtm[valid] = np.minimum(dtm[valid], dsm[valid])
        dtm_band.WriteArray(dtm, w.col_off, w.row_off)
    dtm_band.FlushCache()
    dtm_ds = None
