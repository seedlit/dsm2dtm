"""DSM to DTM Processing Algorithm."""

import numpy as np

# Import vendored library (renamed to avoid conflict with plugin package)
from dsm2dtm_core.algorithm import dsm_to_dtm
from qgis.core import (
    QgsProcessingAlgorithm,
    QgsProcessingException,
    QgsProcessingParameterNumber,
    QgsProcessingParameterRasterDestination,
    QgsProcessingParameterRasterLayer,
)


def _utm_epsg_for(lon: float, lat: float) -> int:
    """EPSG code for the standard UTM zone containing (lon, lat).

    Wraps lon=180 to zone 1 (the antimeridian), matching the convention used by
    proj/UTM tooling. Returns 326xx for the northern hemisphere, 327xx for the
    southern.
    """
    zone = int((lon + 180.0) / 6.0) % 60 + 1
    base = 32700 if lat < 0 else 32600
    return base + zone


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
            "• Radius: Kernel radius in meters. Objects larger than 2× this value "
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
                type=QgsProcessingParameterNumber.Double,
                defaultValue=40.0,
                minValue=1.0,
                maxValue=500.0,
            )
        )
        self.addParameter(
            QgsProcessingParameterNumber(
                self.SLOPE,
                "Slope (0=auto, 0.01-1.0=manual)",
                type=QgsProcessingParameterNumber.Double,
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

        Args:
            parameters: Algorithm parameters from the dialog.
            context: Processing context.
            feedback: Feedback object for progress reporting.

        Returns:
            Dictionary with output layer path.
        """
        # Get parameters
        input_layer = self.parameterAsRasterLayer(parameters, self.INPUT, context)
        radius = self.parameterAsDouble(parameters, self.RADIUS, context)
        slope = self.parameterAsDouble(parameters, self.SLOPE, context)
        output_path = self.parameterAsOutputLayer(parameters, self.OUTPUT, context)

        if input_layer is None:
            raise QgsProcessingException("Invalid input raster layer")

        feedback.pushInfo(f"Processing: {input_layer.source()}")
        feedback.pushInfo(f"Radius: {radius}m, Slope: {'auto' if slope == 0 else slope}")

        # Read raster data using QGIS/GDAL
        provider = input_layer.dataProvider()
        extent = input_layer.extent()
        rows = input_layer.height()
        cols = input_layer.width()

        feedback.pushInfo(f"Raster size: {cols}x{rows} pixels")

        # Get nodata value
        nodata = provider.sourceNoDataValue(1)
        if nodata is None or np.isnan(nodata):
            nodata = -9999.0
            feedback.pushInfo("No nodata value found, using -9999.0")

        # Read raster data via GDAL's NumPy bridge — single call, native speed.
        # The previous pixel-by-pixel block.value() loop was O(rows*cols) Python calls
        # and unusable on real rasters (~10^8 calls for a 10k x 10k input).
        from osgeo import gdal, osr

        feedback.pushInfo("Reading raster data...")
        src_ds = gdal.Open(input_layer.source())
        if src_ds is None:
            raise QgsProcessingException(f"Could not open input raster: {input_layer.source()}")

        # The PMF algorithm interprets cell_size as meters, so geographic (lat/lon)
        # rasters must be reprojected to a metric CRS first. We pick the appropriate
        # UTM zone, warp to it, process, and warp the result back at write time.
        input_crs = input_layer.crs()
        utm_warp_ctx = None  # filled when we reproject; tells the writer to warp back

        try:
            if input_crs.isGeographic():
                center_lon = (extent.xMinimum() + extent.xMaximum()) / 2.0
                center_lat = (extent.yMinimum() + extent.yMaximum()) / 2.0
                utm_epsg = _utm_epsg_for(center_lon, center_lat)
                feedback.pushInfo(
                    f"Input is geographic ({input_crs.authid()}). Reprojecting to EPSG:{utm_epsg} for processing..."
                )

                utm_ds = gdal.Warp(
                    "",
                    src_ds,
                    format="MEM",
                    dstSRS=f"EPSG:{utm_epsg}",
                    srcNodata=nodata,
                    dstNodata=nodata,
                    resampleAlg=gdal.GRA_Bilinear,
                )
                if utm_ds is None:
                    raise QgsProcessingException(f"Failed to reproject input to EPSG:{utm_epsg}")

                dsm = utm_ds.GetRasterBand(1).ReadAsArray()
                if dsm is None:
                    raise QgsProcessingException("Failed to read reprojected raster data")
                dsm = np.ascontiguousarray(dsm, dtype=np.float32)

                gt = utm_ds.GetGeoTransform()
                resolution = (abs(gt[1]), abs(gt[5]))
                # Save what we need to warp the DTM back at write time.
                utm_warp_ctx = {
                    "geotransform": gt,
                    "projection": utm_ds.GetProjection(),
                    "cols": utm_ds.RasterXSize,
                    "rows": utm_ds.RasterYSize,
                }
                utm_ds = None
            else:
                dsm = src_ds.GetRasterBand(1).ReadAsArray()
                if dsm is None:
                    raise QgsProcessingException("Failed to read raster band data")
                dsm = np.ascontiguousarray(dsm, dtype=np.float32)
                resolution = (extent.width() / cols, extent.height() / rows)
        finally:
            src_ds = None  # close source

        if feedback.isCanceled():
            return {}

        feedback.setProgress(10)

        x_res, y_res = resolution
        feedback.pushInfo(f"Processing resolution: {x_res:.4f}m x {y_res:.4f}m")

        feedback.pushInfo("Running DSM to DTM conversion...")

        # Run algorithm
        try:
            dtm = dsm_to_dtm(
                dsm,
                resolution=resolution,
                nodata=nodata,
                kernel_radius_meters=radius if radius > 0 else None,
                slope=slope if slope > 0 else None,
            )
        except Exception as e:
            raise QgsProcessingException(f"Algorithm failed: {str(e)}")

        feedback.setProgress(80)
        feedback.pushInfo("Writing output raster...")

        if utm_warp_ctx is not None:
            # Geographic input: dtm is in UTM. Wrap it in a MEM dataset, then warp
            # back to the original CRS, matching the input raster's grid exactly.
            mem_drv = gdal.GetDriverByName("MEM")
            utm_dtm_ds = mem_drv.Create("", utm_warp_ctx["cols"], utm_warp_ctx["rows"], 1, gdal.GDT_Float32)
            utm_dtm_ds.SetGeoTransform(utm_warp_ctx["geotransform"])
            utm_dtm_ds.SetProjection(utm_warp_ctx["projection"])
            utm_band = utm_dtm_ds.GetRasterBand(1)
            utm_band.WriteArray(dtm)
            utm_band.SetNoDataValue(float(nodata))

            out_ds = gdal.Warp(
                output_path,
                utm_dtm_ds,
                format="GTiff",
                dstSRS=input_layer.crs().toWkt(),
                outputBounds=(extent.xMinimum(), extent.yMinimum(), extent.xMaximum(), extent.yMaximum()),
                width=cols,
                height=rows,
                srcNodata=nodata,
                dstNodata=nodata,
                resampleAlg=gdal.GRA_Bilinear,
            )
            utm_dtm_ds = None
            if out_ds is None:
                raise QgsProcessingException(f"Could not create output file: {output_path}")
            out_ds = None
        else:
            # Projected input: write the DTM directly with the input grid.
            driver = gdal.GetDriverByName("GTiff")
            out_ds = driver.Create(output_path, cols, rows, 1, gdal.GDT_Float32)
            if out_ds is None:
                raise QgsProcessingException(f"Could not create output file: {output_path}")

            xmin = extent.xMinimum()
            ymax = extent.yMaximum()
            pixel_width = extent.width() / cols
            pixel_height = extent.height() / rows
            out_ds.SetGeoTransform([xmin, pixel_width, 0.0, ymax, 0.0, -pixel_height])

            srs = osr.SpatialReference()
            srs.ImportFromWkt(input_layer.crs().toWkt())
            out_ds.SetProjection(srs.ExportToWkt())

            out_band = out_ds.GetRasterBand(1)
            out_band.WriteArray(dtm)
            out_band.SetNoDataValue(float(nodata))
            out_band.FlushCache()
            out_ds = None

        feedback.setProgress(100)
        feedback.pushInfo("Done!")

        return {self.OUTPUT: output_path}
