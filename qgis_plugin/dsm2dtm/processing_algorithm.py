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

        # Read raster block
        block = provider.block(1, extent, cols, rows)

        # Convert to numpy array
        feedback.pushInfo("Reading raster data...")
        dsm = np.zeros((rows, cols), dtype=np.float32)
        for r in range(rows):
            if feedback.isCanceled():
                return {}
            for c in range(cols):
                dsm[r, c] = block.value(r, c)

        feedback.setProgress(10)

        # Get resolution
        x_res = extent.width() / cols
        y_res = extent.height() / rows
        resolution = (x_res, y_res)
        feedback.pushInfo(f"Resolution: {x_res:.2f}m x {y_res:.2f}m")

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

        # Write output using GDAL (native to QGIS)
        from osgeo import gdal, osr

        driver = gdal.GetDriverByName("GTiff")
        out_ds = driver.Create(output_path, cols, rows, 1, gdal.GDT_Float32)
        if out_ds is None:
            raise QgsProcessingException(f"Could not create output file: {output_path}")

        # Set transform
        xmin = extent.xMinimum()
        ymax = extent.yMaximum()
        pixel_width = extent.width() / cols
        pixel_height = extent.height() / rows
        # GDAL transform: [top_left_x, w_resolution, 0, top_left_y, 0, n_s_resolution (negative)]
        out_transform = [xmin, pixel_width, 0.0, ymax, 0.0, -pixel_height]
        out_ds.SetGeoTransform(out_transform)

        # Set projection
        srs = osr.SpatialReference()
        srs.ImportFromWkt(input_layer.crs().toWkt())
        out_ds.SetProjection(srs.ExportToWkt())

        # Write data
        out_band = out_ds.GetRasterBand(1)
        out_band.WriteArray(dtm)
        out_band.SetNoDataValue(float(nodata))

        # Close dataset to flush to disk
        out_band.FlushCache()
        out_ds = None

        feedback.setProgress(100)
        feedback.pushInfo("Done!")

        return {self.OUTPUT: output_path}
