"""Main plugin class for DSM to DTM."""

from qgis.core import QgsApplication

from .provider import Dsm2DtmProvider


class Dsm2DtmPlugin:
    """QGIS Plugin implementation for dsm2dtm.

    This plugin registers a Processing Provider that adds the
    DSM to DTM algorithm to the QGIS Processing Toolbox.
    """

    def __init__(self, iface):
        """Initialize the plugin.

        Args:
            iface: QGIS interface object.
        """
        self.iface = iface
        self.provider = None

    def initProcessing(self):
        """Register the processing provider with QGIS."""
        self.provider = Dsm2DtmProvider()
        QgsApplication.processingRegistry().addProvider(self.provider)

    def initGui(self):
        """Called when plugin is loaded into QGIS."""
        self.initProcessing()

    def unload(self):
        """Called when plugin is unloaded from QGIS."""
        if self.provider:
            QgsApplication.processingRegistry().removeProvider(self.provider)
