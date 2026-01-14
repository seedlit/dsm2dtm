"""Processing Provider for dsm2dtm algorithms."""

import os

from qgis.core import QgsProcessingProvider
from qgis.PyQt.QtGui import QIcon

from .processing_algorithm import Dsm2DtmAlgorithm


class Dsm2DtmProvider(QgsProcessingProvider):
    """Processing Provider that groups dsm2dtm algorithms.

    This provider appears in the QGIS Processing Toolbox and
    contains all algorithms related to DSM to DTM conversion.
    """

    def loadAlgorithms(self):
        """Register algorithms with this provider."""
        self.addAlgorithm(Dsm2DtmAlgorithm())

    def id(self):
        """Return the unique provider ID."""
        return "dsm2dtm"

    def name(self):
        """Return the provider name shown in the toolbox."""
        return "DSM to DTM"

    def longName(self):
        """Return the full provider name."""
        return "DSM to DTM - Bare Earth Extraction"

    def icon(self):
        """Return the provider icon."""
        icon_path = os.path.join(os.path.dirname(__file__), "icon.png")
        if os.path.exists(icon_path):
            return QIcon(icon_path)
        return QgsProcessingProvider.icon(self)
