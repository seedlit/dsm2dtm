"""QGIS Plugin Entry Point for dsm2dtm."""

import os
import sys

# Add vendored libraries to path BEFORE any imports
_ext_libs = os.path.join(os.path.dirname(__file__), "ext_libs")
if _ext_libs not in sys.path:
    sys.path.insert(0, _ext_libs)


def classFactory(iface):
    """QGIS calls this to instantiate the plugin.

    Args:
        iface: QGIS interface object.

    Returns:
        Plugin instance.
    """
    from .plugin import Dsm2DtmPlugin

    return Dsm2DtmPlugin(iface)
