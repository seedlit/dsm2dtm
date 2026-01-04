"""Generate DTM (Digital Terrain Model) from DSM (Digital Surface Model)"""

from importlib.metadata import PackageNotFoundError, version

from .core import generate_dtm, save_dtm

__all__ = ["generate_dtm", "save_dtm"]

try:
    __version__ = version("dsm2dtm")
except PackageNotFoundError:
    # Package is not installed
    __version__ = "unknown"
