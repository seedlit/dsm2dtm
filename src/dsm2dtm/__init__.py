"""Generate DTM (Digital Terrain Model) from DSM (Digital Surface Model)"""

from .core import generate_dtm, save_dtm

__all__ = ["generate_dtm", "save_dtm"]
__version__ = "0.3.0"
