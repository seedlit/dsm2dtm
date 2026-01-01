"""
Constants for the DSM to DTM conversion pipeline.
"""

# No-data value
DEFAULT_NODATA = -99999.0

# Base parameters defined for 1.0 meter resolution
BASE_RESOLUTION = 1.0

# Progressive Morphological Filter (PMF) Parameters
PMF_INITIAL_WINDOW_METERS = 3.0  # Start with 3m x 3m window
PMF_MAX_WINDOW_METERS = 161.0  # Max window size ~161m
PMF_SLOPE = 0.05  # Dimensionless (rise/run)
PMF_INITIAL_THRESHOLD = 0.1  # Meters
PMF_MAX_THRESHOLD = 20.0  # Meters - Relaxed for steep terrain

# Smoothing Refinement Parameters
REFINEMENT_SMOOTH_SIGMA_METERS = 5.0
REFINEMENT_ELEVATION_THRESHOLD = 1.0
FINAL_SMOOTH_SIGMA_METERS = 0.5

# Processing Resolution
# If input DSM resolution is finer than this (e.g. 0.05m), we downsample to this resolution
# for the filtering steps to improve stability and performance, then upsample the result.
MIN_PROCESSING_RESOLUTION_METERS = 0.5

# Gap Filling Parameters
GAP_FILL_MAX_SEARCH_DISTANCE_METERS = 100.0
GAP_FILL_SMOOTHING_ITERATIONS = 0

# Default Kernel Radius for CLI/Main (Meters)
DEFAULT_KERNEL_RADIUS_METERS = 40.0
