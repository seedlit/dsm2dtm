# dsm2dtm

<img align="right" width = 200 height=80 src="./data/logo.png">

**Generate DTM (Digital Terrain Model) from DSM (Digital Surface Model)**

[![CI](https://github.com/seedlit/dsm2dtm/actions/workflows/ci.yml/badge.svg)](https://github.com/seedlit/dsm2dtm/actions/workflows/ci.yml)
[![Python Version](https://img.shields.io/badge/python-3.12%2B-blue)](https://python.org)
[![License](https://img.shields.io/github/license/seedlit/dsm2dtm?style=flat-square)](LICENSE)

A modern, high-performance, pure-Python library for generating Digital Terrain Models (DTM) from Digital Surface Models (DSM).

Key features:
*   **Pure Python**: No external binary dependencies like SAGA GIS or GDAL CLI tools. Just `pip install`.
*   **Memory Efficient**: Processes large rasters block-by-block, ensuring low memory usage.
*   **Fast**: Optimized `numpy` and `scipy` operations for slope calculation, morphological filtering, and smoothing.

## Installation

### From Source

1.  Clone the repository:
    ```bash
    git clone https://github.com/seedlit/dsm2dtm.git
    cd dsm2dtm
    ```

2.  Install using `uv` (recommended) or `pip`:
    ```bash
    # Using uv (faster)
    uv pip install .

    # Using standard pip
    pip install .
    ```

### Development Setup

To contribute or run tests, install the development dependencies:

```bash
uv pip install -e '.[test,dev]'
pre-commit install
```

## Usage

### Command Line Interface (CLI)

After installation, use the `dsm2dtm` command:

```bash
dsm2dtm --dsm data/sample_dsm.tif --out_dir results/
```

**Arguments:**
*   `--dsm`: Path to the input Digital Surface Model (GeoTIFF).
*   `--out_dir`: Directory where the generated DTM will be saved. Defaults to `generated_dtm`.

### Python API

You can also use `dsm2dtm` as a library in your own Python scripts:

```python
from dsm2dtm.core import main

dsm_path = "path/to/dsm.tif"
out_dir = "path/to/output"

# Generate the DTM
dtm_path = main(dsm_path, out_dir)

print(f"DTM generated at: {dtm_path}")
```

## How it Works

The pipeline performs the following steps to extract the terrain:

1.  **Slope Calculation**: Generates a slope raster from the input DSM.
2.  **Morphological Filtering**: Uses a "Grey Opening" operation to estimate the bare earth surface (DTM) by removing non-ground features (like trees and buildings).
3.  **Smoothing**: Applies a Gaussian filter to smoothen the estimated ground surface.
4.  **Difference Calculation**: Subtracts the smoothed ground from the initial ground estimate.
5.  **Thresholding**: Replaces values in the ground model where the difference exceeds a threshold.
6.  **Noise Removal**: Removes statistical outliers (spikes).
7.  **Gap Filling**: Interpolates remaining holes to produce a continuous DTM.

All steps are performed using block-wise processing to handle rasters larger than available RAM.

## Examples

### Example 1: Flat Terrain
Input DSM vs. Generated DTM over a flat area.
![example](./results/result.png)

### Example 2: Hillside Terrain
Comparison of Input DSM, Generated DTM, and Ground Truth DTM (Lidar derived).
![example](./results/example2_dsm2dtm_hillside.png)

## License

[MIT License](LICENSE)
