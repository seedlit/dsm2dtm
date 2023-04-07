# dsm2dtm
<img align="right" width = 200 height=80 src="./data/logo.png">

This repo generates DTM (Digital Terrain Model) from DSM (Digital Surface Model).

[![GitHub](https://img.shields.io/github/license/seedlit/dsm2dtm?style=flat-square)](https://github.com/seedlit/dsm2dtm/blob/main/LICENSE)
[![Anaconda-Server Badge](https://anaconda.org/conda-forge/dsm2dtm/badges/downloads.svg)](https://anaconda.org/conda-forge/dsm2dtm)
[![Conda Version](https://img.shields.io/conda/vn/conda-forge/dsm2dtm.svg)](https://anaconda.org/conda-forge/dsm2dtm)
[![GitHub contributors](https://img.shields.io/github/contributors/seedlit/dsm2dtm?style=flat-square)](https://github.com/seedlit/dsm2dtm/graphs/contributors)
![Python Version Supported](https://img.shields.io/badge/python-3.5%2B-blue)
[![Anaconda-Server Badge](https://anaconda.org/conda-forge/dsm2dtm/badges/platforms.svg)](https://anaconda.org/conda-forge/dsm2dtm)

## Installation

**Note**: We are unable to install Saga as part of the dependency, as it is not avilable on PyPI or conda. <br/>
To install saga_cmd - `sudo apt update; sudo apt install saga`

### From Conda:
```bash
conda install -c conda-forge dsm2dtm
```
These step are for Linux. This will differ a bit for MacOS and windows.
### From Source

```bash
# Step 1: Clone the repo
% git clone https://github.com/seedlit/dsm2dtm.git
# Step 2: Move in the folder
% cd dsm2dtm
# Step 3: Create a virtual environment
% python3 -m venv venv
# Step 4: Activate the environment
% source venv/bin/activate
# Step 5: Install requirements
% pip install -r requirements.txt
# Step 6: Install saga_cmd
% sudo apt update
% sudo apt install saga
```

## Usage
Run the script dsm2dtm.py and pass the dsm path as argument.
```bash
python dsm2dtm.py --dsm data/sample_dsm.tif
```

### Example1: Input DSM and generated DTM over a flat terrain
![example](./results/result.png)

### Example2: Input DSM, generated DTM, and groundtruth DTM (Lidar derived) over a hillside terrain
DSM was derived from [this point cloud data](https://cloud.rockrobotic.com/share/f42b5b69-c87c-4433-94f8-4bc0d8eaee90#lidar)
![example](./results/example2_dsm2dtm_hillside.png)

## TODO
 - Add tests and coverage
 - Add poetry (with separate dependencies for dev: black, isort, pyest, etc.)
 - Add pre-commit hooks (isort, black, mypy)
 - Add documentation
 - Move test file(s) to remote server OR use gitlfs OR use fake-geo-images
 - Reduce I/O by passing rasterio object instead of raster path
 - Add exception handling
 - use [SAGA python API](https://saga-gis.sourceforge.io/saga_api_python/index.html) instead of command line ineterface (saga_cmd)
