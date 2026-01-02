from pathlib import Path

import pooch
import pytest

# URL to the directory containing the file, or the direct link
DATA_URL = "https://github.com/seedlit/dsm2dtm/releases/download/test-data-v0.1/"
DATA_FILENAME = "dsm2dtm_test_data.zip"


@pytest.fixture(scope="session")
def test_data_dir():
    """
    Downloads and caches the test data zip file from GitHub Releases.
    Returns the path to the directory containing the unzipped files.
    """
    p = pooch.create(
        path=pooch.os_cache("dsm2dtm"),
        base_url=DATA_URL,
        registry={
            DATA_FILENAME: "sha256:9ffb44da60e04584bd5ccf2ae743c2798f2dfe39e002a33ac905d00ffeae92b1",
        },
    )
    paths = p.fetch(DATA_FILENAME, processor=pooch.Unzip())
    if not paths:
        pytest.fail("Downloaded archive appears to be empty.")
    return Path(paths[0]).parent


@pytest.fixture(scope="session")
def real_dsm_path(test_data_dir):
    """
    Finds the DSM file within the downloaded test data.
    Adjust the filename 'test_dsm.tif' if the zip contains something else.
    """
    tiffs = list(test_data_dir.rglob("*.tif"))
    if not tiffs:
        pytest.fail(f"No .tif files found in downloaded test data at {test_data_dir}")
    # TODO: we will return all paths and run in a loop
    return str(tiffs[0])
