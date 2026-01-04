from pathlib import Path

import pooch
import pytest

# URL to the directory containing the file, or the direct link
DATA_URL = "https://github.com/seedlit/dsm2dtm/releases/download/test-data-v0.1/"
DATA_FILENAME = "dsm2dtm_test_data.zip"
SHA256 = "sha256:9ffb44da60e04584bd5ccf2ae743c2798f2dfe39e002a33ac905d00ffeae92b1"


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
            DATA_FILENAME: SHA256,
        },
    )
    paths = p.fetch(DATA_FILENAME, processor=pooch.Unzip())
    if not paths:
        pytest.fail("Downloaded archive appears to be empty.")
    return Path(paths[0]).parent
