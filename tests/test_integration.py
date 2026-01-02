import numpy as np
import pytest
import rasterio

from dsm2dtm.core import generate_dtm, save_dtm


def test_integration_real_data(real_dsm_path, tmp_path):
    """
    Run the full pipeline on a real-world DSM downloaded from GitHub.
    This ensures the code works on realistic data distributions and sizes.
    """
    print(f"Running integration test on: {real_dsm_path}")

    # 1. Run Generation
    # Use default parameters for now
    dtm, profile = generate_dtm(real_dsm_path)

    # 2. Basic Sanity Checks
    assert isinstance(dtm, np.ndarray)
    assert dtm.ndim == 2
    assert profile["count"] == 1

    # Check dimensions match input
    with rasterio.open(real_dsm_path) as src:
        assert dtm.shape == (src.height, src.width)
        dsm_data = src.read(1)
        nodata = src.nodata if src.nodata is not None else -9999.0

    # 3. Value Checks
    # DTM should generally be <= DSM (it's bare earth)
    # Ignore nodata
    valid_mask = (dsm_data != nodata) & (dtm != nodata)

    if np.any(valid_mask):
        # Allow small tolerance for smoothing effects where DTM > DSM slightly
        # But generally DTM <= DSM.
        # Let's count how many pixels are significantly higher
        diff = dtm[valid_mask] - dsm_data[valid_mask]

        # We expect DTM <= DSM. So diff <= 0.
        # Positive diff implies DTM is above ground surface (bad, but happens with smoothing).
        # Check that 99% of points are not drastically above DSM
        high_points = np.sum(diff > 0.5)  # Points more than 50cm above DSM
        total_points = np.sum(valid_mask)
        fraction_bad = high_points / total_points

        assert fraction_bad < 0.05, f"Too many DTM points are significantly higher than DSM ({fraction_bad:.1%})"

        print(f"Integration Test Stats: Processed {total_points} valid pixels.")
        print(f"Max Elevation: {np.max(dtm[valid_mask]):.2f}m")
        print(f"Mean Elevation: {np.mean(dtm[valid_mask]):.2f}m")
    else:
        pytest.fail("Resulting DTM has no valid data overlap with DSM.")

    # 4. Save check (optional, just to ensure save_dtm works with this profile)
    out_file = tmp_path / "integration_result.tif"
    save_dtm(dtm, profile, str(out_file))
    assert out_file.exists()
