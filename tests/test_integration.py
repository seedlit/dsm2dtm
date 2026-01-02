import numpy as np
import pytest
import rasterio

from dsm2dtm.core import generate_dtm


def calculate_metrics(predicted: np.ndarray, actual: np.ndarray, nodata: float = -9999.0):
    """Calculate RMSE, MAE, and Bias between predicted and actual arrays."""
    mask = (predicted != nodata) & (actual != nodata)
    if not np.any(mask):
        return None
    predicted = predicted[mask]
    actual = actual[mask]
    diff = predicted - actual
    rmse = np.sqrt(np.mean(diff**2))
    mae = np.mean(np.abs(diff))
    bias = np.mean(diff)
    return {"rmse": rmse, "mae": mae, "bias": bias, "count": np.sum(mask)}


@pytest.mark.parametrize(
    "dsm_name, gt_name, expected_rmse",
    [
        ("dsm_1m_istanbul_hilly_urban.tif", "dtm_1m_istanbul_hilly_urban.tif", 5.0),
        ("dsm_50cm_river_and_urban.tif", "dtm_50cm_river_and_urban.tif", 2.0),
        ("dsm_50cm_vegetaion_urban.tif", "dtm_50cm_vegetation_urban.tif", 8.0),
    ],
)
# TODO: look at why RMSE is so high?
def test_integration_accuracy(test_data_dir, dsm_name, gt_name, expected_rmse):
    """
    Run full pipeline and compare against Ground Truth DTM.
    """
    dsm_path = str(test_data_dir / dsm_name)
    gt_path = str(test_data_dir / gt_name)
    print(f"\nProcessing: {dsm_name} -> Comparing with: {gt_name}")
    dtm_pred, profile = generate_dtm(dsm_path)
    with rasterio.open(gt_path) as src:
        dtm_gt = src.read(1)
        gt_nodata = src.nodata if src.nodata is not None else -9999.0
        if dtm_pred.shape != dtm_gt.shape:
            pytest.fail(f"Dimension mismatch! Pred: {dtm_pred.shape}, GT: {dtm_gt.shape}")
    metrics = calculate_metrics(dtm_pred, dtm_gt, nodata=gt_nodata)
    assert metrics is not None, "No valid overlapping pixels found."
    print(f"Metrics for {dsm_name}:")
    print(f"  RMSE: {metrics['rmse']:.4f} m")
    print(f"  MAE:  {metrics['mae']:.4f} m")
    print(f"  Bias: {metrics['bias']:.4f} m")
    print(f"  Pixels: {metrics['count']}")
    assert metrics["rmse"] < expected_rmse, f"RMSE {metrics['rmse']:.2f} exceeds threshold {expected_rmse}"


@pytest.mark.parametrize(
    "dsm_name",
    [
        "dsm_1m_istanbul_hilly_urban.tif",
        "dsm_50cm_river_and_urban.tif",
        "dsm_50cm_vegetaion_urban.tif",
    ],
)
def test_integration_monotonicity(test_data_dir, dsm_name):
    """
    Verify that DTM <= DSM (Monotonicity Check).
    The Digital Terrain Model should generally represent the bare earth,
    so it should not be significantly higher than the Surface Model.
    """
    dsm_path = str(test_data_dir / dsm_name)
    print(f"\nMonotonicity Check: {dsm_name}")
    dtm_pred, profile = generate_dtm(dsm_path)
    with rasterio.open(dsm_path) as src:
        dsm_data = src.read(1)
        dsm_nodata = src.nodata if src.nodata is not None else -9999.0
    valid_mask = (dsm_data != dsm_nodata) & (dtm_pred != profile["nodata"])
    if not np.any(valid_mask):
        pytest.fail("No valid overlap between DSM and DTM.")
    diff = dtm_pred[valid_mask] - dsm_data[valid_mask]
    tolerance = 0.1  # allow 10cm tolerance for smoothing
    violations = np.sum(diff > tolerance)
    total = np.sum(valid_mask)
    fraction = violations / total
    print(f"  Violations (> {tolerance}m): {violations} / {total} ({fraction:.2%})")
    # Strict assertion: < 1.5% of pixels can float above DSM
    assert fraction < 0.015, f"DTM floats above DSM in {fraction:.1%} of pixels!"
