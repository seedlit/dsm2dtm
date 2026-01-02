import numpy as np
import pytest
import rasterio

from dsm2dtm.core import generate_dtm


def calculate_metrics(predicted: np.ndarray, actual: np.ndarray, nodata: float = -9999.0):
    """Calculate RMSE, MAE, and Bias between predicted and actual arrays."""
    # Mask invalid data in either
    mask = (predicted != nodata) & (actual != nodata)

    if not np.any(mask):
        return None

    p = predicted[mask]
    a = actual[mask]

    diff = p - a
    rmse = np.sqrt(np.mean(diff**2))
    mae = np.mean(np.abs(diff))
    bias = np.mean(diff)

    return {"rmse": rmse, "mae": mae, "bias": bias, "count": np.sum(mask)}


@pytest.mark.parametrize(
    "dsm_name, gt_name, expected_rmse",
    [
        ("dsm_1m_istanbul_hilly_urban.tif", "dtm_1m_istanbul_hilly_urban.tif", 5.0),
        ("dsm_50cm_river_and_urban.tif", "dtm_50cm_river_and_urban.tif", 5.0),
        ("dsm_50cm_vegetaion_urban.tif", "dtm_50cm_vegetation_urban.tif", 10.0),
    ],
)
def test_integration_accuracy(test_data_dir, dsm_name, gt_name, expected_rmse):
    """
    Run full pipeline and compare against Ground Truth DTM.
    """
    dsm_path = str(test_data_dir / dsm_name)
    gt_path = str(test_data_dir / gt_name)

    print(f"\nProcessing: {dsm_name} -> Comparing with: {gt_name}")

    # 1. Generate DTM
    dtm_pred, profile = generate_dtm(dsm_path)

    # 2. Load Ground Truth
    with rasterio.open(gt_path) as src:
        dtm_gt = src.read(1)
        gt_nodata = src.nodata if src.nodata is not None else -9999.0

        # Ensure dimensions match
        if dtm_pred.shape != dtm_gt.shape:
            # If shapes differ slightly due to different processing/cropping in creation,
            # we might need to crop or reproject.
            # For this test data, we assume they are pixel-aligned pairs.
            pytest.fail(f"Dimension mismatch! Pred: {dtm_pred.shape}, GT: {dtm_gt.shape}")

    # 3. Compare
    metrics = calculate_metrics(dtm_pred, dtm_gt, nodata=gt_nodata)

    assert metrics is not None, "No valid overlapping pixels found."

    print(f"Metrics for {dsm_name}:")
    print(f"  RMSE: {metrics['rmse']:.4f} m")
    print(f"  MAE:  {metrics['mae']:.4f} m")
    print(f"  Bias: {metrics['bias']:.4f} m")
    print(f"  Pixels: {metrics['count']}")

    # 4. Assertions
    # We use a relatively loose RMSE threshold initially because:
    # a) The algorithm parameters (slope, radius) are defaults and might not be optimal for all scenes.
    # b) "Ground Truth" DTMs might have different generation methods/artifacts.
    assert metrics["rmse"] < expected_rmse, f"RMSE {metrics['rmse']:.2f} exceeds threshold {expected_rmse}"
