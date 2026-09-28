import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin
from rasterio.warp import calculate_default_transform, reproject

from dsm2dtm import core, generate_dtm_file


@pytest.fixture
def projected_dsm_path(tmp_path):
    rng = np.random.default_rng(1)
    yy, xx = np.mgrid[0:700, 0:600]
    data = (400 + 0.04 * xx + 2 * np.sin(yy / 70)).astype(np.float32)
    for _ in range(40):
        r, c = rng.integers(0, 660), rng.integers(0, 560)
        size = rng.integers(8, 40)
        data[r : r + size, c : c + size] += rng.uniform(4, 20)
    data[:, :20] = -9999.0
    path = tmp_path / "dsm_utm.tif"
    profile = dict(
        driver="GTiff", height=700, width=600, count=1, dtype="float32", nodata=-9999.0,
        crs="EPSG:32631", transform=from_origin(500000, 4000000, 1.0, 1.0),
    )  # fmt: skip
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(data, 1)
    return path


@pytest.fixture
def geographic_dsm_path(projected_dsm_path, tmp_path):
    path = tmp_path / "dsm_4326.tif"
    with rasterio.open(projected_dsm_path) as src:
        transform, width, height = calculate_default_transform(src.crs, "EPSG:4326", src.width, src.height, *src.bounds)
        profile = src.profile | {"crs": "EPSG:4326", "transform": transform, "width": width, "height": height}
        with rasterio.open(path, "w", **profile) as dst:
            reproject(rasterio.band(src, 1), rasterio.band(dst, 1), src_nodata=-9999.0, dst_nodata=-9999.0)
    return path


def test_streaming_matches_in_memory_for_projected_input(projected_dsm_path, tmp_path):
    expected, _ = core.generate_dtm(str(projected_dsm_path), kernel_radius_meters=15.0, slope=0.1)
    out = tmp_path / "dtm.tif"
    generate_dtm_file(projected_dsm_path, out, kernel_radius_meters=15.0, slope=0.1, tile_size=256, workers=3)
    with rasterio.open(out) as dst, rasterio.open(projected_dsm_path) as src:
        assert dst.crs == src.crs and dst.transform == src.transform and dst.shape == src.shape
        assert dst.nodata == -9999.0 and dst.dtypes[0] == "float32"
        np.testing.assert_allclose(dst.read(1), expected, atol=1e-4)


def test_streaming_geographic_input_preserves_grid_and_stays_below_dsm(geographic_dsm_path, tmp_path):
    out = tmp_path / "dtm.tif"
    generate_dtm_file(geographic_dsm_path, out, kernel_radius_meters=15.0, tile_size=256)
    with rasterio.open(out) as dst, rasterio.open(geographic_dsm_path) as src:
        assert dst.crs == src.crs and dst.transform == src.transform and dst.shape == src.shape
        dtm, dsm = dst.read(1), src.read(1)
    valid = (dsm != -9999.0) & (dtm != -9999.0)
    assert valid.mean() > 0.5
    assert np.all(dtm[valid] <= dsm[valid])
    assert np.mean(dsm[valid] - dtm[valid] > 3.0) > 0.02


def test_streaming_refuses_to_overwrite_input(projected_dsm_path):
    with pytest.raises(ValueError):
        generate_dtm_file(projected_dsm_path, projected_dsm_path)


def test_cli_uses_streaming_path_and_matches_in_memory(projected_dsm_path, tmp_path, monkeypatch):
    expected, _ = core.generate_dtm(str(projected_dsm_path))
    monkeypatch.setattr("sys.argv", ["dsm2dtm", "--dsm", str(projected_dsm_path), "--out_dir", str(tmp_path / "out")])
    core.main_cli()
    with rasterio.open(tmp_path / "out" / "dsm_utm_dtm.tif") as dst:
        np.testing.assert_allclose(dst.read(1), expected, atol=1e-4)
