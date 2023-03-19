from src import dsm2dtm


TEST_DSM = "data/sample_dsm.tif"


def test_get_raster_resolution():
    x_res, y_res = dsm2dtm.get_raster_resolution(TEST_DSM)
    assert x_res == 2.513751187083714e-07
    assert y_res == 2.51398813677825e-07