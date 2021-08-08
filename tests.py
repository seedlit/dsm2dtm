import os
import dsm2dtm
import shutil
import gdal


if __name__ == "__main__":

    dsm_path = "data/sample_dsm.tif"
    out_dir = "test_results"
    dtm_path = dsm2dtm.main(dsm_path, out_dir)

    dtm_array = gdal.Open(dtm_path).ReadAsArray()
    assert dtm_array.shape == (286, 315)
    assert dsm2dtm.get_raster_resolution(dtm_path) == (2.5193e-06, 2.5193e-06)
    print("All tests passed!")
    shutil.rmtree(out_dir)
