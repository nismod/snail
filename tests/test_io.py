import numpy as np
import rasterio
from numpy.testing import assert_array_equal
from rasterio.crs import CRS

from snail.intersection import GridDefinition
from snail.io import write_grid_to_raster


def test_write_grid_to_raster(tmp_path):
    grid = GridDefinition(
        crs=CRS.from_epsg(4326),
        width=2,
        height=2,
        transform=(1.0, 0.0, 5.0, 0.0, -1.0, 10.0),
    )
    array = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=float)
    output_path = tmp_path / "length.tif"
    write_grid_to_raster(
        array,
        output_path,
        grid.transform,
        grid.crs,
        dtype="float32",
        nodata=-99.0,
    )
    with rasterio.open(output_path) as dataset:
        data = dataset.read(1)
        assert dataset.count == 1
        assert dataset.height == array.shape[0]
        assert dataset.width == array.shape[1]
        assert tuple(dataset.transform)[:6] == grid.transform
        assert dataset.meta["dtype"] == "float32"
        assert dataset.nodata == -99.0
    assert_array_equal(data, array.astype("float32"))
