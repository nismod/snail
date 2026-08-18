import gc
import os
from pathlib import Path

import numpy as np
import rasterio
import pytest
import xarray
from numpy.testing import assert_array_equal
from rasterio.crs import CRS

from snail.intersection import GridDefinition
from snail.io import (
    read_raster_band_data,
    read_raster_metadata,
    write_grid_to_raster,
)


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


def test_read_raster_metadata_from_dataarray(sample_dataarray):
    data_array, transform = sample_dataarray
    grid, bands = read_raster_metadata(data_array)

    assert grid.transform == tuple(transform)[:6]
    assert grid.width == data_array.sizes[data_array.rio.x_dim]
    assert grid.height == data_array.sizes[data_array.rio.y_dim]
    assert bands == (1,)


def test_read_raster_metadata_from_multiband_dataarray(sample_dataarray):
    xr = pytest.importorskip("xarray")
    data_array, _ = sample_dataarray
    multi = xr.concat([data_array, data_array + 5], dim="band")
    multi.coords["band"] = [1, 2]

    grid, bands = read_raster_metadata(multi)

    assert grid.width == data_array.sizes[data_array.rio.x_dim]
    assert bands == (1, 2)


def test_read_raster_band_data_from_dataarray(sample_dataarray):
    data_array, _ = sample_dataarray
    result = read_raster_band_data(data_array, band_number=1)

    assert_array_equal(result.values, data_array.values)
    assert result.dims == data_array.dims


@pytest.mark.skipif(not Path("/proc").is_dir(), reason="Linux only")
def test_lazy_raster_reads_do_not_leak_file_descriptors():
    path = Path(__file__).parent / "integration" / "range.tif"
    gc.collect()
    initial_fds = len(os.listdir(f"/proc/{os.getpid()}/fd"))

    for _ in range(200):
        result = read_raster_band_data(path, lazy=True)
        del result

    gc.collect()
    final_fds = len(os.listdir(f"/proc/{os.getpid()}/fd"))
    assert final_fds <= initial_fds + 4


def test_read_raster_band_data_rejects_zero_band_for_lazy_path():
    path = Path(__file__).parent / "integration" / "range.tif"

    with pytest.raises(ValueError, match="band_number must be >= 1"):
        read_raster_band_data(path, band_number=0, lazy=True)


def test_read_raster_band_data_rejects_zero_band_for_lazy_dataarray(
    sample_dataarray,
):
    data_array, _ = sample_dataarray

    with pytest.raises(ValueError, match="band_number must be >= 1"):
        read_raster_band_data(data_array, band_number=0, lazy=True)


def test_read_raster_band_data_from_multiband_dataarray(sample_dataarray):
    data_array, _ = sample_dataarray

    second_band = data_array + 7
    multi = xarray.concat([data_array, second_band], dim="band")
    multi.coords["band"] = [1, 2]

    result = read_raster_band_data(multi, band_number=2)
    assert_array_equal(result.values, second_band.values)
    assert result.dims == data_array.dims

    with pytest.raises(ValueError):
        read_raster_band_data(multi, band_number=3)

    with pytest.raises(ValueError, match="band_number must be >= 1"):
        read_raster_band_data(multi, band_number=0)
