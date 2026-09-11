import gc
import os
import time
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
import rasterio
from pandas.testing import assert_series_equal
from rasterio.transform import from_origin
from rasterio.windows import Window
from shapely.geometry import Point

from snail.intersection import read_raster_values_for_splits
from snail.overlay import overlay_raster


class RecordingRaster:
    """Small rasterio-compatible source which records requested windows."""

    indexes = (1, 2)
    dtypes = ("float64", "float64")

    def __init__(self, width=1024, height=1024, block_shape=(256, 256)):
        self.width = width
        self.height = height
        self.block_shapes = (block_shape,) * len(self.indexes)
        self.windows = []

    def read(self, indexes, window):
        self.windows.append(window)
        rows = np.arange(window.row_off, window.row_off + window.height)[:, None]
        cols = np.arange(window.col_off, window.col_off + window.width)[None, :]
        values = rows * self.width + cols
        return np.stack([values + band * 1_000_000 for band in indexes])


class GuardedDataset:
    """Rasterio dataset proxy which rejects full or oversized reads."""

    def __init__(self, dataset, max_bytes):
        self._dataset = dataset
        self.max_bytes = max_bytes
        self.windows = []
        for name in (
            "block_shapes",
            "crs",
            "dtypes",
            "height",
            "indexes",
            "name",
            "transform",
            "width",
        ):
            setattr(self, name, getattr(dataset, name))

    def read(self, indexes, window=None):
        assert window is not None, "raster attribution attempted an unwindowed read"
        data = self._dataset.read(indexes=indexes, window=window)
        assert data.nbytes <= self.max_bytes
        self.windows.append(window)
        return data


def test_reader_preserves_order_duplicates_indices_and_invalid_cells():
    raster = RecordingRaster()
    splits = pd.DataFrame(
        {"index_i": [0, 1023, -1, 0, 3000], "index_j": [0, 1023, 0, 0, 0]},
        index=["first", "last", "outside", "duplicate", "far"],
    )

    values = read_raster_values_for_splits(splits, raster, [1, 2])

    assert_series_equal(
        values[1],
        pd.Series(
            [1_000_000.0, 2_048_575.0, np.nan, 1_000_000.0, np.nan],
            index=splits.index,
        ),
    )
    assert_series_equal(
        values[2],
        pd.Series(
            [2_000_000.0, 3_048_575.0, np.nan, 2_000_000.0, np.nan],
            index=splits.index,
        ),
    )
    assert raster.windows


def test_reader_has_path_and_open_dataset_parity(two_band_raster):
    splits = pd.DataFrame(
        {"index_i": [0, 3, -1, 0], "index_j": [0, 3, 0, 0]},
        index=[4, 8, 9, 11],
    )
    from_path = read_raster_values_for_splits(splits, two_band_raster, [1, 2])
    with rasterio.open(two_band_raster) as dataset:
        from_dataset = read_raster_values_for_splits(splits, dataset, [1, 2])

    for band in (1, 2):
        assert_series_equal(from_path[band], from_dataset[band])


def test_reader_uses_one_tight_window_when_extent_fits_budget():
    raster = RecordingRaster()
    splits = pd.DataFrame({"index_i": [10, 30], "index_j": [20, 40]})

    read_raster_values_for_splits(splits, raster, [1, 2], max_raster_memory_mb=1)

    assert raster.windows == [Window(10, 20, 21, 21)]


def test_reader_only_reads_occupied_blocks_with_bounded_buffers():
    raster = RecordingRaster()
    splits = pd.DataFrame({"index_i": [1, 900], "index_j": [1, 900]})

    values = read_raster_values_for_splits(
        splits, raster, [1, 2], max_raster_memory_mb=1
    )

    assert values[1].tolist() == [1_001_025.0, 1_922_500.0]
    assert len(raster.windows) == 2
    assert all(
        window.width * window.height * 16 <= 1024 * 1024 for window in raster.windows
    )
    assert {(window.col_off, window.row_off) for window in raster.windows} == {
        (0, 0),
        (768, 768),
    }


@pytest.mark.parametrize(
    ("indices_i", "indices_j", "expected"),
    [
        ([1, 300, 900], [1, 1, 900], Window(0, 0, 512, 256)),
        ([1, 1, 900], [1, 300, 900], Window(0, 0, 256, 512)),
    ],
)
def test_reader_coalesces_adjacent_occupied_blocks(indices_i, indices_j, expected):
    raster = RecordingRaster()
    splits = pd.DataFrame({"index_i": indices_i, "index_j": indices_j})

    read_raster_values_for_splits(splits, raster, [1, 2], max_raster_memory_mb=2)

    assert len(raster.windows) == 2
    assert expected in raster.windows


def test_reader_clips_native_blocks_at_raster_edges():
    raster = RecordingRaster(width=1000, height=1000)
    splits = pd.DataFrame({"index_i": [1, 999], "index_j": [1, 999]})

    read_raster_values_for_splits(splits, raster, [1, 2], max_raster_memory_mb=1)

    assert Window(768, 768, 232, 232) in raster.windows


def test_reader_subdivides_oversized_occupied_block():
    raster = RecordingRaster(block_shape=(1024, 1024))
    splits = pd.DataFrame({"index_i": [1, 900], "index_j": [1, 900]})

    values = read_raster_values_for_splits(
        splits, raster, [1, 2], max_raster_memory_mb=1
    )

    assert values[1].tolist() == [1_001_025.0, 1_922_500.0]
    assert len(raster.windows) == 2
    assert all(
        window.width * window.height * 16 <= 1024 * 1024 for window in raster.windows
    )


@pytest.mark.parametrize(
    ("splits", "message"),
    [
        (pd.DataFrame({"index_i": [0.5], "index_j": [0]}), "index_i"),
        (pd.DataFrame({"index_i": [0], "index_j": ["bad"]}), "index_j"),
    ],
)
def test_reader_rejects_non_integer_indices(splits, message):
    with pytest.raises(ValueError, match=message):
        read_raster_values_for_splits(splits, RecordingRaster(), [1])


def test_reader_validates_bands_and_budget():
    splits = pd.DataFrame({"index_i": [0], "index_j": [0]})
    with pytest.raises(ValueError, match="positive"):
        read_raster_values_for_splits(
            splits, RecordingRaster(), [1], max_raster_memory_mb=0
        )
    with pytest.raises(ValueError, match="band"):
        read_raster_values_for_splits(splits, RecordingRaster(), [3])
    with pytest.raises(ValueError, match="one pixel"):
        read_raster_values_for_splits(
            splits, RecordingRaster(), [1, 2], max_raster_memory_mb=0.000001
        )


def test_reader_returns_nan_without_reading_when_all_cells_are_invalid():
    raster = RecordingRaster()
    splits = pd.DataFrame({"index_i": [-1, 5000], "index_j": [0, 5000]})

    values = read_raster_values_for_splits(splits, raster, [1])

    assert values[1].isna().all()
    assert raster.windows == []


def test_reader_scales_to_many_sparse_blocks():
    count = 10_000
    raster = RecordingRaster(
        width=count * 2048,
        height=1,
        block_shape=(1, 1),
    )
    columns = np.arange(count, dtype=np.int64) * 2048
    splits = pd.DataFrame(
        {"index_i": columns, "index_j": np.zeros(count, dtype=np.int64)}
    )

    started = time.perf_counter()
    values = read_raster_values_for_splits(splits, raster, [1], max_raster_memory_mb=1)

    assert len(raster.windows) == count
    assert len(values[1]) == count
    assert time.perf_counter() - started < 5


@pytest.mark.skipif(not Path("/proc").is_dir(), reason="Linux only")
def test_windowed_raster_reads_do_not_leak_file_descriptors(two_band_raster):
    splits = pd.DataFrame({"index_i": [0], "index_j": [0]})
    gc.collect()
    initial_fds = len(os.listdir(f"/proc/{os.getpid()}/fd"))

    for _ in range(100):
        read_raster_values_for_splits(splits, two_band_raster, [1, 2])

    gc.collect()
    final_fds = len(os.listdir(f"/proc/{os.getpid()}/fd"))
    assert final_fds <= initial_fds + 4


def test_sparse_bigtiff_attributes_distant_values_with_bounded_reads(tmp_path):
    path = tmp_path / "huge.tif"
    width = height = 120_000
    expected = [7, 19, 255]
    cells = [(1, 1), (60_000, 70_000), (119_999, 119_999)]
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        width=width,
        height=height,
        count=1,
        dtype="uint8",
        transform=from_origin(0, height, 1, 1),
        crs="EPSG:3857",
        tiled=True,
        blockxsize=512,
        blockysize=512,
        nodata=255,
        BIGTIFF="YES",
        SPARSE_OK="YES",
    ) as dataset:
        for (column, row), value in zip(cells, expected):
            dataset.write(
                np.array([[value]], dtype="uint8"),
                1,
                window=Window(column, row, 1, 1),
            )

    assert width * height > 10_000_000_000
    points = gpd.GeoDataFrame(
        geometry=[Point(column + 0.5, height - row - 0.5) for column, row in cells],
        crs="EPSG:3857",
    )
    with rasterio.open(path) as dataset:
        guarded = GuardedDataset(dataset, max_bytes=1024 * 1024)
        attributed = overlay_raster(points, guarded, bands=[1], max_raster_memory_mb=1)

    assert attributed["huge"].tolist() == expected
    assert attributed["huge"].dtype == np.dtype("uint8")
    assert len(guarded.windows) == 3
