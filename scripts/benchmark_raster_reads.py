"""Report bounded raster attribution reads for representative cell layouts."""

import time

import numpy as np
import pandas as pd
from rasterio.io import MemoryFile
from rasterio.transform import from_origin

from snail.intersection import read_raster_values_for_splits


class CountingDataset:
    def __init__(self, dataset):
        self.dataset = dataset
        self.calls = 0
        self.bytes = 0
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

    def read(self, *args, **kwargs):
        result = self.dataset.read(*args, **kwargs)
        self.calls += 1
        self.bytes += result.nbytes
        return result


def workloads(size):
    dense_j, dense_i = np.indices((512, 512))
    yield "dense", dense_i.ravel(), dense_j.ravel()
    cells = np.arange(size)
    yield "linear", cells, cells
    rng = np.random.default_rng(20260909)
    yield "clustered", rng.integers(0, 512, 100_000), rng.integers(0, 512, 100_000)
    yield "sparse", rng.integers(0, size, 10_000), rng.integers(0, size, 10_000)


def main():
    size = 4096
    with MemoryFile() as memory:
        with memory.open(
            driver="GTiff",
            width=size,
            height=size,
            count=2,
            dtype="uint8",
            transform=from_origin(0, size, 1, 1),
            tiled=True,
            blockxsize=256,
            blockysize=256,
        ) as writable:
            empty = np.zeros((size, size), dtype="uint8")
            writable.write(empty, 1)
            writable.write(empty, 2)
        with memory.open() as source:
            for name, ii, jj in workloads(size):
                dataset = CountingDataset(source)
                splits = pd.DataFrame({"index_i": ii, "index_j": jj})
                started = time.perf_counter()
                read_raster_values_for_splits(
                    splits, dataset, [1, 2], max_raster_memory_mb=16
                )
                elapsed = time.perf_counter() - started
                print(
                    f"{name:<10} rows={len(splits):>8,} reads={dataset.calls:>5,} "
                    f"buffers={dataset.bytes / 2**20:>8.1f} MiB time={elapsed:>7.3f}s"
                )


if __name__ == "__main__":
    main()
