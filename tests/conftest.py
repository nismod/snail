import geopandas as gpd
import numpy as np
import pytest
import rasterio
from rasterio.crs import CRS
from rasterio.transform import Affine
from shapely.geometry import LineString


@pytest.fixture
def band_1_data():
    # value of cell (i, j) is (j * 10 + i) for easy checking
    j, i = np.indices((4, 4))
    return (j * 10 + i).astype("float64")


@pytest.fixture
def two_band_raster(tmp_path, band_1_data):
    """4x4 raster in EPSG:4326, cell size 1, top-left at (0, 4), two bands.

    Band 1 cell (i, j) has value (j * 10 + i), band 2 has band 1 value + 100.
    """
    path = tmp_path / "two_band.tif"
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        width=4,
        height=4,
        count=2,
        dtype="float64",
        crs=CRS.from_epsg(4326),
        transform=Affine(1, 0, 0, 0, -1, 4),
    ) as dataset:
        dataset.write(band_1_data, 1)
        dataset.write(band_1_data + 100, 2)
    return path


@pytest.fixture
def half_cell_raster(tmp_path, band_1_data):
    """8x8 raster over the same extent as two_band_raster, cell size 0.5,
    single band, cell (i, j) has value (j * 100 + i)"""
    path = tmp_path / "half_cell.tif"
    j, i = np.indices((8, 8))
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        width=8,
        height=8,
        count=1,
        dtype="float64",
        crs=CRS.from_epsg(4326),
        transform=Affine(0.5, 0, 0, 0, -0.5, 4),
    ) as dataset:
        dataset.write((j * 100 + i).astype("float64"), 1)
    return path


@pytest.fixture
def lines_over_raster():
    """Two lines over the two_band_raster grid, in EPSG:4326

    The first line crosses from cell (0, 0) to cell (1, 0), the second sits
    within cell (2, 2).
    """
    return gpd.GeoDataFrame(
        {
            "name": ["a", "b"],
            "geometry": [
                LineString([(0.5, 3.5), (1.5, 3.5)]),
                LineString([(2.25, 1.5), (2.75, 1.5)]),
            ],
        },
        crs="EPSG:4326",
    )
