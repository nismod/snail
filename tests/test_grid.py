import os

from rasterio.crs import CRS

from snail.grid import GridDefinition, generate_grid_boxes


def test_grid_from_extent():
    expected = GridDefinition(
        crs=None, width=4, height=5, transform=(1, 0, 10, 0, 2, 20)
    )
    actual = GridDefinition.from_extent(
        xmin=10,
        ymin=20,
        xmax=14,
        ymax=30,
        cell_width=1,
        cell_height=2,
        crs=None,
    )
    assert actual == expected


def test_grid_from_raster():
    fname = os.path.join(
        os.path.dirname(__file__),
        "integration",
        "range.tif",
    )
    actual = GridDefinition.from_raster(fname)
    expected = GridDefinition(
        crs=CRS.from_epsg(4326),
        width=23,
        height=14,
        transform=(
            0.008333333347826087,
            0.0,
            -1.341666667,
            0.0,
            -0.008333333285714315,
            51.808333333,
        ),
    )
    assert actual == expected


def test_box_geom_bounds():
    """Values take from tests/integration/range.tif"""
    grid = GridDefinition(
        crs=CRS.from_epsg(4326),
        width=23,
        height=14,
        transform=(
            0.008333333347826087,
            0.0,
            -1.341666667,
            0.0,
            -0.008333333285714315,
            51.808333333,
        ),
    )
    box_geoms = generate_grid_boxes(grid)
    minb = box_geoms.bounds.min()
    maxb = box_geoms.bounds.max()

    atol = 1e-4
    assert abs(minb.minx - -1.3416667) < atol
    assert abs(minb.miny - 51.6916667) < atol
    assert abs(maxb.maxx - -1.1500000) < atol
    assert abs(maxb.maxy - 51.8083333) < atol
