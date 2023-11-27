import os

import geopandas as gpd
import numpy as np
import pytest
from hilbertcurve.hilbertcurve import HilbertCurve
from numpy.testing import assert_array_equal
from rasterio.crs import CRS
from shapely.geometry import LineString, Polygon
from shapely.geometry.polygon import LinearRing, orient

from snail.intersection import (
    GridDefinition,
    split_linestrings,
    split_polygons,
    generate_grid_boxes,
)


@pytest.fixture
def linestrings():
    test_linestrings = [
        LineString([(0.5, 0.5), (0.75, 0.5), (1.5, 0.5), (1.5, 1.5)]),
        LineString([(0.5, 0.5), (0.75, 0.5), (1.5, 1.5)]),
    ]
    gdf = gpd.GeoDataFrame(
        {"col1": ["name1", "name2"], "geometry": test_linestrings}
    )
    return gdf


@pytest.fixture
def linestrings_split():
    expected_splits = [
        LineString([(0.5, 0.5), (0.75, 0.5), (1.0, 0.5)]),
        LineString([(1.0, 0.5), (1.5, 0.5), (1.5, 1.0)]),
        LineString([(1.5, 1.0), (1.5, 1.5)]),
    ] + [
        LineString([(0.5, 0.5), (0.75, 0.5), (1.0, 0.8333)]),
        LineString([(1.0, 0.8333), (1.125, 1.0)]),
        LineString([(1.125, 1.0), (1.5, 1.5)]),
    ]
    expected_gdf = gpd.GeoDataFrame(
        {"col1": ["name1"] * 3 + ["name2"] * 3, "geometry": expected_splits},
        index=[0] * 3 + [1] * 3,
    )
    return expected_gdf


@pytest.fixture
def polygon():
    test_linearing = LinearRing(
        [
            (1.5, 0.25),
            (2.5, 1.5),
            (2.5, 3.5),
            (1.5, 2.25),
            (0.5, 3.5),
            (0.5, 1.5),
        ]
    )
    counter_clockwise = 1
    test_polygon = orient(Polygon(test_linearing), counter_clockwise)
    return gpd.GeoDataFrame({"col1": ["name1"], "geometry": [test_polygon]})


@pytest.fixture
def polygon_split():
    rings = [
        [
            (0.9, 1.0),
            (0.5, 1.5),
            (0.5, 2.0),
            (1.0, 2.0),
            (1.0, 1.0),
            (0.9, 1.0),
        ],
        [
            (0.9, 3.0),
            (1.0, 2.875),
            (1.0, 2.0),
            (0.5, 2.0),
            (0.5, 3.0),
            (0.9, 3.0),
        ],
        [(1.0, 0.875), (0.9, 1.0), (1.0, 1.0), (1.0, 0.875)],
        [(0.5, 3.0), (0.5, 3.5), (0.9, 3.0), (0.5, 3.0)],
        [
            (2.0, 0.875),
            (1.5, 0.25),
            (1.0, 0.875),
            (1.0, 1.0),
            (2.0, 1.0),
            (2.0, 0.875),
        ],
        [
            (1.0, 2.875),
            (1.5, 2.25),
            (2.0, 2.875),
            (2.0, 2.0),
            (1.0, 2.0),
            (1.0, 2.875),
        ],
        [(2.0, 1.0), (1.0, 1.0), (1.0, 2.0), (2.0, 2.0), (2.0, 1.0)],
        [(2.1, 1.0), (2.0, 0.875), (2.0, 1.0), (2.1, 1.0)],
        [
            (2.5, 2.0),
            (2.5, 1.5),
            (2.1, 1.0),
            (2.0, 1.0),
            (2.0, 2.0),
            (2.5, 2.0),
        ],
        [
            (2.5, 3.0),
            (2.5, 2.0),
            (2.0, 2.0),
            (2.0, 2.875),
            (2.1, 3.0),
            (2.5, 3.0),
        ],
        [(2.1, 3.0), (2.5, 3.5), (2.5, 3.0), (2.1, 3.0)],
    ]
    expected_polygons = [Polygon(ring) for ring in rings]
    expected_idx = ["name1"] * len(rings)
    expected_gdf = gpd.GeoDataFrame(
        {"col1": expected_idx, "geometry": expected_polygons}
    )
    expected_gdf["index"] = 0
    return expected_gdf.set_index("index")


@pytest.fixture
def grid():
    return GridDefinition(
        crs=None, width=4, height=4, transform=(1, 0, 0, 0, 1, 0)
    )


def test_grid_from_extent(grid):
    actual = GridDefinition.from_extent(
        xmin=0, ymin=0, xmax=4, ymax=4, cell_width=1, cell_height=1, crs=None
    )
    assert actual == grid


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


class TestSnailIntersections:
    def test_split_linestrings(self, grid, linestrings, linestrings_split):
        actual = split_linestrings(linestrings, grid)
        expected_gdf = linestrings_split

        # Assertions

        # Ideally we'd like to use geopandas.assert_geodataframe_equal to
        # to compare both expected and actual geodfs, but this function offers
        # little control over tolerance. When using option "check_less_precise",
        # it uses GeoSeries.geom_equals_exact under the hood, which has an kwarg
        # "tolerance". But assert_geodataframe_equal does not recognise kwarg "tolerance".
        assert (
            actual["geometry"]
            .geom_equals_exact(expected_gdf["geometry"], tolerance=1e-3)
            .values.all()
        )
        assert_array_equal(actual["col1"].values, expected_gdf["col1"].values)

    def test_split_polygons(self, grid, polygon, polygon_split):
        actual = sort_polygons(split_polygons(polygon, grid))
        expected = sort_polygons(polygon_split)

        for i in range(len(actual)):
            actual_geom = actual.iloc[i, 1]
            expected_geom = expected.iloc[i, 1]
            assert actual_geom.equals(expected_geom)
        assert_array_equal(actual["col1"].values, expected["col1"].values)


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


def sort_polygons(df):
    iterations = 6  # all coords must be <= (2**p - 1) ; 2**6 - 1 == 63
    ndimensions = 2
    hilbert_curve = HilbertCurve(iterations, ndimensions)
    points = df.geometry.centroid
    coords = np.array(
        list(zip(points.x.values.tolist(), points.y.values.tolist()))
    )
    int_coords = (coords * 10).astype(int)
    distances = hilbert_curve.distances_from_points(int_coords)
    df["hilbert_distance"] = distances
    return df.sort_values(by="hilbert_distance").drop(
        columns="hilbert_distance"
    )
