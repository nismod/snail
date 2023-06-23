import geopandas as gpd
import pytest
from numpy.testing import assert_array_equal
from shapely.geometry import LineString, Polygon
from shapely.geometry.polygon import LinearRing, orient

from snail.intersection import (
    GridDefinition,
    split_linestrings,
    split_polygons,
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
        [(1.0, 0.875), (0.9, 1.0), (1.0, 1.0), (1.0, 0.875)],
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
        [(0.5, 3.0), (0.5, 3.5), (0.9, 3.0), (0.5, 3.0)],
        [
            (2.0, 0.875),
            (1.5, 0.25),
            (1.0, 0.875),
            (1.0, 1.0),
            (2.0, 1.0),
            (2.0, 0.875),
        ],
        [(2.0, 1.0), (1.0, 1.0), (1.0, 2.0), (2.0, 2.0), (2.0, 1.0)],
        [
            (1.0, 2.875),
            (1.5, 2.25),
            (2.0, 2.875),
            (2.0, 2.0),
            (1.0, 2.0),
            (1.0, 2.875),
        ],
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


class TestSnailIntersections:
    def test_split_linestrings(self, grid, linestrings, linestrings_split):
        actual = split_linestrings(linestrings, grid)
        expected_gdf = linestrings_split

        # Assertions

        # Ideally we'd like to use geopandas.assert_geodataframe_equal to
        # to compare both expected and actual geodfs, but this function offers
        # little control over tolerance. When using option "check_less_precise",
        # it used GeoSeries.geom_almost_equals under the hood, which has an kwarg
        # "decimal". But assert_geodataframe_equal does not recognise kwarg "decimal".
        assert (
            actual["geometry"]
            .geom_almost_equals(expected_gdf["geometry"], decimal=3)
            .values.all()
        )
        assert_array_equal(actual["col1"].values, expected_gdf["col1"].values)

    def test_split_polygons(self, grid, polygon, polygon_split):
        actual = split_polygons(polygon, grid)
        expected = polygon_split

        for i in range(len(actual)):
            actual_geom = actual.iloc[i, 1]
            expected_geom = expected.iloc[i, 1]
            assert actual_geom.equals(expected_geom)
        assert_array_equal(actual["col1"].values, expected["col1"].values)
