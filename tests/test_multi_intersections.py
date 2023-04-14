import unittest

from numpy.testing import assert_array_equal
import geopandas as gpd
from shapely.geometry import LineString, Polygon
from shapely.geometry.polygon import LinearRing, orient

from snail.intersection import (
    Transform,
    split_linestrings,
    split_polygons,
)

from split_polygons_rings import expected_polygons_rings


def get_couple_of_linestrings():
    test_linestrings = [
        LineString([(0.5, 0.5), (0.75, 0.5), (1.5, 0.5), (1.5, 1.5)]),
        LineString([(0.5, 0.5), (0.75, 0.5), (1.5, 1.5)]),
    ]
    gdf = gpd.GeoDataFrame(
        {"col1": ["name1", "name2"], "geometry": test_linestrings}
    )
    return gdf


def get_polygon_vector_data():
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


def get_split_polygons():
    expected_polygons = [Polygon(ring) for ring in expected_polygons_rings]
    expected_idx = ["name1"] * len(expected_polygons_rings)
    expected_gdf = gpd.GeoDataFrame(
        {"col1": expected_idx, "geometry": expected_polygons}
    )
    expected_gdf["index"] = 0
    return expected_gdf.set_index("index")


def get_split_linestrings():
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


class TestSnailIntersections(unittest.TestCase):
    def setUp(self):
        self.raster_dataset = Transform(
            crs=None, width=4, height=4, transform=(1, 0, 0, 0, 1, 0)
        )

    def test_split_linestrings(self):
        vector_data = get_couple_of_linestrings()
        gdf = split_linestrings(vector_data, self.raster_dataset)
        expected_gdf = get_split_linestrings()

        # Assertions

        # Ideally we'd like to use geopandas.assert_geodataframe_equal to
        # to compare both expected and actual geodfs, but this function offers
        # little control over tolerance. When using option "check_less_precise",
        # it used GeoSeries.geom_almost_equals under the hood, which has an kwarg
        # "decimal". But assert_geodataframe_equal does not recognise kwarg "decimal".
        self.assertTrue(
            list(
                gdf["geometry"]
                .geom_almost_equals(expected_gdf["geometry"], decimal=3)
                .values
            )
        )
        assert_array_equal(gdf["col1"].values, expected_gdf["col1"].values)

    def test_split_polygons(self):
        vector_data = get_polygon_vector_data()
        gdf = split_polygons(vector_data, self.raster_dataset)
        expected_gdf = get_split_polygons()
        self.assertTrue(
            list(
                gdf["geometry"]
                .geom_almost_equals(expected_gdf["geometry"], decimal=3)
                .values
            )
        )
        assert_array_equal(gdf["col1"].values, expected_gdf["col1"].values)
