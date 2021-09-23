import unittest

from affine import Affine
import numpy as np
from numpy.testing import assert_array_equal, assert_array_almost_equal
import geopandas as gpd
from rasterio.io import MemoryFile
from shapely.geometry import LineString, Polygon
from shapely.geometry.polygon import LinearRing, orient

from snail.multi_intersections import (
    split_linestrings,
    split_polygons,
    raster2split,
)


def make_raster_data():
    data = np.random.randn(2, 2)
    memfile = MemoryFile()
    new_dataset = memfile.open(
        driver="GTiff",
        width=data.shape[1],
        height=data.shape[0],
        count=1,
        crs="+proj=latlong",
        transform=Affine.identity(),
        dtype=data.dtype,
    )
    new_dataset.write(data, 1)
    return new_dataset


def get_couple_of_linestrings():
    test_linestrings = [
        LineString([(0.5, 0.5), (0.75, 0.5), (1.5, 0.5), (1.5, 1.5)]),
        LineString([(0.5, 0.5), (0.75, 0.5), (1.5, 1.5)]),
    ]
    gdf = gpd.GeoDataFrame(
        {"col1": ["name1", "name2"], "geometry": test_linestrings}
    )
    return gdf


def make_polygon_vector_data():
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


def get_expected_polygon_gdf():
    expected_polygons_rings = [
        [
            (2.0, 0.875),
            (1.5, 0.25),
            (1.0, 0.875),
            (1.0, 1.0),
            (2.0, 1.0),
            (2.0, 0.875),
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
        [
            (1.0, 2.875),
            (1.5, 2.25),
            (2.0, 2.875),
            (2.0, 2.0),
            (1.0, 2.0),
            (1.0, 2.875),
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
            (0.9, 1.0),
            (0.5, 1.5),
            (0.5, 2.0),
            (1.0, 2.0),
            (1.0, 1.0),
            (0.9, 1.0),
        ],
        [(1.0, 0.875), (0.9, 1.0), (1.0, 1.0), (1.0, 0.875)],
        [(2.0, 1.0), (1.0, 1.0), (1.0, 2.0), (2.0, 2.0), (2.0, 1.0)],
    ]
    expected_polygons = [Polygon(ring) for ring in expected_polygons_rings]
    expected_idx = [0] * len(expected_polygons_rings)
    expected_gdf = gpd.GeoDataFrame(
        {"line index": expected_idx, "geometry": expected_polygons}
    )
    return expected_gdf


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
    expected_idx = [0] * 3 + [1] * 3
    expected_gdf = gpd.GeoDataFrame(
        {"line index": expected_idx, "geometry": expected_splits}
    )
    return expected_gdf


class TestSnailIntersections(unittest.TestCase):
    def setUp(self):
        self.raster_dataset = make_raster_data()

    def tearDown(self):
        self.raster_dataset.close()

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
        assert_array_equal(
            gdf["line index"].values, expected_gdf["line index"].values
        )

    def test_split_polygons(self):
        vector_data = make_polygon_vector_data()
        gdf = split_polygons(vector_data, self.raster_dataset)
        expected_gdf = get_expected_polygon_gdf()

        self.assertTrue(
            list(
                gdf["geometry"]
                .geom_almost_equals(expected_gdf["geometry"], decimal=3)
                .values
            )
        )
        assert_array_equal(
            gdf["line index"].values, expected_gdf["line index"].values
        )

    def test_raster2split(self):
        vector_data = get_split_linestrings()

        output_gdf = raster2split(vector_data, self.raster_dataset, bands=[1])

        # Expected raster values are points (0,1), (1,0) and (1,1) of the grid
        data_array_indices = ([0, 1, 1], [0, 0, 1])
        raster_data = self.raster_dataset.read(1)
        expected_raster_values = np.tile(raster_data[data_array_indices], 2)
        assert_array_almost_equal(
            output_gdf["band1"].values, expected_raster_values
        )
