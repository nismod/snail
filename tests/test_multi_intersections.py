import os
import unittest

from affine import Affine
import numpy as np
from numpy.testing import assert_array_equal, assert_array_almost_equal
import geopandas as gpd
import rasterio
from shapely.geometry import LineString, Polygon
from shapely.geometry.polygon import LinearRing, orient

from snail.multi_intersections import (
    split_linestrings,
    split_polygons,
    raster2split,
)

from split_polygons_rings import expected_polygons_rings


def make_raster_data():
    data = np.random.randn(2, 2)
    memfile = rasterio.io.MemoryFile()
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


def write_raster_to_tempfile(fname, data):
    with rasterio.open(
        fname,
        "w",
        driver="GTiff",
        width=data.shape[1],
        height=data.shape[0],
        count=1,
        crs="+proj=latlong",
        transform=Affine.identity(),
        dtype=data.dtype,
    ) as dst:
        dst.write(data, 1)


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
        assert_array_equal(
            gdf["line index"].values, expected_gdf["line index"].values
        )


class TestRaster2Split(unittest.TestCase):
    def setUp(self):
        self.fname = "test_raster.tif"
        self.raster_data = np.random.randn(2, 2)
        self.width = self.raster_data.shape[1]
        self.height = self.raster_data.shape[0]
        self.transform = list(Affine.identity())
        write_raster_to_tempfile(fname=self.fname, data=self.raster_data)

    def tearDown(self):
        try:
            os.remove(self.fname)
        except FileNotFoundError:
            pass

    def test_raster2split(self):
        vector_data = get_split_linestrings()
        rasters = {"key": self.fname}
        output_gdf = raster2split(
            vector_data,
            rasters,
            width=self.width,
            height=self.height,
            transform=self.transform,
        )
        # Expected raster values are points (0,1), (1,0) and (1,1) of the grid
        # x == columns and y == rows so elements [0,0], [0,1] and [1,1] of
        # underlying data array
        data_array_indices = ([0, 0, 1], [0, 1, 1])
        expected_raster_values = np.tile(
            self.raster_data[data_array_indices], 2
        )
        assert_array_almost_equal(
            output_gdf["key"].values, expected_raster_values
        )
