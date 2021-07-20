import unittest

import rasterio
from affine import Affine
import numpy
from shapely.geometry import LineString
import geopandas as gpd
from geopandas.testing import assert_geodataframe_equal
from numpy.testing import assert_array_almost_equal

import snail.cli


def make_raster_data():
    data = numpy.random.randn(2, 2)
    raster_data = "/tmp/test_raster_data.tif"
    new_dataset = rasterio.open(
        raster_data,
        "w",
        driver="GTiff",
        height=data.shape[0],
        width=data.shape[1],
        count=1,
        dtype=data.dtype,
        crs="+proj=latlong",
        transform=Affine.identity(),
    )
    new_dataset.write(data, 1)
    new_dataset.close()
    return raster_data


def make_vector_data():
    test_linestrings = [
        LineString([(0.5, 0.5), (0.75, 0.5), (1.5, 0.5), (1.5, 1.5)]),
        LineString([(0.5, 0.5), (0.75, 0.5), (1.5, 1.5)]),
    ]
    gdf = gpd.GeoDataFrame({"col1": ["name1", "name2"], "geometry": test_linestrings})
    vector_data = "/tmp/test_vector_data.gpkg"
    gdf.to_file(vector_data)
    return vector_data


class TestCli(unittest.TestCase):
    def test_cli(self):
        raster_data = make_raster_data()
        vector_data = make_vector_data()
        output_data = "/tmp/test_output.gpkg"
        args = [
            "--raster",
            raster_data,
            "--vector",
            vector_data,
            "--output",
            output_data,
        ]

        snail.cli.main(args)

        gdf = gpd.read_file(output_data)
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
        assert_array_almost_equal(
            gdf["line index"].values, expected_gdf["line index"].values
        )
