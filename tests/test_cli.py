import os
import tempfile
import unittest

from affine import Affine
import numpy
from numpy.testing import assert_array_equal
import geopandas as gpd
import rasterio
from shapely.geometry import LineString

import snail.cli


def make_raster_data(filename):
    data = numpy.random.randn(2, 2)
    new_dataset = rasterio.open(
        filename,
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


def make_vector_data(filename):
    test_linestrings = [
        LineString([(0.5, 0.5), (0.75, 0.5), (1.5, 0.5), (1.5, 1.5)]),
        LineString([(0.5, 0.5), (0.75, 0.5), (1.5, 1.5)]),
    ]
    gdf = gpd.GeoDataFrame({"col1": ["name1", "name2"], "geometry": test_linestrings})
    gdf.to_file(filename)


def get_expected_gdf():
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


class TestCli(unittest.TestCase):
    def test_cli(self):
        tmp_dir = tempfile.TemporaryDirectory()
        raster_file = os.path.join(tmp_dir.name, "test_raster.tif")
        vector_file = os.path.join(tmp_dir.name, "test_vector.gpkg")
        output_file = os.path.join(tmp_dir.name, "test_output.gpkg")

        make_raster_data(raster_file)
        make_vector_data(vector_file)

        args = [
            "--raster",
            raster_file,
            "--vector",
            vector_file,
            "--output",
            output_file,
        ]

        snail.cli.main(args)

        gdf = gpd.read_file(output_file)
        expected_gdf = get_expected_gdf()

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

        tmp_dir.cleanup()
