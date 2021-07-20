import unittest

import rasterio
from affine import Affine
import numpy
from shapely.geometry import LineString
import geopandas as gpd

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
    linestring = LineString([(0.5, 0.5), (0.75, 0.5), (1.5, 1.5)])
    gdf = gpd.GeoDataFrame({"col1": ["name1"], "geometry": [linestring]})
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
        self.assertTrue(True)
