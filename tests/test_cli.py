import unittest

import rasterio
from affine import Affine
import numpy
from shapely.geometry import LineString


def make_raster_data():
    data = numpy.random.randn(2, 2)
    new_dataset = rasterio.open(
        '/tmp/new.tif',
        'w',
        driver='GTiff',
        height=data.shape[0],
        width=data.shape[1],
        count=1,
        dtype=data.dtype,
        crs='+proj=latlong',
        transform=Affine.identity()
    )
    new_dataset.write(data, 1)
    new_dataset.close()


def make_vector_data():
    pass

class TestCli(unittest.TestCase):
    def test_cli(self):
        raster_data = make_raster_data()
        vector_data = make_vector_data()
