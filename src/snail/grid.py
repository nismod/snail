import math
from dataclasses import dataclass
from typing import Tuple

import geopandas
import numpy
import rasterio
from shapely import box


@dataclass(frozen=True)
class GridDefinition:
    """Store a raster transform and CRS

    A note on `transform` - these six numbers define the transform from `i,j`
    cell index (column/row) coordinates in the rectangular grid to `x,y`
    geographic coordinates, in the coordinate reference system of the input and
    output files. They effectively form the first two rows of a 3x3 matrix:


    .. code-block:: text

        | x |   | a  b  c | | i |
        | y | = | d  e  f | | j |
        | 1 |   | 0  0  1 | | 1 |


    In cases without shear or rotation, `a` and `e` define scaling or grid cell
    size, while `c` and `f` define the offset or grid upper-left corner:

    .. code-block:: text

        | x_scale 0       x_offset |
        | 0       y_scale y_offset |
        | 0       0       1        |

    """

    crs: str
    width: int
    height: int
    transform: Tuple[float]

    @classmethod
    def from_rasterio_dataset(cls, dataset):
        """GridDefinition for a rasterio dataset"""
        crs = dataset.crs
        width = dataset.width
        height = dataset.height
        # trim transform to 6 - we expect the first two rows of 3x3 matrix
        transform = tuple(dataset.transform)[:6]
        return GridDefinition(crs, width, height, transform)

    @classmethod
    def from_raster(cls, fname):
        """GridDefinition for a raster file (readable by rasterio)"""
        with rasterio.open(fname) as dataset:
            grid = GridDefinition.from_rasterio_dataset(dataset)
        return grid

    @classmethod
    def from_extent(
        cls,
        xmin: float,
        ymin: float,
        xmax: float,
        ymax: float,
        cell_width: float,
        cell_height: float,
        crs,
    ):
        """GridDefinition for a given extent, cell size and CRS"""
        return GridDefinition(
            crs=crs,
            width=math.ceil((xmax - xmin) / cell_width),
            height=math.ceil((ymax - ymin) / cell_height),
            transform=(cell_width, 0.0, xmin, 0.0, cell_height, ymin),
        )


def idx_to_ij(idx: int, width: int, height: int) -> Tuple[int]:
    return numpy.unravel_index(idx, (height, width))


def ij_to_idx(ij: Tuple[int], width: int, height: int):
    return numpy.ravel_multi_index(ij, (height, width))


def generate_grid_boxes(grid: GridDefinition):
    """Generate all the box polygons for a grid"""
    a, b, c, d, e, f = grid.transform
    idx = numpy.arange(grid.width * grid.height)
    i, j = numpy.unravel_index(idx, (grid.width, grid.height))
    xmin = i * a + j * b + c
    ymax = i * d + j * e + f
    xmax = (i + 1) * a + (j + 1) * b + c
    ymin = (i + 1) * d + (j + 1) * e + f
    return geopandas.GeoDataFrame(
        data={}, geometry=box(xmin, ymin, xmax, ymax), crs=grid.crs
    )
