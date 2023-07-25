import logging
import math
import os
from dataclasses import dataclass
from functools import partial
from itertools import product
from multiprocessing import Pool, cpu_count
from typing import Callable, List, Tuple

import geopandas
import numpy
import pandas
import rasterio
from shapely.geometry import mapping, shape, box
from shapely.ops import linemerge, polygonize

from snail.core.intersections import (
    get_cell_indices,
    split_linestring,
    split_polygon,
)

# optional progress bars
if "SNAIL_PROGRESS" in os.environ and os.environ["SNAIL_PROGRESS"] in (
    "1",
    "TRUE",
):
    try:
        from tqdm import tqdm
    except ImportError:
        from snail.tqdm_standin import tqdm_standin as tqdm
else:
    from snail.tqdm_standin import tqdm_standin as tqdm

# Use some high degree of precision to round polygon coordinates
# when polygonizing split edges to help avoid floating point errors
POLYGON_COORDINATE_PRECISION = 9


@dataclass(frozen=True)
class GridDefinition:
    """Store a raster transform and CRS

    A note on `transform` - these six numbers define the transform from `i,j`
    cell index (column/row) coordinates in the rectangular grid to `x,y`
    geographic coordinates, in the coordinate reference system of the input and
    output files. They effectively form the first two rows of a 3x3 matrix:


    ```
    | x |   | a  b  c | | i |
    | y | = | d  e  f | | j |
    | 1 |   | 0  0  1 | | 1 |
    ```

    In cases without shear or rotation, `a` and `e` define scaling or grid cell
    size, while `c` and `f` define the offset or grid upper-left corner:

    ```
    | x_scale 0       x_offset |
    | 0       y_scale y_offset |
    | 0       0       1        |
    ```
    """

    crs: str
    width: int
    height: int
    transform: Tuple[float]

    @classmethod
    def from_rasterio_dataset(cls, dataset):
        crs = dataset.crs
        width = dataset.width
        height = dataset.height
        # trim transform to 6 - we expect the first two rows of 3x3 matrix
        transform = tuple(dataset.transform)[:6]
        return GridDefinition(crs, width, height, transform)

    @classmethod
    def from_raster(cls, fname):
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
        return GridDefinition(
            crs=crs,
            width=math.ceil((xmax - xmin) / cell_width),
            height=math.ceil((ymax - ymin) / cell_height),
            transform=(cell_width, 0, xmin, 0, cell_height, ymin),
        )


def split_features_for_rasters(
    features: geopandas.GeoDataFrame,
    grids: List[GridDefinition],
    split_func: Callable,
):
    # lookup per transform
    for i, t in enumerate(grids):
        logging.info("Splitting on grid %s %s", i, t)
        # transform to grid CRS
        crs_features = features.to_crs(t.crs)
        crs_features = split_func(crs_features, t)
        # save cell index for fast lookup of raster values
        crs_features = apply_indices(crs_features, t, f"i_{i}", f"j_{i}")
        # transform back
        features = crs_features.to_crs(features.crs)
    return features


def prepare_points(features: geopandas.GeoDataFrame) -> geopandas.GeoDataFrame:
    """Prepare points for splitting"""
    return features.explode(ignore_index=True)


def prepare_linestrings(
    features: geopandas.GeoDataFrame,
) -> geopandas.GeoDataFrame:
    features.geometry = features.geometry.apply(_try_merge)
    return features.explode(ignore_index=True)


def prepare_polygons(
    features: geopandas.GeoDataFrame,
) -> geopandas.GeoDataFrame:
    return features.explode(ignore_index=True)


def split_points(
    points: geopandas.GeoDataFrame, grid: GridDefinition
) -> geopandas.GeoDataFrame:
    """Split points along a grid

    This is a no-op, written for equivalence when processing multiple
    geometry types.
    """
    return points


def split_linestrings(
    linestring_features: geopandas.GeoDataFrame, grid: GridDefinition
) -> geopandas.GeoDataFrame:
    """Split linestrings along a grid"""
    # TODO check for MultiLineString
    # throw error or coerce (df.explode)
    pieces = []
    for i in tqdm(range(len(linestring_features))):
        # split edge
        geom_splits = split_linestring(
            linestring_features.geometry[i],
            grid.width,
            grid.height,
            grid.transform,
        )
        for j, s in enumerate(geom_splits):
            # splitting sometimes returns zero-length linestrings on edge of raster
            # see below for example linestring on eastern (lon=70W) extent of box
            # (Pdb) geometry.coords.xy
            # (array('d', [-70.0, -70.0]), array('d', [18.445832920952196, 18.445832920952196]))
            # this split geometry has: j = raster_width
            # however j should be in range: 0 <= j < raster_width
            # as a hacky workaround, drop any splits with length 0
            # do we need a nudge off a cell boundary somewhere when performing the splits?
            if not s.length == 0:
                new_row = linestring_features.iloc[i].copy()
                new_row.geometry = s
                new_row["split"] = j
                pieces.append(new_row)
    logging.info(
        f"Split {len(linestring_features)} edges into {len(pieces)} pieces"
    )
    splits_df = geopandas.GeoDataFrame(
        pieces, crs=grid.crs, geometry="geometry"
    )
    return splits_df


def _transform(i, j, a, b, c, d, e, f) -> Tuple[float]:
    return (i * a + j * b + c, i * d + j * e + f)


def split_polygons(
    polygon_features: geopandas.GeoDataFrame, grid: GridDefinition
) -> geopandas.GeoDataFrame:
    """Split polygons along a grid"""
    pieces = []
    ##
    # Fairly slow but solid approach, loop over cells and
    # use geopandas (shapely/GEOS) intersection
    ##
    num_processes = max(1, cpu_count() - 2)
    logging.info(f"Running with {num_processes=}")
    pool = Pool(processes=num_processes)
    partial_func = partial(
        _intersect_boxes,
        grid=grid,
        features=polygon_features,
        num_processes=num_processes,
    )
    pieces = pool.imap(
        func=partial_func,
        iterable=range(num_processes),
    )

    splits_df = pandas.concat(pieces)
    return splits_df


def _intersect_boxes(n, grid, features, num_processes):
    for k in tqdm(
        range(math.ceil((grid.width * grid.height) / num_processes))
    ):
        idx = k * num_processes + n
        ij = idx_to_ij(idx, grid.width, grid.height)
        _intersect_box(ij, grid, features)


def _intersect_box(ij, grid, features):
    i, j = ij
    a, b, c, d, e, f = grid.transform
    ulx, uly = _transform(i, j, a, b, c, d, e, f)
    lrx, lry = _transform(i + 1, j + 1, a, b, c, d, e, f)
    cell_geom = box(ulx, uly, lrx, lry)
    idx = features.geometry.sindex.query(cell_geom)
    subset = features.iloc[idx].copy()
    if len(subset):
        subset.geometry = subset.intersection(cell_geom)
        subset = subset[~(subset.geometry.is_empty | subset.geometry.isna())]
        subset = subset.explode(ignore_index=True)
        subset = subset[subset.geometry.type == "Polygon"]
    return subset


def split_polygons_experimental(
    polygon_features: geopandas.GeoDataFrame, grid: GridDefinition
) -> geopandas.GeoDataFrame:
    """Split polygons along a grid

    Experimental implementation of `split_polygons`, possibly fast/incorrect
    with some inputs.
    """
    pieces = []
    ##
    # Approach using snail::splitPolygon to produce a mesh of
    # half-line pieces within the polygon interior, plus the boundary
    # split into pieces, then passed to shapely (GEOS) polygonize
    # - doesn't handle polygons with holes
    # - polygonize doesn't always piece back together correctly, or leaves
    #   gaps - perhaps especially for coarse grids (vs shape size)
    # - should be possible to write all at the lower level
    ##
    polygon_features["split"] = 0
    for i in tqdm(range(len(polygon_features))):
        # split area
        geom_splits = split_polygon(
            polygon_features.geometry[i],
            grid.width,
            grid.height,
            grid.transform,
        )
        # round to high precision (avoid floating point errors)
        geom_splits = [
            _set_precision(s, POLYGON_COORDINATE_PRECISION)
            for s in geom_splits
        ]
        # to polygons
        geom_splits = list(polygonize(geom_splits))
        # add to collection
        for j, s in enumerate(geom_splits):
            new_row = polygon_features.iloc[i].copy()
            new_row.geometry = s
            new_row["split"] = j
            pieces.append(new_row)
    logging.info(
        f"  Split {len(polygon_features)} areas into {len(pieces)} pieces"
    )
    splits_df = geopandas.GeoDataFrame(pieces)
    splits_df.crs = grid.crs
    return splits_df


def _try_merge(geom):
    if geom.geom_type == "MultiLineString":
        geom = linemerge(geom)
    return geom


def _set_precision(geom, precision):
    """Set geometry coordinate precision"""
    geom_mapping = mapping(geom)
    geom_mapping["coordinates"] = numpy.round(
        numpy.array(geom_mapping["coordinates"]), precision
    )
    return shape(geom_mapping)


def get_raster_values_for_splits(
    splits: pandas.DataFrame,
    data: numpy.ndarray,
    index_i: str = "index_i",
    index_j: str = "index_j",
) -> pandas.Series:
    """For each split geometry, lookup the relevant raster value.

    Cell indices must have been previously calculated and stored as index_i and
    index_j.

    N.B. This will pass through no data values from the raster (no filtering).

    Parameters
    ----------
    splits: pandas.DataFrame
        Table of features, each with cell indices
        to look up raster pixel. Indices must be stored under columns with
        names referenced by index_i and index_j.
    data:  numpy.ndarray
        Raster data (2D array)
    index_i: str
        Column name for i-indices
    index_j: str
        Column name for j-indices

    Returns
    -------
    pd.Series
        Series of raster values, with same row indexing as df.
    """
    # 2D numpy indexing is j, i (i.e. row, column)
    with_data = pandas.Series(
        index=splits.index, data=data[splits[index_j], splits[index_i]]
    )
    # set NaN for out-of-bounds
    with_data[(splits[index_i] == -1) | (splits[index_j] == -1)] = numpy.nan
    return with_data


def apply_indices(
    features: geopandas.GeoDataFrame,
    grid: GridDefinition,
    index_i="index_i",
    index_j="index_j",
) -> geopandas.GeoDataFrame:
    def f(geom, *args, **kwargs):
        return get_indices(geom, grid, index_i, index_j)

    indices = features.geometry.apply(f, result_type="expand")
    return pandas.concat([features, indices], axis="columns")


def get_indices(
    geom, grid: GridDefinition, index_i="index_i", index_j="index_j"
) -> pandas.Series:
    """Given a geometry, find the cell index (i, j) of its midpoint
    for the enclosing grid.

    N.B. There is no checking whether a geometry spans more than one cell.
    """
    i, j = get_cell_indices(geom, grid.height, grid.width, grid.transform)

    # Raise error if cell index would be out of bounds
    # assert 0 <= i < t.width
    # assert 0 <= j < t.height

    # Or - special value (-1,-1) if cell would be out of bounds
    if i >= grid.width or i < 0 or j >= grid.height or j < 0:
        i = -1
        j = -1
    return pandas.Series(index=(index_i, index_j), data=[i, j])


def idx_to_ij(idx: int, width: int, height: int):
    return numpy.unravel_index(idx, (height, width))


def ij_to_idx(ij: tuple[int], width: int, height: int):
    return numpy.ravel_multi_index(ij, (height, width))
