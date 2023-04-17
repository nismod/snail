import logging
import os
from dataclasses import dataclass
from itertools import product
from typing import Callable, List, Tuple

import geopandas
import numpy
import pandas
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


@dataclass
class Transform:
    """Store a raster transform and CRS"""

    crs: str
    width: int
    height: int
    transform: Tuple[float]


def split_features_for_rasters(
    features: geopandas.GeoDataFrame,
    transforms: List[Transform],
    split_func: Callable,
):
    # lookup per transform
    for i, t in enumerate(transforms):
        logging.info("Splitting on transform %s %s", i, t)
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
    points: geopandas.GeoDataFrame, t: Transform
) -> geopandas.GeoDataFrame:
    """Split points along the grid defined by a transform

    This is a no-op, written for equivalence when processing multiple
    geometry types.
    """
    return points


def split_linestrings(
    linestring_features: geopandas.GeoDataFrame, t: Transform
) -> geopandas.GeoDataFrame:
    """Split linestrings along the grid defined by a transform"""
    pieces = []
    for i in tqdm(range(len(linestring_features))):
        # split edge
        geom_splits = split_linestring(
            linestring_features.geometry[i], t.width, t.height, t.transform
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
    splits_df = geopandas.GeoDataFrame(pieces, crs=t.crs, geometry="geometry")
    return splits_df


def _transform(i, j, a, b, c, d, e, f) -> Tuple[float]:
    return (i * a + j * b + c, i * d + j * e + f)


def split_polygons(
    polygon_features: geopandas.GeoDataFrame, t: Transform
) -> geopandas.GeoDataFrame:
    """Split polygons along the grid defined by a transform"""
    pieces = []
    ##
    # Fairly slow but solid approach, loop over cells and
    # use geopandas (shapely/GEOS) intersection
    ##
    a, b, c, d, e, f = t.transform
    for i, j in tqdm(
        product(range(t.width), range(t.height)), total=t.width * t.height
    ):
        ulx, uly = _transform(i, j, a, b, c, d, e, f)
        lrx, lry = _transform(i + 1, j + 1, a, b, c, d, e, f)
        cell_geom = box(ulx, uly, lrx, lry)
        idx = polygon_features.geometry.sindex.query(cell_geom)
        subset = polygon_features.iloc[idx].copy()
        if len(subset):
            subset.geometry = subset.intersection(cell_geom)
            subset = subset[
                ~(subset.geometry.is_empty | subset.geometry.isna())
            ]
            subset = subset.explode(ignore_index=True)
            subset = subset[subset.geometry.type == "Polygon"]
            pieces.append(subset)
    splits_df = pandas.concat(pieces)
    return splits_df


def split_polygons_experimental(
    polygon_features: geopandas.GeoDataFrame, t: Transform
) -> geopandas.GeoDataFrame:
    """Split polygons along the grid defined by a transform

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
    for i in tqdm(range(len(polygon_features))):
        # split area
        geom_splits = split_polygon(
            polygon_features.geometry[i], t.width, t.height, t.transform
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
    splits_df.crs = t.crs
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


def associate_raster(
    features: pandas.DataFrame,
    data: numpy.ndarray,
    index_i: str = "index_i",
    index_j: str = "index_j",
) -> pandas.Series:
    """For each split geometry, lookup the relevant raster value.

    Cell indices must have been previously calculated and stored as index_i and
    index_j.

    N.B. This will pass through no data values from the raster (no filtering).

    Args:
        df: Table of features, each with cell indices
            to look up raster pixel. Indices must be stored under columns with
            names referenced by index_i and index_j.
        fname: Filename of raster file to read data from
    Returns:
        pd.Series: Series of raster values, with same row indexing as df.
    """
    # 2D numpy indexing is j, i (i.e. row, column)
    return pandas.Series(
        index=features.index, data=data[features[index_j], features[index_i]]
    )


def apply_indices(
    features: geopandas.GeoDataFrame,
    transform: Transform,
    index_i="index_i",
    index_j="index_j",
) -> geopandas.GeoDataFrame:
    def f(geom, *args, **kwargs):
        return get_indices(geom, transform, index_i, index_j)

    indices = features.geometry.apply(f, result_type="expand")
    return pandas.concat([features, indices], axis="columns")


def get_indices(
    geom, t: Transform, index_i="index_i", index_j="index_j"
) -> pandas.Series:
    """Given a geometry, find the cell index (i, j) of its midpoint
    for the enclosing raster transform.

    N.B. There is no checking whether a geometry spans more than one cell.
    """
    i, j = get_cell_indices(geom, t.height, t.width, t.transform)

    # Raise error if cell index would be out of bounds
    # assert 0 <= i < t.width
    # assert 0 <= j < t.height

    # Or - special value (-1,-1) if cell would be out of bounds
    if i > t.width or i < 0 or j > t.height or j < 0:
        i = -1
        j = -1
    return pandas.Series(index=(index_i, index_j), data=[i, j])
