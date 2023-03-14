import logging
import os
from dataclasses import dataclass
from typing import Tuple

import geopandas
import numpy
import pandas
from shapely.geometry import mapping, shape
from shapely.ops import polygonize

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


@dataclass
class Transform:
    """Store a raster transform and CRS"""

    crs: str
    width: int
    height: int
    transform: Tuple[float]


# Use some high degree of precision to round polygon coordinates
# when polygonizing split edges to help avoid floating point errors
POLYGON_COORDINATE_PRECISION = 9


def split_linestrings(df, t: Transform):
    core_splits = []
    for i in tqdm(range(len(df))):
        # split edge
        splits = split_linestring(
            df.geometry[i], t.width, t.height, t.transform
        )
        for j, s in enumerate(splits):
            # splitting sometimes returns zero-length linestrings on edge of raster
            # see below for example linestring on eastern (lon=70W) extent of box
            # (Pdb) geometry.coords.xy
            # (array('d', [-70.0, -70.0]), array('d', [18.445832920952196, 18.445832920952196]))
            # this split geometry has: j = raster_width
            # however j should be in range: 0 <= j < raster_width
            # as a hacky workaround, drop any splits with length 0
            # do we need a nudge off a cell boundary somewhere when performing the splits?
            if not s.length == 0:
                new_row = df.iloc[i].copy()
                new_row.geometry = s
                new_row["split"] = j
                core_splits.append(new_row)
    logging.info(f"Split {len(df)} edges into {len(core_splits)} pieces")
    sdf = geopandas.GeoDataFrame(core_splits, crs=t.crs, geometry="geometry")
    return sdf


def split_polygons(df, t: Transform):
    core_splits = []
    for area in df.itertuples():
        # split area
        splits = split_polygon(area.geometry, t.width, t.height, t.transform)
        # round to high precision (avoid floating point errors)
        splits = [
            set_precision(s, POLYGON_COORDINATE_PRECISION) for s in splits
        ]
        # to polygons
        splits = list(polygonize(splits))
        # add to collection
        for s in splits:
            s_dict = area._asdict()
            del s_dict["Index"]
            s_dict["geometry"] = s
            core_splits.append(s_dict)
    logging.info(f"  Split {len(df)} areas into {len(core_splits)} pieces")
    sdf = geopandas.GeoDataFrame(core_splits)
    sdf.crs = t.crs
    return sdf


def set_precision(geom, precision):
    """Set geometry precision"""
    geom_mapping = mapping(geom)
    geom_mapping["coordinates"] = numpy.round(
        numpy.array(geom_mapping["coordinates"]), precision
    )
    return shape(geom_mapping)


def associate_raster(
    df: pandas.DataFrame,
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
    return pandas.Series(index=df.index, data=data[df[index_j], df[index_i]])


def apply_indices(
    df, transform: Transform, index_i="index_i", index_j="index_j"
):
    def f(geom, *args, **kwargs):
        return get_indices(geom, transform, index_i, index_j)

    indices = df.geometry.apply(f, result_type="expand")
    return pandas.concat([df, indices], axis="columns")


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
