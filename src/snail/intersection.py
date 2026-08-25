import logging
import math
import os
from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass
from itertools import pairwise
from typing import Union

import dask.array
import geopandas
import numpy
import pandas
import pyarrow
import rasterio
import xarray
from shapely.ops import linemerge

from snail.core.intersections import (  # type: ignore
    get_cell_indices,
)
from snail.core.intersections import (
    split_geometries as split_geometries_core,
)
from snail.core.intersections import (
    split_linestrings as split_linestrings_core,
)
from snail.core.intersections import (
    split_polygons as split_polygons_core,
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


#: Batch size for splitting an in-memory geometry column.
#:
#: The extension splits one Arrow batch at a time, so an in-memory geometry
#: column is presented to it in batches of this many features: splitting a
#: whole table at once would give no sign of progress on a long job, and
#: would hold every piece of every feature in memory at once. A source that
#: brings its own batching - a GeoParquet reader, a Dataset scan - can be
#: split directly, and keeps whatever batch size it was read with; this only
#: governs the batches :func:`to_geoarrow` slices an in-memory column into.
#:
#: Each batch costs something fixed to set up and read back, so batches much
#: smaller than this measurably slow a large split down; much larger ones buy
#: no more speed and only raise the peak memory.
SPLIT_BATCH_SIZE = 5000


def to_geoarrow(
    geometries: geopandas.GeoSeries,
    batch_size: int = SPLIT_BATCH_SIZE,
    encoding: str = "geoarrow",
):
    """Geometry column as a stream of GeoArrow batches, for the extension

    GeoArrow holds the geometries as flat coordinate and offset buffers,
    which the extension reads directly - no geometry object is built per
    feature on either side of the interface. Batches are zero-copy slices
    of the one Arrow array, and the geometry type travels with them.

    This is what :func:`split_linestrings` and :func:`split_polygons`
    use to feed a geometry column to :func:`snail.core.intersections.split_linestrings`
    or :func:`snail.core.intersections.split_polygons`; call it directly
    only if you are working with those lower-level, Arrow-native functions
    yourself.

    Parameters
    ----------
    geometries: geopandas.GeoSeries
        Column of LineString or Polygon geometries to split.
    batch_size: int
        Number of features per batch. See :data:`SPLIT_BATCH_SIZE`.
    encoding: str
        ``"geoarrow"`` (the default) holds the coordinates in Arrow buffers,
        which the extension reads in place. It requires every geometry in the
        column to be the same, single-part type - geopandas raises
        ``ValueError: Geometry type combination is not supported`` otherwise.
        ``"WKB"`` serialises each geometry to a blob instead, which is slower
        to read but can carry a column of mixed or multi-part geometries;
        that is what :func:`split_geometries` uses.

    Returns
    -------
    pyarrow.Table
        A single-column ``"geometry"`` table, chunked into batches of
        ``batch_size`` features, implementing the Arrow PyCapsule stream
        interface (``__arrow_c_stream__``) that the extension consumes.
    """
    # Import the geometry column through its capsules rather than with
    # pyarrow.array, which would keep the values but drop the GeoArrow
    # extension name: that lives on the Arrow *field*, not on its type, and
    # is what tells the extension - and any later reader - which geometry
    # type these are.
    exported = (
        geometries.to_arrow(geometry_encoding="geoarrow", interleaved=True)
        if encoding == "geoarrow"
        else geometries.to_arrow(geometry_encoding=encoding)
    )
    schema_capsule, array_capsule = exported.__arrow_c_array__()
    field = pyarrow.Field._import_from_c_capsule(schema_capsule).with_name("geometry")
    array = pyarrow.Array._import_from_c_capsule(
        field.__arrow_c_schema__(), array_capsule
    )
    # A table of one column, chunked: only a table carries the field, and so
    # the extension name, through __arrow_c_stream__ to the extension.
    if batch_size <= 0:
        raise ValueError("batch_size must be greater than zero")
    batches = [array.slice(at, batch_size) for at in range(0, len(array), batch_size)]
    return pyarrow.Table.from_arrays(
        [pyarrow.chunked_array(batches, type=array.type)],
        schema=pyarrow.schema([field]),
    )


def read_split_stream(stream) -> tuple[numpy.ndarray, numpy.ndarray]:
    """Read a stream of split pieces from the extension into memory

    The split runs as the stream is read, a batch at a time, but this
    drains the stream fully and concatenates every batch: use it when you
    want the pieces as shapely geometries and are content to hold them all
    at once, which is what :func:`split_linestrings` and
    :func:`split_polygons` do. To keep the streaming memory
    benefit - splitting a source larger than memory, for example - iterate
    the stream yourself instead, e.g. with
        ``pyarrow.RecordBatchReader.from_stream(stream)``,
    and consume each record batch as it arrives.

    Parameters
    ----------
    stream
        A :class:`snail.core.intersections.SplitStream`, or any object
        implementing the Arrow PyCapsule stream interface
        (``__arrow_c_stream__``) with a ``"geometry"`` and a ``"parent"``
        column, such as the result of
        :func:`snail.core.intersections.split_linestrings` or
        :func:`snail.core.intersections.split_polygons`.

    Returns
    -------
    geometry: numpy.ndarray
        The split pieces, as shapely geometries.
    parent: numpy.ndarray
        For each piece, the index of the geometry it was split from.
    """
    reader = pyarrow.RecordBatchReader.from_stream(stream)
    geometry = []
    parent = []
    for batch in tqdm(reader):
        # read the batch as a frame rather than picking the geometry column
        # out first: the GeoArrow extension name is on the schema field, and
        # a column taken off a batch no longer carries it
        geometry.append(geopandas.GeoDataFrame.from_arrow(batch).geometry.to_numpy())
        parent.append(batch.column("parent").to_numpy())
    if not geometry:
        return numpy.empty(0, dtype=object), numpy.empty(0, dtype=numpy.int64)
    return numpy.concatenate(geometry), numpy.concatenate(parent)


def _splits_frame(features, geometry, parent, grid):
    """Assemble split pieces back into a frame beside their parent features

    Each piece carries the attributes of the feature it came from, and a
    "split" column numbering that feature's pieces from zero.
    """
    # repeat each parent feature's attributes for each of its pieces
    splits_df = geopandas.GeoDataFrame(features.iloc[parent]).set_crs(
        grid.crs, allow_override=True
    )
    # number each parent's pieces from zero
    piece_counts = numpy.bincount(parent, minlength=len(features))
    splits_df["split"] = numpy.arange(len(parent)) - numpy.repeat(
        numpy.cumsum(piece_counts) - piece_counts, piece_counts
    )
    splits_df.geometry = geometry
    return splits_df


def _split(split_core, geometries, grid, encoding="geoarrow", **split_kwargs):
    """Split a geometry column, streaming batches through the extension"""
    if len(geometries) == 0:
        return numpy.empty(0, dtype=object), numpy.empty(0, dtype=numpy.int64)
    # The extension counts rows and columns: nrows spans the grid in y and
    # ncols in x, so they are the grid's height and width respectively.
    # Passed by name - transposing them silently moves the grid bounds,
    # which only shows up on a grid that is not square.
    return read_split_stream(
        split_core(
            to_geoarrow(geometries, encoding=encoding),
            nrows=grid.height,
            ncols=grid.width,
            transform=grid.transform,
            **split_kwargs,
        )
    )


# Module-level logger
logger = logging.getLogger(__name__)


def _is_dask_array(value):
    return isinstance(value, dask.array.Array)


def _is_xarray_dataarray(value):
    return isinstance(value, xarray.DataArray)


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
    transform: tuple[float]

    @classmethod
    def from_raster(cls, fname):
        """GridDefinition for a raster file (readable by rasterio)"""
        with rasterio.open(fname) as dataset:
            driver = dataset.driver

        if driver in ("netCDF", "Zarr"):
            if driver == "netCDF":
                engine = "netcdf4"
            elif driver == "Zarr":
                engine = "zarr"
            else:
                raise OSError("Unrecognised driver, expected netCDF or Zarr")

            with xarray.open_dataset(
                fname, chunks="auto", decode_coords="all", engine=engine
            ) as dataset:
                grid = GridDefinition.from_xarray(dataset)
        else:
            with rasterio.open(fname) as dataset:
                grid = GridDefinition.from_rasterio(dataset)

        return grid

    @classmethod
    def from_rasterio(cls, dataset):
        """GridDefinition for a rasterio dataset"""
        crs = dataset.crs
        width = dataset.width
        height = dataset.height
        # trim transform to 6 - we expect the first two rows of 3x3 matrix
        transform = tuple(dataset.transform)[:6]
        return GridDefinition(crs, width, height, transform)

    @classmethod
    def from_xarray(cls, data_array: xarray.Dataset | xarray.DataArray):
        """GridDefinition for an xarray DataArray or Dataset with spatial metadata.

        Requires the DataArray to have a rioxarray accessor with CRS and
        transform information derived from explicit coordinates.
        """
        crs = data_array.rio.crs

        if crs is None:
            raise ValueError(
                "data_array.rio.crs is None; assign a CRS using DataArray.rio.write_crs"
            )

        transform = tuple(data_array.rio.transform())[:6]
        width = int(data_array.rio.width)
        height = int(data_array.rio.height)

        return GridDefinition(crs=crs, width=width, height=height, transform=transform)

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


def split_features_for_rasters(
    features: geopandas.GeoDataFrame,
    grids: list[GridDefinition],
    split_func: Callable,
):
    """Split features on a list of grids, attaching cell indices

    Features are implicitly reprojected to each grid CRS for splitting and
    indexing (columns "i_{n}", "j_{n}" refer to cells of the nth grid), then
    returned in their original CRS. If either the features or a grid have no
    CRS defined, they are assumed to share the same CRS and no reprojection
    happens.
    """
    # lookup per transform
    for i, grid in enumerate(grids):
        logger.info("Splitting on grid %s %s", i, grid)
        # transform to grid CRS
        if features.crs is None or grid.crs is None:
            if features.crs is None and grid.crs is None:
                logger.warning(
                    "CRS undefined for features (%s) and grid (%s): assuming they share a CRS",
                    features.crs,
                    grid.crs,
                )
                source_crs = None
                crs_features = features
            elif features.crs is None:
                logger.warning(
                    "CRS undefined for features (%s): assuming they share the grid CRS (%s)",
                    features.crs,
                    grid.crs,
                )
                source_crs = grid.crs
                crs_features = features.set_crs(grid.crs)
            else:  # grid.crs is None
                logger.warning(
                    "CRS undefined for grid (%s): assuming grid shares the features CRS (%s)",
                    grid.crs,
                    features.crs,
                )
                source_crs = features.crs
                crs_features = features
                grid.crs = features.crs
        else:
            source_crs = features.crs
            crs_features = features.to_crs(grid.crs)
        crs_features = split_func(crs_features, grid)
        # save cell index for fast lookup of raster values
        crs_features = apply_indices(crs_features, grid, f"i_{i}", f"j_{i}")
        # transform back
        if source_crs is not None:
            features = crs_features.to_crs(source_crs)
        else:
            features = crs_features
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
    points: geopandas.GeoDataFrame, _: GridDefinition
) -> geopandas.GeoDataFrame:
    """Split points along a grid

    This is a no-op, written for equivalence when processing multiple
    geometry types.
    """
    return points


def split_linestrings(
    linestring_features: geopandas.GeoDataFrame,
    grid: GridDefinition,
    bounded=False,
) -> geopandas.GeoDataFrame:
    """Split linestrings along a grid

    Each piece lies within a single grid cell; together the pieces of a
    feature are that feature, cut up, so the split conserves its length.

    Any MultiLineString geometries are coerced to LineStrings (merged where
    contiguous, then exploded to one row per part, resetting the index) with
    a warning. Call :func:`prepare_linestrings` beforehand to opt in to this
    explicitly and keep control of the row index.

    Parameters
    ----------
    linestring_features: geopandas.GeoDataFrame
        Features to split; other columns are carried over onto each piece.
    grid: GridDefinition
        Grid to split along.
    bounded: bool
        If False (the default), a feature is split for its whole length,
        including any part that falls outside the grid. If True, splitting
        stops at the grid's edge: pieces outside the grid are left whole
        rather than cut at every gridline they would otherwise cross.
    """
    if (linestring_features.geometry.geom_type == "MultiLineString").any():
        logger.warning(
            "Found MultiLineString geometries, coercing to LineStrings "
            "(matching prepare_linestrings: merge then explode, index is reset)"
        )
        linestring_features = prepare_linestrings(linestring_features.copy())
    # split every feature in one call
    geometry, parent = _split(
        split_linestrings_core,
        linestring_features.geometry,
        grid,
        bounded=bounded,
    )
    logger.info(f"Split {len(linestring_features)} edges into {len(geometry)} pieces")
    return _splits_frame(linestring_features, geometry, parent, grid)


def split_polygons(
    polygon_features: geopandas.GeoDataFrame, grid: GridDefinition
) -> geopandas.GeoDataFrame:
    """Split polygons along a grid using the bounded C++ splitter."""
    # split every feature in one call: crossing into the extension per
    # feature costs far more than the splitting itself
    geometry, parent = _split(
        split_polygons_core, polygon_features.geometry.to_numpy(), grid
    )
    logger.info(f"Split {len(polygon_features)} features into {len(geometry)} pieces")
    return _splits_frame(polygon_features, geometry, parent, grid)


def split_geometries(
    features: geopandas.GeoDataFrame,
    grid: GridDefinition,
    bounded=False,
) -> geopandas.GeoDataFrame:
    """Split features of any geometry type along a grid

    Unlike :func:`split_linestrings` and :func:`split_polygons_experimental`,
    this does not require every feature to be the same type. Each is handled
    on its own terms: LineStrings and Polygons are split, Points pass through
    unchanged, multi-part geometries are split part by part, and a
    GeometryCollection is split member by member. Every piece carries the
    attributes of the feature it came from, whatever that feature was, and an
    empty geometry comes back as itself rather than dropping its row.

    This is what to reach for when a layer holds more than one geometry type.
    For a layer that holds only one, the typed functions are cheaper: they
    read and write the coordinates as Arrow buffers, where this serialises
    every geometry through WKB in both directions.

    Parameters
    ----------
    features: geopandas.GeoDataFrame
        Features to split; other columns are carried over onto each piece.
    grid: GridDefinition
        Grid to split along.
    bounded: bool
        As :func:`split_linestrings`. Applies to the LineStrings among the
        features; Polygons are always split for their whole extent.

    Returns
    -------
    geopandas.GeoDataFrame
        One row per piece, with a ``"split"`` column numbering each feature's
        pieces from zero.
    """
    geometry, parent = _split(
        split_geometries_core,
        features.geometry,
        grid,
        encoding="WKB",
        bounded=bounded,
    )
    logger.info(f"Split {len(features)} features into {len(geometry)} pieces")
    return _splits_frame(features, geometry, parent, grid)


def _try_merge(geom):
    if geom.geom_type == "MultiLineString":
        geom = linemerge(geom)
    return geom


def get_raster_values_for_splits(
    splits: pandas.DataFrame,
    data: Union[
        numpy.ndarray,
        "xarray.DataArray",
        "dask.array.Array",
    ],
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
    data: numpy.ndarray or xarray.DataArray or dask.array.Array
        2D raster values. DataArray inputs must have two spatial
        dimensions (band dimensions should be squeezed prior to calling).
    index_i: str
        Column name for i-indices
    index_j: str
        Column name for j-indices

    Returns
    -------
    pd.Series
        Series of raster values, with same row indexing as df.
    """
    indices_i = splits[index_i].to_numpy()
    indices_j = splits[index_j].to_numpy()
    valid_mask = (indices_i >= 0) & (indices_j >= 0)
    valid_positions = numpy.nonzero(valid_mask)[0]

    if len(valid_positions) == 0:
        return _build_series(splits.index, valid_positions, numpy.array([]))

    if isinstance(data, numpy.ndarray) or _is_dask_array(data):
        backing_data = data
    elif _is_xarray_dataarray(data):
        backing_data = data.data
    else:
        raise TypeError(
            "Unsupported raster data type; expected numpy.ndarray, xarray.DataArray or dask.array.Array."
        )

    indices_i_valid = indices_i[valid_positions]
    indices_j_valid = indices_j[valid_positions]

    if isinstance(backing_data, numpy.ndarray):
        splits_values = backing_data[indices_j_valid, indices_i_valid]
        return _build_series(splits.index, valid_positions, splits_values)

    elif _is_dask_array(backing_data):
        splits_values_da = backing_data.vindex[indices_j_valid, indices_i_valid]
        splits_values = numpy.asarray(splits_values_da.compute())
        return _build_series(splits.index, valid_positions, splits_values)

    else:
        raise NotImplementedError("data array backends must be NumPy or Dask arrays.")


def read_raster_values_for_splits(
    splits: pandas.DataFrame,
    raster,
    bands,
    index_i: str = "index_i",
    index_j: str = "index_j",
    max_raster_memory_mb: int = 256,
) -> dict[int, pandas.Series]:
    """Read selected raster bands through bounded rasterio windows.

    This is the file-backed counterpart to :func:`get_raster_values_for_splits`.
    The raster is opened once and all requested bands are read together in each
    window.  Invalid cells are left as NaN and duplicate split rows retain their
    original order and labels. Index columns use whole-raster coordinates and
    must contain integer values; positive and negative out-of-bounds indices are
    treated as invalid.
    """
    if max_raster_memory_mb <= 0:
        raise ValueError("max_raster_memory_mb must be positive")
    bands = tuple(dict.fromkeys(int(b) for b in bands))
    if not bands:
        return {}
    opener = rasterio.open(raster) if isinstance(raster, (str, os.PathLike)) else None
    dataset = opener or raster
    try:
        invalid_bands = [band for band in bands if band not in dataset.indexes]
        if invalid_bands:
            raise ValueError(f"Raster does not contain band(s) {invalid_bands}")
        ii, finite_i = _integer_indices(splits[index_i], index_i)
        jj, finite_j = _integer_indices(splits[index_j], index_j)
        valid = (
            finite_i
            & finite_j
            & (ii >= 0)
            & (jj >= 0)
            & (ii < dataset.width)
            & (jj < dataset.height)
        )
        positions = numpy.flatnonzero(valid)
        result_values = {
            band: numpy.empty(len(positions), dtype=dataset.dtypes[band - 1])
            for band in bands
        }
        if not len(positions):
            return {
                band: _build_series(splits.index, positions, values)
                for band, values in result_values.items()
            }
        budget = int(max_raster_memory_mb * 1024 * 1024)
        read_dtype = numpy.result_type(
            *(numpy.dtype(dataset.dtypes[band - 1]) for band in bands)
        )
        bytes_per_pixel = read_dtype.itemsize * len(bands)
        max_pixels = budget // bytes_per_pixel
        if max_pixels < 1:
            raise ValueError(
                "max_raster_memory_mb is too small to read one pixel "
                "across the requested bands"
            )
        row_to_value = numpy.full(len(splits), -1, dtype=int)
        row_to_value[positions] = numpy.arange(len(positions))

        def consume(window, window_positions):
            data = dataset.read(indexes=bands, window=window)
            r0, c0 = int(window.row_off), int(window.col_off)
            for k, band in enumerate(bands):
                result_values[band][row_to_value[window_positions]] = data[
                    k,
                    jj[window_positions] - r0,
                    ii[window_positions] - c0,
                ]

        block_h, block_w = dataset.block_shapes[bands[0] - 1]
        for window, window_positions in _plan_raster_windows(
            ii,
            jj,
            positions,
            dataset.width,
            dataset.height,
            block_h,
            block_w,
            max_pixels,
        ):
            consume(window, window_positions)
        return {
            band: _build_series(splits.index, positions, values)
            for band, values in result_values.items()
        }
    finally:
        if opener is not None:
            opener.close()


def _integer_indices(values: pandas.Series, name: str):
    """Return integer indices and a mask of finite input values."""
    try:
        numeric = values.to_numpy(dtype="float64", na_value=numpy.nan)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must contain numeric integer indices") from error
    finite = numpy.isfinite(numeric)
    if numpy.any(numeric[finite] != numpy.floor(numeric[finite])):
        raise ValueError(f"{name} must contain integer indices")
    indices = numpy.zeros(len(numeric), dtype=numpy.int64)
    indices[finite] = numeric[finite].astype(numpy.int64)
    return indices, finite


@dataclass
class _BlockRectangle:
    first_row: int
    last_row: int
    first_column: int
    last_column: int
    position_chunks: list[numpy.ndarray]


def _group_positions(row_keys, column_keys, positions):
    """Group positions by a pair of integer raster tile keys."""
    order = numpy.lexsort((column_keys, row_keys))
    ordered_rows = row_keys[order]
    ordered_columns = column_keys[order]
    ordered_positions = positions[order]
    changes = (
        numpy.flatnonzero(
            (ordered_rows[1:] != ordered_rows[:-1])
            | (ordered_columns[1:] != ordered_columns[:-1])
        )
        + 1
    )
    boundaries = numpy.concatenate(([0], changes, [len(order)]))
    return {
        (int(ordered_rows[start]), int(ordered_columns[start])): ordered_positions[
            start:end
        ]
        for start, end in pairwise(boundaries)
    }


def _plan_raster_windows(
    ii,
    jj,
    positions,
    raster_width,
    raster_height,
    block_height,
    block_width,
    max_pixels,
):
    """Plan bounded windows and the split positions served by each window."""
    min_i, max_i = int(ii[positions].min()), int(ii[positions].max())
    min_j, max_j = int(jj[positions].min()), int(jj[positions].max())
    tight_window = rasterio.windows.Window(
        min_i, min_j, max_i - min_i + 1, max_j - min_j + 1
    )
    if tight_window.width * tight_window.height <= max_pixels:
        return [(tight_window, positions)]

    native_block_pixels = block_height * block_width
    if native_block_pixels > max_pixels:
        return _plan_oversized_block_windows(
            ii,
            jj,
            positions,
            raster_width,
            raster_height,
            block_height,
            block_width,
            max_pixels,
        )

    block_rows = jj[positions] // block_height
    block_columns = ii[positions] // block_width
    grouped = _group_positions(block_rows, block_columns, positions)
    columns_by_row = defaultdict(list)
    for block_row, block_column in grouped:
        columns_by_row[block_row].append(block_column)

    max_blocks = max_pixels // native_block_pixels
    active = {}
    rectangles = []
    previous_row = None
    for block_row in sorted(columns_by_row):
        if previous_row is not None and block_row != previous_row + 1:
            rectangles.extend(active.values())
            active = {}
        runs = _horizontal_block_runs(
            block_row,
            sorted(columns_by_row[block_row]),
            grouped,
            max_blocks,
        )
        next_active = {}
        for run in runs:
            key = (run.first_column, run.last_column)
            rectangle = active.pop(key, None)
            run_width = run.last_column - run.first_column + 1
            if (
                rectangle is not None
                and (rectangle.last_row - rectangle.first_row + 2) * run_width
                <= max_blocks
            ):
                rectangle.last_row = block_row
                rectangle.position_chunks.extend(run.position_chunks)
                next_active[key] = rectangle
            else:
                if rectangle is not None:
                    rectangles.append(rectangle)
                next_active[key] = run
        rectangles.extend(active.values())
        active = next_active
        previous_row = block_row
    rectangles.extend(active.values())

    planned = []
    for rectangle in rectangles:
        row_off = rectangle.first_row * block_height
        col_off = rectangle.first_column * block_width
        row_end = min((rectangle.last_row + 1) * block_height, raster_height)
        col_end = min((rectangle.last_column + 1) * block_width, raster_width)
        planned.append(
            (
                rasterio.windows.Window(
                    col_off, row_off, col_end - col_off, row_end - row_off
                ),
                numpy.concatenate(rectangle.position_chunks),
            )
        )
    return planned


def _horizontal_block_runs(block_row, columns, grouped, max_blocks):
    """Build contiguous, budget-capped runs for one native block row."""
    runs = []
    start = 0
    while start < len(columns):
        end = start + 1
        while (
            end < len(columns)
            and columns[end] == columns[end - 1] + 1
            and end - start < max_blocks
        ):
            end += 1
        run_columns = columns[start:end]
        runs.append(
            _BlockRectangle(
                block_row,
                block_row,
                run_columns[0],
                run_columns[-1],
                [grouped[(block_row, column)] for column in run_columns],
            )
        )
        start = end
    return runs


def _plan_oversized_block_windows(
    ii,
    jj,
    positions,
    raster_width,
    raster_height,
    block_height,
    block_width,
    max_pixels,
):
    """Subdivide occupied native blocks which exceed the read budget."""
    tile_width = min(block_width, max_pixels)
    tile_height = max(1, min(block_height, max_pixels // tile_width))
    block_rows = jj[positions] // block_height
    block_columns = ii[positions] // block_width
    row_offsets = (
        block_rows * block_height
        + (jj[positions] % block_height // tile_height) * tile_height
    )
    column_offsets = (
        block_columns * block_width
        + (ii[positions] % block_width // tile_width) * tile_width
    )
    grouped = _group_positions(row_offsets, column_offsets, positions)
    planned = []
    for (row_off, col_off), window_positions in sorted(grouped.items()):
        native_row_end = min(
            (row_off // block_height + 1) * block_height, raster_height
        )
        native_col_end = min((col_off // block_width + 1) * block_width, raster_width)
        planned.append(
            (
                rasterio.windows.Window(
                    col_off,
                    row_off,
                    min(col_off + tile_width, native_col_end) - col_off,
                    min(row_off + tile_height, native_row_end) - row_off,
                ),
                window_positions,
            )
        )
    return planned


def _build_series(
    index: pandas.Index,
    positions: numpy.ndarray,
    values: numpy.ndarray,
) -> pandas.Series:
    """Create a Series indexed by index, filled with specific values at given
    positions, default numpy.nan.
    """
    if len(positions) == len(index):
        series = pandas.Series(numpy.empty(len(index), dtype=values.dtype), index=index)
        series.iloc[positions] = values
        return series
    series = pandas.Series(numpy.full(len(index), numpy.nan), index=index)
    if len(positions):
        series.iloc[positions] = values
    return series


def apply_indices(
    features: geopandas.GeoDataFrame,
    grid: GridDefinition,
    index_i="index_i",
    index_j="index_j",
) -> geopandas.GeoDataFrame:
    if features.empty:
        logger.info("Returning empty dataframe")
        # return an empty dataframe with the expected columns
        empty_df = features.copy()
        empty_df[index_i] = numpy.array([], dtype="int64")
        empty_df[index_j] = numpy.array([], dtype="int64")
        return empty_df

    def f(geom, *_, **__):
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
    i, j = get_cell_indices(
        geom, nrows=grid.height, ncols=grid.width, transform=grid.transform
    )

    # Raise error if cell index would be out of bounds
    # assert 0 <= i < t.width
    # assert 0 <= j < t.height

    # Or - special value (-1,-1) if cell would be out of bounds
    if i >= grid.width or i < 0 or j >= grid.height or j < 0:
        i = -1
        j = -1
    return pandas.Series(index=(index_i, index_j), data=[i, j])


def idx_to_ij(idx: int, width: int, height: int) -> tuple[int]:
    return numpy.unravel_index(idx, (height, width))


def ij_to_idx(ij: tuple[int], width: int, height: int):
    return numpy.ravel_multi_index(ij, (height, width))


def aggregate_values_to_grid(
    splits: geopandas.GeoDataFrame,
    value_column: str,
    grid: GridDefinition,
    index_i: str = "index_i",
    index_j: str = "index_j",
    fill_value: float = 0.0,
    dtype=None,
) -> numpy.ndarray:
    """Aggregate split-geometry attributes onto a raster-shaped array.

    Parameters
    ----------
    splits
        GeoDataFrame containing split geometries and raster index columns.
    value_column
        Name of the column to aggregate per cell (e.g. ``length_km``).
    grid
        A :class:`GridDefinition` that defines the raster bounds.
    index_i, index_j
        Column names storing raster column (``i``) and row (``j``) indices.
    fill_value
        Initial fill value for cells without observations.
    dtype
        Optional dtype for the resulting array. Defaults to promoting the
        column dtype with the fill value dtype.
    """
    missing = {value_column, index_i, index_j} - set(splits.columns)
    if missing:
        raise KeyError(
            f"Required columns {sorted(missing)} not present in splits dataframe"
        )
    df = (
        splits[[index_i, index_j, value_column]]
        .dropna(subset=(index_i, index_j))
        .copy()
    )

    height = grid.height
    width = grid.width

    df[index_i] = df[index_i].astype(int)
    df[index_j] = df[index_j].astype(int)

    in_bounds = (
        (df[index_i] >= 0)
        & (df[index_i] < width)
        & (df[index_j] >= 0)
        & (df[index_j] < height)
    )
    if not in_bounds.all():
        df = df[in_bounds]

    if df.empty:
        inferred_dtype = (
            numpy.array(fill_value).dtype if dtype is None else numpy.dtype(dtype)
        )
        return numpy.full((height, width), fill_value, dtype=inferred_dtype)

    # Grouby cell index and sum values
    df = df.groupby([index_j, index_i]).sum().reset_index()

    # Infer target dtype
    value_dtype = df[value_column].to_numpy(copy=False).dtype
    fill_dtype = numpy.array(fill_value).dtype
    target_dtype = (
        numpy.promote_types(value_dtype, fill_dtype)
        if dtype is None
        else numpy.dtype(dtype)
    )
    # Set up full result array
    result = numpy.full((height, width), fill_value, dtype=target_dtype)

    # Insert values from splits DataFrame
    rows = df[index_j].to_numpy(dtype=int, copy=False)
    cols = df[index_i].to_numpy(dtype=int, copy=False)
    vals = df[value_column].to_numpy(dtype=target_dtype, copy=False)
    result[(rows, cols)] = vals

    return result
