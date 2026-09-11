"""High-level overlay of raster values onto vector features

These functions wrap the lower-level steps in :mod:`snail.intersection`
(prepare, split, index, attribute) into single calls that work for point,
linestring and polygon features, handle single- or multi-band rasters, and
implicitly reproject features to the raster CRS (and back) if they differ.
"""

import json
import logging
from collections import defaultdict
from os import PathLike
from pathlib import Path

import geopandas
import numpy
import pandas
import pyarrow
import pyarrow.compute

from snail.intersection import (
    GridDefinition,
    apply_indices,
    get_raster_values_for_splits,
    prepare_linestrings,
    prepare_points,
    prepare_polygons,
    read_raster_values_for_splits,
    split_features_for_rasters,
    split_geometries,
    split_geometries_core,
    split_linestrings,
    split_linestrings_core,
    split_points,
    split_polygons,
    split_polygons_core,
)
from snail.io import (
    _is_rasterio_dataset,
    _open_raster,
    band_column_name,
    extend_rasters_metadata,
    read_raster_band_data,
    read_raster_metadata,
)

# Module-level logger
logger = logging.getLogger(__name__)

STREAM_BATCH_SIZE = 65_536


def _arrow_reader(features, batch_size=STREAM_BATCH_SIZE):
    """Normalise an Arrow object or GeoDataFrame to a RecordBatchReader."""
    if isinstance(features, geopandas.GeoDataFrame):
        table = pyarrow.table(features.to_arrow(geometry_encoding="WKB"))
        return pyarrow.RecordBatchReader.from_batches(
            table.schema,
            table.to_batches(max_chunksize=batch_size),
        )
    if isinstance(features, pyarrow.RecordBatchReader):
        return features
    if isinstance(features, pyarrow.RecordBatch):
        return pyarrow.RecordBatchReader.from_batches(features.schema, [features])
    if isinstance(features, pyarrow.Table):
        return pyarrow.RecordBatchReader.from_batches(
            features.schema, features.to_batches(max_chunksize=batch_size)
        )
    return pyarrow.RecordBatchReader.from_stream(features)


def _geometry_field(schema):
    fields = []
    for field in schema:
        extension = (field.metadata or {}).get(b"ARROW:extension:name", b"")
        if extension.startswith((b"geoarrow.", b"ogc.")):
            fields.append(field)
    if len(fields) != 1:
        raise ValueError("Expected exactly one GeoArrow geometry column")
    return fields[0]


def _wkb_field(field, fallback_crs=None):
    metadata = dict(field.metadata or {})
    metadata[b"ARROW:extension:name"] = b"geoarrow.wkb"
    extension_metadata = json.loads(metadata.get(b"ARROW:extension:metadata", b"{}"))
    if extension_metadata.get("crs") is None and fallback_crs is not None:
        import pyproj

        extension_metadata["crs"] = pyproj.CRS.from_user_input(
            fallback_crs
        ).to_json_dict()
        metadata[b"ARROW:extension:metadata"] = json.dumps(extension_metadata).encode()
    return pyarrow.field(
        field.name, pyarrow.binary(), nullable=False, metadata=metadata
    )


def _split_schema(schema, index_i, index_j, fallback_crs=None):
    geometry = _geometry_field(schema)
    replaced = {"split", index_i, index_j}
    fields = []
    for field in schema:
        if field.name in replaced:
            continue
        fields.append(
            _wkb_field(field, fallback_crs) if field.name == geometry.name else field
        )
    fields.extend(
        [
            pyarrow.field("split", pyarrow.int64(), nullable=False),
            pyarrow.field(index_i, pyarrow.int64(), nullable=False),
            pyarrow.field(index_j, pyarrow.int64(), nullable=False),
        ]
    )
    return pyarrow.schema(fields, metadata=schema.metadata)


def _split_arrow_stage(
    source,
    grid,
    index_i="index_i",
    index_j="index_j",
    split_batch_size=STREAM_BATCH_SIZE,
):
    """Split one Arrow stream on one grid without collecting its batches."""
    if split_batch_size <= 0:
        raise ValueError("split_batch_size must be positive")
    source = _arrow_reader(source)
    source_geometry = _geometry_field(source.schema)
    output_schema = _split_schema(source.schema, index_i, index_j, grid.crs)

    def batches():
        for source_batch in source:
            if source_batch.num_rows == 0:
                continue
            valid_geometry = pyarrow.compute.invert(
                pyarrow.compute.is_null(
                    source_batch.column(source_geometry.name), nan_is_null=False
                )
            )
            source_batch = source_batch.filter(valid_geometry)
            if source_batch.num_rows == 0:
                continue
            source_frame = geopandas.GeoDataFrame.from_arrow(source_batch)
            source_crs = source_frame.crs
            grid_frame = source_frame
            if source_crs is not None and grid.crs is not None:
                if not _crs_equal(source_crs, grid.crs):
                    grid_frame = source_frame.to_crs(grid.crs)
            elif source_crs is None and grid.crs is not None:
                grid_frame = source_frame.set_crs(grid.crs)

            geometry_types = set(grid_frame.geometry.geom_type.dropna())
            if geometry_types == {"LineString"}:
                encoding = "geoarrow"
                split_core = split_linestrings_core
            elif geometry_types == {"Polygon"}:
                encoding = "geoarrow"
                split_core = split_polygons_core
            else:
                encoding = "WKB"
                split_core = split_geometries_core
            arrow_kwargs = {"geometry_encoding": encoding}
            if encoding == "geoarrow":
                arrow_kwargs["interleaved"] = True
            geometry_table = pyarrow.table(
                grid_frame[[grid_frame.geometry.name]].to_arrow(**arrow_kwargs)
            )
            stream = split_core(
                geometry_table,
                nrows=grid.height,
                ncols=grid.width,
                transform=grid.transform,
                max_output_rows=split_batch_size,
            )
            piece_reader = pyarrow.RecordBatchReader.from_stream(stream)
            piece_numbers = defaultdict(int)
            for pieces in piece_reader:
                parents = pieces.column("parent")
                parent_values = parents.to_numpy(zero_copy_only=False)
                piece_frame = geopandas.GeoDataFrame.from_arrow(
                    pyarrow.Table.from_arrays(
                        [pieces.column("geometry")],
                        schema=pyarrow.schema([pieces.schema.field("geometry")]),
                    )
                )
                indexed = apply_indices(piece_frame, grid, index_i, index_j)
                if source_crs is not None and grid.crs is not None:
                    if not _crs_equal(source_crs, grid.crs):
                        indexed = indexed.to_crs(source_crs)
                elif source_crs is not None and indexed.crs is None:
                    indexed = indexed.set_crs(source_crs)

                ordinals = numpy.empty(len(parent_values), dtype=numpy.int64)
                for offset, parent in enumerate(parent_values):
                    ordinals[offset] = piece_numbers[int(parent)]
                    piece_numbers[int(parent)] += 1

                arrays = []
                for field in output_schema:
                    if field.name == source_geometry.name:
                        geometry = (
                            pyarrow.table(
                                indexed[[indexed.geometry.name]].to_arrow(
                                    geometry_encoding="WKB"
                                )
                            )
                            .column(indexed.geometry.name)
                            .combine_chunks()
                        )
                        arrays.append(geometry)
                    elif field.name == "split":
                        arrays.append(pyarrow.array(ordinals, type=field.type))
                    elif field.name in (index_i, index_j):
                        arrays.append(
                            pyarrow.array(indexed[field.name], type=field.type)
                        )
                    else:
                        arrays.append(
                            pyarrow.compute.take(
                                source_batch.column(field.name), parents
                            )
                        )
                yield pyarrow.RecordBatch.from_arrays(arrays, schema=output_schema)

    return pyarrow.RecordBatchReader.from_batches(output_schema, batches())


def _raster_info(raster):
    with _open_raster(raster) as dataset:
        grid = GridDefinition.from_rasterio(dataset)
        bands = tuple(dataset.indexes)
        layouts = {
            band: (dataset.dtypes[band - 1], dataset.block_shapes[band - 1])
            for band in bands
        }
    return grid, bands, layouts


def _raster_output_schema(schema, layouts, bands, column, all_bands):
    additions = {}
    for band in bands:
        name = band_column_name(column, band, len(all_bands))
        additions[name] = pyarrow.field(
            name,
            pyarrow.from_numpy_dtype(numpy.dtype(layouts[band][0])),
            nullable=True,
        )
    fields = [field for field in schema if field.name not in additions]
    fields.extend(additions.values())
    return pyarrow.schema(fields, metadata=schema.metadata), additions


def _attribute_arrow_stage(
    source,
    raster,
    bands,
    column,
    all_bands,
    index_i,
    index_j,
    max_raster_memory_mb,
    layouts,
):
    source = _arrow_reader(source)
    invalid = [band for band in bands if band not in layouts]
    if invalid:
        raise ValueError(f"Raster does not contain band(s) {invalid}")
    output_schema, additions = _raster_output_schema(
        source.schema, layouts, bands, column, all_bands
    )
    bands_by_column = {
        band_column_name(column, band, len(all_bands)): band for band in bands
    }

    def batches():
        with _open_raster(raster) as dataset:
            for batch in source:
                frame = geopandas.GeoDataFrame.from_arrow(batch)
                values = read_raster_values_for_splits(
                    frame,
                    dataset,
                    bands,
                    index_i,
                    index_j,
                    max_raster_memory_mb,
                )
                arrays = []
                for field in output_schema:
                    if field.name in additions:
                        band = bands_by_column[field.name]
                        arrays.append(
                            pyarrow.array(
                                values[band], type=field.type, from_pandas=True
                            )
                        )
                    else:
                        arrays.append(batch.column(field.name))
                yield pyarrow.RecordBatch.from_arrays(arrays, schema=output_schema)

    return pyarrow.RecordBatchReader.from_batches(output_schema, batches())


def iter_split_features_batches(
    features,
    grid: GridDefinition,
    *,
    split_batch_size: int = STREAM_BATCH_SIZE,
) -> pyarrow.RecordBatchReader:
    """Lazily split an Arrow feature stream and return Arrow result batches."""
    return _split_arrow_stage(features, grid, split_batch_size=split_batch_size)


def iter_overlay_raster_batches(
    features,
    raster,
    bands: list[int] | None = None,
    column: str | None = None,
    *,
    max_raster_memory_mb: int = 256,
    split_batch_size: int = STREAM_BATCH_SIZE,
) -> pyarrow.RecordBatchReader:
    """Lazily split Arrow feature batches and attribute one raster."""
    if max_raster_memory_mb <= 0:
        raise ValueError("max_raster_memory_mb must be positive")
    if not (isinstance(raster, (str, PathLike)) or _is_rasterio_dataset(raster)):
        raise TypeError(
            "Streaming raster attribution requires a rasterio dataset or path"
        )
    grid, all_bands, layouts = _raster_info(raster)
    selected = list(all_bands if bands is None else dict.fromkeys(map(int, bands)))
    column = _raster_key(raster) if column is None else column
    splits = _split_arrow_stage(features, grid, split_batch_size=split_batch_size)
    return _attribute_arrow_stage(
        splits,
        raster,
        selected,
        column,
        all_bands,
        "index_i",
        "index_j",
        max_raster_memory_mb,
        layouts,
    )


def iter_overlay_rasters_batches(
    features,
    rasters: list | pandas.DataFrame,
    *,
    max_raster_memory_mb: int = 256,
    split_batch_size: int = STREAM_BATCH_SIZE,
) -> pyarrow.RecordBatchReader:
    """Lazily split Arrow batches over all grids and attribute all rasters."""
    if max_raster_memory_mb <= 0:
        raise ValueError("max_raster_memory_mb must be positive")
    rasters = _normalise_rasters(rasters)
    if any(
        not (isinstance(path, (str, PathLike)) or _is_rasterio_dataset(path))
        for path in rasters.path
    ):
        raise TypeError(
            "Streaming raster attribution requires rasterio datasets or paths"
        )

    grids = []
    grid_ids = []
    selected_bands = []
    raster_infos = []
    for raster in rasters.itertuples():
        grid, all_bands, layouts = _raster_info(raster.path)
        if grid not in grids:
            grids.append(grid)
        grid_ids.append(grids.index(grid))
        given = getattr(raster, "bands", None)
        selected_bands.append(all_bands if given is None else given)
        raster_infos.append((all_bands, layouts))
    rasters["grid_id"] = grid_ids
    rasters["bands"] = selected_bands

    result = _arrow_reader(features)
    for grid_id, grid in enumerate(grids):
        result = _split_arrow_stage(
            result,
            grid,
            f"i_{grid_id}",
            f"j_{grid_id}",
            split_batch_size,
        )
    for raster, (all_bands, layouts) in zip(rasters.itertuples(), raster_infos):
        result = _attribute_arrow_stage(
            result,
            raster.path,
            list(raster.bands),
            raster.key,
            all_bands,
            f"i_{raster.grid_id}",
            f"j_{raster.grid_id}",
            max_raster_memory_mb,
            layouts,
        )
    return result


def overlay_raster(
    features: geopandas.GeoDataFrame,
    raster,
    bands: list[int] | None = None,
    column: str | None = None,
    max_raster_memory_mb: int = 256,
) -> geopandas.GeoDataFrame:
    """Split features along a raster grid and attribute cell values

    Parameters
    ----------
    features : geopandas.GeoDataFrame
        Point, LineString or Polygon features (multi-geometries are exploded)
    raster : str | pathlib.Path | rasterio dataset | xarray.DataArray
        Raster file path, open rasterio dataset or DataArray, defining the
        splitting grid and providing cell values
    bands : list of int, optional
        Band numbers to attribute (default: all bands)
    column : str, optional
        Output column name (default: raster filename stem). Values from a
        single band are attributed under this name directly, multiple bands
        under "{column}_band_{n}" for each band n.
    max_raster_memory_mb : int
        Maximum size, in MiB, of each application-managed raster read buffer.

    Returns
    -------
    geopandas.GeoDataFrame
        Split features in the CRS of the input features, with grid cell
        indices in columns "index_i" and "index_j" and one column of raster
        values per band. Features that fall outside the raster are attributed
        NaN. If the features and raster CRS differ, features are reprojected
        to the raster CRS for splitting and lookup, then reprojected back.
    """
    if isinstance(raster, (str, PathLike)) or _is_rasterio_dataset(raster):
        return geopandas.GeoDataFrame.from_arrow(
            iter_overlay_raster_batches(
                features,
                raster,
                bands,
                column,
                max_raster_memory_mb=max_raster_memory_mb,
            ).read_all()
        )

    grid, all_bands = read_raster_metadata(raster)
    if bands is None:
        bands = list(all_bands)
    if column is None:
        column = _raster_key(raster)

    splits = split_features(features, grid)
    if isinstance(raster, (str, PathLike)) or _is_rasterio_dataset(raster):
        values = read_raster_values_for_splits(
            splits, raster, bands, max_raster_memory_mb=max_raster_memory_mb
        )
    else:
        values = {
            band_number: get_raster_values_for_splits(
                splits, read_raster_band_data(raster, int(band_number))
            )
            for band_number in bands
        }
    for band_number in bands:
        band_column = band_column_name(column, band_number, len(all_bands))
        logger.info(
            "Attributing values from %s band %s in column %s",
            _describe_raster(raster),
            band_number,
            band_column,
        )
        splits[band_column] = values[band_number]
    return splits


def overlay_rasters(
    features: geopandas.GeoDataFrame,
    rasters: list | pandas.DataFrame,
    max_raster_memory_mb: int = 256,
) -> geopandas.GeoDataFrame:
    """Split features along multiple raster grids and attribute cell values

    All features are intersected with all rasters: each raster band
    contributes one column of values to the output.

    Parameters
    ----------
    features : geopandas.GeoDataFrame
        Point, LineString or Polygon features (multi-geometries are exploded)
    rasters : list | pandas.DataFrame
        Either a sequence of raster file paths or open rasterio datasets, or
        a DataFrame. Its required ``path`` column contains a path or dataset;
        optional ``bands`` and ``key`` columns select bands and output names.
    max_raster_memory_mb : int
        Maximum size, in MiB, of each application-managed raster read buffer.

    Returns
    -------
    geopandas.GeoDataFrame
        Split features in the CRS of the input features, with cell indices in
        columns "i_{n}", "j_{n}" for each distinct grid n, and one column of
        raster values per raster band, named by raster key (with a
        "_band_{n}" suffix for each band of a multi-band raster)
    """
    normalised = _normalise_rasters(rasters)
    if all(
        isinstance(path, (str, PathLike)) or _is_rasterio_dataset(path)
        for path in normalised.path
    ):
        return geopandas.GeoDataFrame.from_arrow(
            iter_overlay_rasters_batches(
                features,
                normalised,
                max_raster_memory_mb=max_raster_memory_mb,
            ).read_all()
        )

    rasters = normalised
    rasters, grids = extend_rasters_metadata(rasters)
    prepare, split_func = _prepare_and_split_funcs(features)
    prepared = prepare(features)
    splits = split_features_for_rasters(prepared, grids, split_func)

    # to prevent a fragmented dataframe (and a memory explosion), add series to a dict
    # and then concat afterwards -- do not append to an existing dataframe
    raster_data: dict[str, pandas.Series] = {}
    # associate values
    for raster in rasters.itertuples():
        _, all_bands = read_raster_metadata(raster.path)
        if isinstance(raster.path, (str, PathLike)) or _is_rasterio_dataset(
            raster.path
        ):
            values = read_raster_values_for_splits(
                splits,
                raster.path,
                raster.bands,
                f"i_{raster.grid_id}",
                f"j_{raster.grid_id}",
                max_raster_memory_mb,
            )
        else:
            values = {
                band_number: get_raster_values_for_splits(
                    splits,
                    read_raster_band_data(raster.path, int(band_number)),
                    f"i_{raster.grid_id}",
                    f"j_{raster.grid_id}",
                )
                for band_number in raster.bands
            }
        for band_number in raster.bands:
            logger.info(
                "Associating values from raster %s grid %s band %s",
                raster.key,
                raster.grid_id,
                band_number,
            )
            column = band_column_name(raster.key, band_number, len(all_bands))
            raster_data[column] = values[band_number]

    raster_data = pandas.DataFrame(raster_data)
    splits = pandas.concat([splits, raster_data], axis="columns")
    return splits


def split_features(
    features: geopandas.GeoDataFrame,
    grid: GridDefinition,
) -> geopandas.GeoDataFrame:
    """Split point, linestring or polygon features along a grid

    Features are implicitly reprojected to the grid CRS for splitting and
    indexing, then returned in their original CRS. If either the features or
    the grid have no CRS defined, they are assumed to share the same CRS.

    Parameters
    ----------
    features : geopandas.GeoDataFrame
        Point, LineString or Polygon features (multi-geometries are exploded)
    grid : GridDefinition
        Grid to split features along

    Returns
    -------
    geopandas.GeoDataFrame
        Split features with grid cell indices in columns "index_i" and
        "index_j" (set to -1 for features outside the grid)
    """
    if features.empty:
        return apply_indices(features, grid)
    return geopandas.GeoDataFrame.from_arrow(
        iter_split_features_batches(features, grid).read_all()
    )


def _prepare_and_split_funcs(features: geopandas.GeoDataFrame):
    """Pick prepare and split functions for the features' geometry type"""
    kinds = _geom_kinds(features)
    if len(kinds) > 1 or "GeometryCollection" in kinds:
        # No typed split can take a layer of several kinds at once, and the
        # first feature's type is no guide to the rest of them
        logger.info("Splitting mixed geometries (%s)", ", ".join(sorted(kinds)))
        # split_geometries explodes multi-part geometries itself, so there is
        # no separate preparation step
        return lambda f: f, split_geometries
    geom_type = _sample_geom_type(features)
    if "Point" in geom_type:
        return prepare_points, split_points
    elif "LineString" in geom_type:
        return prepare_linestrings, split_linestrings
    elif "Polygon" in geom_type:
        return prepare_polygons, split_polygons
    raise ValueError(f"Could not process vector data of type {geom_type}")


def _sample_geom_type(features: geopandas.GeoDataFrame) -> str:
    if features.empty:
        raise ValueError("Expected features, got an empty GeoDataFrame")
    return features.iloc[0].geometry.geom_type


def _geom_kinds(features: geopandas.GeoDataFrame) -> set:
    """What kinds of geometry a layer holds.

    Multi-part types count as their single-part kind, because that is what
    decides how a layer is split: LineStrings and MultiLineStrings together
    are one kind of thing, and prepare_linestrings turns the second into the
    first. LineStrings and Polygons together are two, and no single typed
    split can take them.
    """
    return {
        kind.replace("Multi", "")
        for kind in features.geometry.geom_type.dropna().unique()
    }


def _crs_equal(a, b) -> bool:
    import pyproj

    return pyproj.CRS.from_user_input(a) == pyproj.CRS.from_user_input(b)


def _raster_key(raster) -> str:
    """Default output column name for a raster path or open dataset"""
    name = getattr(raster, "name", None)
    if name is None:
        if isinstance(raster, (str, Path)):
            name = raster
        else:
            return "raster"
    stem = Path(str(name)).stem
    return stem if stem else "raster"


def _format_key(row, colnames):
    if colnames:
        # stitch together from metadata columns
        parts = []
        for c in colnames:
            parts.append(f"{c}:{row.loc[c]}")
        key = "|".join(parts)
    else:
        # fall back to path as key
        key = _raster_key(row.loc["path"])
    return key


def _describe_raster(raster) -> str:
    return str(getattr(raster, "name", raster))


def _normalise_rasters(rasters) -> pandas.DataFrame:
    """Coerce a sequence of paths/datasets or a DataFrame to a rasters table

    Ensures "path" and "key" columns, and parses any "bands" values.
    """
    if isinstance(rasters, pandas.DataFrame):
        df = rasters.copy()
        if "path" not in df.columns:
            raise ValueError("Expected rasters DataFrame to have a 'path' column")
    else:
        df = pandas.DataFrame({"path": list(rasters)})

    if "key" not in df.columns:
        colnames = sorted(set(df.columns) - {"path", "bands"})
        keys = [_format_key(row, colnames) for _, row in df.iterrows()]

        if len(set(keys)) != len(keys):
            # duplicate filename stems - fall back to full paths as keys
            keys = [_raster_key(p) for p in df.path]
        df["key"] = keys

    if "bands" in df.columns:
        df["bands"] = df["bands"].apply(parse_bands)
    return df


def parse_bands(value) -> tuple | None:
    """Parse a band numbers value to a tuple of ints

    Accepts an int, a comma-separated string ("1,2,3"), or a list/tuple of
    ints. Returns None for missing values (None or NaN), meaning "all bands".
    """
    if value is None:
        return None
    if isinstance(value, (list, tuple)):
        return tuple(int(b) for b in value)
    if isinstance(value, float):
        if pandas.isna(value):
            return None
        return (int(value),)
    if isinstance(value, str) and not value.strip():
        return None
    if isinstance(value, str):
        return tuple(int(b) for b in value.split(","))
    return (int(value),)
