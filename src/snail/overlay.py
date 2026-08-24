"""High-level overlay of raster values onto vector features

These functions wrap the lower-level steps in :mod:`snail.intersection`
(prepare, split, index, attribute) into single calls that work for point,
linestring and polygon features, handle single- or multi-band rasters, and
implicitly reproject features to the raster CRS (and back) if they differ.
"""

import logging
from pathlib import Path

import geopandas
import pandas

from snail.intersection import (
    GridDefinition,
    apply_indices,
    get_raster_values_for_splits,
    prepare_linestrings,
    prepare_points,
    prepare_polygons,
    split_features_for_rasters,
    split_linestrings,
    split_points,
    split_polygons,
    split_polygons_experimental,
)
from snail.io import (
    associate_raster_files,
    band_column_name,
    extend_rasters_metadata,
    read_raster_band_data,
    read_raster_metadata,
)

# Module-level logger
logger = logging.getLogger(__name__)


def overlay_raster(
    features: geopandas.GeoDataFrame,
    raster,
    bands: list[int] | None = None,
    column: str | None = None,
    experimental: bool = False,
    lazy: bool = False,
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
    experimental : bool
        Use the experimental (faster, less robust) polygon splitting routine
    lazy : bool
        Read raster bands lazily via xarray/dask rather than into memory.
        Only applies when `raster` is a file path.

    Returns
    -------
    geopandas.GeoDataFrame
        Split features in the CRS of the input features, with grid cell
        indices in columns "index_i" and "index_j" and one column of raster
        values per band. Features that fall outside the raster are attributed
        NaN. If the features and raster CRS differ, features are reprojected
        to the raster CRS for splitting and lookup, then reprojected back.
    """
    grid, all_bands = read_raster_metadata(raster)
    if bands is None:
        bands = list(all_bands)
    if column is None:
        column = _raster_key(raster)

    splits = split_features(features, grid, experimental=experimental)
    for band_number in bands:
        band_column = band_column_name(column, band_number, len(all_bands))
        logger.info(
            "Attributing values from %s band %s in column %s",
            _describe_raster(raster),
            band_number,
            band_column,
        )
        band_data = read_raster_band_data(raster, int(band_number), lazy=lazy)
        splits[band_column] = get_raster_values_for_splits(splits, band_data)
    return splits


def overlay_rasters(
    features: geopandas.GeoDataFrame,
    rasters: list | pandas.DataFrame,
    experimental: bool = False,
    lazy: bool = False,
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
        a DataFrame with columns:
            - "path" (required): file path or open rasterio dataset
            - "bands" (optional): band numbers to attribute, as an int, a
              comma-separated string ("1,2,3"), or a list/tuple of ints -
              defaults to all bands
            - "key" (optional): output column name - defaults to the raster
              filename stem
    experimental : bool
        Use the experimental (faster, less robust) polygon splitting routine
    lazy : bool
        Read raster bands lazily via xarray/dask rather than into memory

    Returns
    -------
    geopandas.GeoDataFrame
        Split features in the CRS of the input features, with cell indices in
        columns "i_{n}", "j_{n}" for each distinct grid n, and one column of
        raster values per raster band, named by raster key (with a
        "_band_{n}" suffix for each band of a multi-band raster)
    """
    rasters = _normalise_rasters(rasters)
    rasters, grids = extend_rasters_metadata(rasters)
    prepare, split_func = _prepare_and_split_funcs(features, experimental)
    prepared = prepare(features)
    splits = split_features_for_rasters(prepared, grids, split_func)
    return associate_raster_files(splits, rasters, lazy=lazy)


def split_features(
    features: geopandas.GeoDataFrame,
    grid: GridDefinition,
    experimental: bool = False,
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
    experimental : bool
        Use the experimental (faster, less robust) polygon splitting routine

    Returns
    -------
    geopandas.GeoDataFrame
        Split features with grid cell indices in columns "index_i" and
        "index_j" (set to -1 for features outside the grid)
    """
    if features.empty:
        return apply_indices(features, grid)
    source_crs = features.crs
    prepare, split_func = _prepare_and_split_funcs(features, experimental)
    prepared = prepare(features)

    if source_crs is None or grid.crs is None:
        if (source_crs is None) != (grid.crs is None):
            logger.warning(
                "CRS undefined for features (%s) or grid (%s): assuming they share a CRS",
                source_crs,
                grid.crs,
            )
        reprojected = False
    elif _crs_equal(source_crs, grid.crs):
        reprojected = False
    else:
        logger.info(
            "Reprojecting features from %s to grid CRS %s for splitting",
            source_crs,
            grid.crs,
        )
        prepared = prepared.to_crs(grid.crs)
        reprojected = True

    splits = split_func(prepared, grid)
    splits = apply_indices(splits, grid)

    if reprojected:
        splits = splits.to_crs(source_crs)
    elif source_crs is not None and splits.crs is None:
        splits = splits.set_crs(source_crs)
    return splits


def _prepare_and_split_funcs(features: geopandas.GeoDataFrame, experimental: bool):
    """Pick prepare and split functions for the features' geometry type"""
    geom_type = _sample_geom_type(features)
    if "Point" in geom_type:
        return prepare_points, split_points
    elif "LineString" in geom_type:
        return prepare_linestrings, split_linestrings
    elif "Polygon" in geom_type:
        if experimental:
            return prepare_polygons, split_polygons_experimental
        return prepare_polygons, split_polygons
    raise ValueError(f"Could not process vector data of type {geom_type}")


def _sample_geom_type(features: geopandas.GeoDataFrame) -> str:
    if features.empty:
        raise ValueError("Expected features, got an empty GeoDataFrame")
    return features.iloc[0].geometry.geom_type


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
        keys = [_raster_key(p) for p in df.path]
        if len(set(keys)) != len(keys):
            # duplicate filename stems - fall back to full paths as keys
            keys = [_describe_raster(p) for p in df.path]
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
