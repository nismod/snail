import importlib.util
import logging
from contextlib import contextmanager
from os import PathLike
from pathlib import Path
from typing import TYPE_CHECKING, Union

import geopandas
import numpy
import pandas
import rasterio
import rioxarray

from snail.intersection import (
    GridDefinition,
    _is_xarray_dataarray,
    get_raster_values_for_splits,
)

if TYPE_CHECKING:
    import xarray

# Module-level logger
logger = logging.getLogger(__name__)


def band_column_name(key: str, band_number: int, number_of_bands: int) -> str:
    """Name the output column for a raster band.

    Single-band rasters attribute values in a column named by `key`,
    multi-band rasters in a column per band, named "{key}_band_{band_number}".
    """
    if number_of_bands == 1:
        return key
    return f"{key}_band_{band_number}"


def associate_raster_files(splits, rasters, lazy: bool = False):
    """Read values from a list of raster files for a set of indexed split geometries

    Parameters
    ----------
    splits: pandas.DataFrame
        split geometries with raster indices in columns named "i_{grid_id}", "j_{grid_id}"
        for each grid_id in `rasters`

    rasters: pandas.DataFrame
        table of raster metadata with columns: key, grid_id, path, bands

    lazy: bool, default False
        When True, raster bands are opened lazily via xarray/dask. Requires
        optional dependencies (xarray, dask, rioxarray).

    Returns
    -------
    pandas.DataFrame
        split geometries with raster data values at indexed locations, one
        column per raster band, named by raster key (with a "_band_{n}" suffix
        for each band of a multi-band raster)
    """
    # to prevent a fragmented dataframe (and a memory explosion), add series to a dict
    # and then concat afterwards -- do not append to an existing dataframe
    raster_data: dict[str, pandas.Series] = {}

    # associate values
    for raster, band_number, band_data in read_rasters(rasters, lazy=lazy):
        logger.info(
            "Associating values from raster %s grid %s band %s",
            raster.key,
            raster.grid_id,
            band_number,
        )
        column = band_column_name(raster.key, band_number, len(raster.bands))
        raster_data[column] = get_raster_values_for_splits(
            splits,
            band_data,
            f"i_{raster.grid_id}",
            f"j_{raster.grid_id}",
        )

    raster_data = pandas.DataFrame(raster_data)
    splits = pandas.concat([splits, raster_data], axis="columns")

    return splits


def read_rasters(rasters, lazy: bool = False):
    for raster in rasters.itertuples():
        try:
            if lazy:
                data_array = rioxarray.open_rasterio(raster.path, chunks="auto")
                source = data_array
            else:
                data_array = None
                source = raster.path

            for band_number in raster.bands:
                yield (
                    raster,
                    band_number,
                    read_raster_band_data(source, band_number, lazy=lazy),
                )
        finally:
            if data_array is not None:
                data_array.close()


def _is_rasterio_dataset(value) -> bool:
    """True for an open rasterio dataset (duck-typed, covers the reader classes)"""
    return isinstance(value, rasterio.DatasetReader) or (
        hasattr(value, "read") and hasattr(value, "transform") and hasattr(value, "crs")
    )


@contextmanager
def _open_raster(raster):
    """Yield a rasterio dataset from either a path or an open dataset"""
    if _is_rasterio_dataset(raster):
        # already open - pass through, the caller owns closing it
        yield raster
    else:
        with rasterio.open(raster) as dataset:
            yield dataset


def read_raster_band_data(
    source: Union[str, PathLike, "xarray.DataArray"],
    band_number: int = 1,
    lazy: bool = False,
) -> Union[numpy.ndarray, "xarray.DataArray"]:
    """Read a single band from a raster path, open rasterio dataset or DataArray"""
    if band_number < 1:
        raise ValueError(f"band_number must be >= 1, got {band_number}")
    if _is_xarray_dataarray(source):
        return _select_dataarray_band(source, band_number)

    if _is_rasterio_dataset(source):
        return source.read(band_number)

    if isinstance(source, (str, PathLike)):
        if not lazy:
            with rasterio.open(source) as dataset:
                band_data: numpy.ndarray = dataset.read(band_number)
        else:
            data_array = rioxarray.open_rasterio(source, chunks="auto")
            band_data = _select_dataarray_band(data_array, band_number)
        return band_data

    raise TypeError(
        "Unsupported raster source; expected a path-like object, "
        "an open rasterio dataset or an xarray.DataArray."
    )


def extend_rasters_metadata(
    rasters: pandas.DataFrame,
) -> tuple[pandas.DataFrame, list[GridDefinition]]:
    grids = []
    grid_ids = []
    raster_bands = []

    for raster in rasters.itertuples():
        logger.info("Reading metadata from raster %s", raster.path)
        grid, bands = read_raster_metadata(raster.path)

        # add transform to list if not present
        if grid not in grids:
            grids.append(grid)

        # record raster/transform details
        grid_id = grids.index(grid)
        grid_ids.append(grid_id)
        raster_bands.append(bands)

    rasters["grid_id"] = grid_ids
    if "bands" in rasters.columns:
        # fill any missing values with all the bands discovered in the raster
        rasters["bands"] = [
            discovered if given is None else given
            for given, discovered in zip(rasters.bands, raster_bands)
        ]
    else:
        rasters["bands"] = raster_bands

    return rasters, grids


def read_raster_metadata(
    source: Union[str, PathLike, "xarray.DataArray"],
) -> tuple[GridDefinition, tuple[int]]:
    """Read grid definition and band indexes from a raster path, an open
    rasterio dataset or an xarray DataArray"""
    if _is_xarray_dataarray(source):
        return _read_dataarray_metadata(source)

    with _open_raster(source) as dataset:
        bands = dataset.indexes
        grid = GridDefinition.from_rasterio(dataset)
    return grid, bands


def read_features(path, layer=None):
    if Path(path).suffix in (".parquet", ".geoparquet"):
        features = geopandas.read_parquet(path)
    else:
        if importlib.util.find_spec("pyogrio"):
            engine = "pyogrio"
        else:
            engine = "fiona"
        if layer is not None:
            features = geopandas.read_file(path, layer=layer, engine=engine)
        else:
            features = geopandas.read_file(path, engine=engine)

    return features[~features.geometry.isna()]


def read_layer_names(path) -> list[str]:
    """List the layer names in a vector file"""
    if importlib.util.find_spec("pyogrio"):
        import pyogrio

        return [name for name, _geometry_type in pyogrio.list_layers(path)]
    else:
        import fiona

        return fiona.listlayers(path)


def write_features(features: geopandas.GeoDataFrame, path, layer=None):
    """Write features to a vector file or GeoParquet, depending on file extension

    Paths ending ".parquet" or ".geoparquet" are written with
    `geopandas.GeoDataFrame.to_parquet`, anything else is passed to
    `geopandas.GeoDataFrame.to_file` (with `layer` if provided, for formats
    such as GeoPackage which support multiple layers).
    """
    if Path(path).suffix in (".parquet", ".geoparquet"):
        if layer is not None:
            raise ValueError(
                f"Cannot write layer {layer!r} to {path}: Parquet output does not support layers"
            )
        logger.info("Writing %s", path)
        features.to_parquet(path)
    elif layer is not None:
        logger.info("Writing %s:%s", path, layer)
        features.to_file(path, layer=layer)
    else:
        logger.info("Writing %s", path)
        features.to_file(path)


def write_grid_to_raster(
    array: numpy.ndarray,
    output_path,
    transform,
    crs,
    *,
    nodata=None,
    dtype=None,
    driver: str = "GTiff",
    compress: str = "lzw",
    **profile_kwargs,
):
    """Write a 2D NumPy array to a single-band raster using rasterio."""
    if array.ndim != 2:
        raise ValueError("Only 2D arrays can be written to raster output")

    height, width = array.shape
    target_dtype = numpy.dtype(dtype or array.dtype)
    profile = {
        "driver": driver,
        "height": height,
        "width": width,
        "count": 1,
        "dtype": target_dtype,
        "transform": transform,
        "crs": crs,
    }

    if nodata is not None:
        profile["nodata"] = nodata
    if compress:
        profile["compress"] = compress
    profile.update(profile_kwargs)

    with rasterio.open(output_path, "w", **profile) as dataset:
        dataset.write(array.astype(target_dtype, copy=False), 1)


def _get_spatial_dims(data_array: "xarray.DataArray") -> tuple[str, str]:
    x_dim = data_array.rio.x_dim
    y_dim = data_array.rio.y_dim
    if not x_dim or not y_dim:
        raise ValueError("DataArray lacks named spatial dimensions for x/y.")
    return y_dim, x_dim


def _select_dataarray_band(
    data_array: "xarray.DataArray", band_number: int
) -> "xarray.DataArray":
    if band_number < 1:
        raise ValueError(f"band_number must be >= 1, got {band_number}")
    spatial_dims = set(_get_spatial_dims(data_array))
    non_spatial_dims = [dim for dim in data_array.dims if dim not in spatial_dims]

    if not non_spatial_dims:
        if band_number != 1:
            raise ValueError("Single-band DataArray only supports band_number=1.")
        return data_array

    if len(non_spatial_dims) > 1:
        raise ValueError(
            "DataArray has multiple non-spatial dimensions; select a single "
            "band or reduce the array before calling read_raster_band_data."
        )

    band_dim = non_spatial_dims[0]
    coord = data_array.coords.get(band_dim)
    try:
        if coord is not None and band_number in coord.values:
            selected = data_array.sel({band_dim: band_number})
        else:
            selected = data_array.isel({band_dim: band_number - 1})
    except (IndexError, KeyError) as exc:
        raise ValueError(
            f"Band index {band_number} is out of range for dimension {band_dim}."
        ) from exc

    return selected.squeeze(drop=True)


def _read_dataarray_metadata(
    data_array: "xarray.DataArray",
) -> tuple[GridDefinition, tuple[int]]:
    spatial_dims = set(_get_spatial_dims(data_array))
    non_spatial_dims = [dim for dim in data_array.dims if dim not in spatial_dims]

    grid = GridDefinition.from_xarray(data_array)

    if not non_spatial_dims:
        band_numbers: tuple[int, ...] = (1,)
    elif len(non_spatial_dims) == 1:
        band_dim = non_spatial_dims[0]
        band_size = data_array.sizes[band_dim]
        if band_size < 1:
            raise ValueError(
                f"DataArray dimension '{band_dim}' has no elements to treat as bands."
            )
        band_numbers = tuple(range(1, band_size + 1))
    else:
        raise ValueError(
            "DataArray has multiple non-spatial dimensions; provide a single "
            "band DataArray when reading metadata."
        )

    return grid, band_numbers
