import importlib.util
import logging
from contextlib import contextmanager
from pathlib import Path
from typing import List, Tuple

import geopandas
import numpy
import pandas
import rasterio

from snail.intersection import GridDefinition, get_raster_values_for_splits


def band_column_name(key: str, band_number: int, number_of_bands: int) -> str:
    """Name the output column for a raster band.

    Single-band rasters attribute values in a column named by `key`,
    multi-band rasters in a column per band, named "{key}_band_{band_number}".
    """
    if number_of_bands == 1:
        return key
    return f"{key}_band_{band_number}"


def associate_raster_files(splits, rasters):
    """Read values from a list of raster files for a set of indexed split geometries

    Parameters
    ----------
    splits: pandas.DataFrame
        split geometries with raster indices in columns named "i_{grid_id}", "j_{grid_id}"
        for each grid_id in `rasters`

    rasters: pandas.DataFrame
        table of raster metadata with columns: key, grid_id, path, bands

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
    for raster, band_number, band_data in read_rasters(rasters):
        logging.info(
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


def read_rasters(rasters):
    for raster in rasters.itertuples():
        for band_number in raster.bands:
            yield raster, band_number, read_raster_band_data(raster.path, band_number)


@contextmanager
def _open_raster(raster):
    """Yield a rasterio dataset from either a path or an open dataset"""
    if hasattr(raster, "read") and hasattr(raster, "transform"):
        # looks like an open rasterio dataset - pass through, caller closes
        yield raster
    else:
        with rasterio.open(raster) as dataset:
            yield dataset


def read_raster_band_data(
    raster,
    band_number: int = 1,
) -> numpy.ndarray:
    """Read a single band of data from a raster path or open rasterio dataset"""
    with _open_raster(raster) as dataset:
        band_data: numpy.ndarray = dataset.read(band_number)
    return band_data


def extend_rasters_metadata(
    rasters: pandas.DataFrame,
) -> Tuple[pandas.DataFrame, List[GridDefinition]]:
    grids = []
    grid_ids = []
    raster_bands = []

    for raster in rasters.itertuples():
        logging.info("Reading metadata from raster %s", raster.path)
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


def read_raster_metadata(raster) -> Tuple[GridDefinition, Tuple[int]]:
    """Read grid definition and band indexes from a raster path or open
    rasterio dataset"""
    with _open_raster(raster) as dataset:
        bands = dataset.indexes
        grid = GridDefinition.from_rasterio_dataset(dataset)
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


def read_layer_names(path) -> List[str]:
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
        features.to_parquet(path)
    elif layer is not None:
        features.to_file(path, layer=layer)
    else:
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
