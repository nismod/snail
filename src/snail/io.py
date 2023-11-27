import importlib.util
import logging
from typing import List, Tuple

import geopandas
import numpy
import pandas
import rasterio

from snail.intersection import GridDefinition, get_raster_values_for_splits


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
        split geometries with raster data values at indexed locations
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
        raster_data[raster.key] = get_raster_values_for_splits(
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
            yield raster, band_number, read_raster_band_data(
                raster.path, band_number
            )


def read_raster_band_data(
    fname: str,
    band_number: int = 1,
) -> numpy.ndarray:
    with rasterio.open(fname) as dataset:
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
    if "bands" not in rasters.columns:
        rasters["bands"] = raster_bands

    return rasters, grids


def read_raster_metadata(path) -> Tuple[GridDefinition, Tuple[int]]:
    with rasterio.open(path) as dataset:
        bands = dataset.indexes
        grid = GridDefinition.from_rasterio_dataset(dataset)
    return grid, bands


def read_features(path, layer=None):
    if path.suffix in (".parquet", ".geoparquet"):
        features = geopandas.read_parquet(path)
    else:
        if importlib.util.find_spec("pyogrio"):
            engine = "pyogrio"
        else:
            engine = "fiona"

        if layer:
            features = geopandas.read_file(path, layer=layer, engine=engine)
        else:
            features = geopandas.read_file(path, engine=engine)
    return features
