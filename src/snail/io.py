import logging
from typing import List, Tuple

import geopandas
import numpy
import pandas
import rasterio

from snail.intersection import Transform, associate_raster


def associate_raster_files(features, rasters):
    # to prevent a fragmented dataframe (and a memory explosion), add series to a dict
    # and then concat afterwards -- do not append to an existing dataframe
    raster_data: dict[str, pandas.Series] = {}

    # associate values
    for raster in rasters.itertuples():
        logging.info(
            "Associating values from raster %s transform %s",
            raster.key,
            raster.transform_id,
        )
        for band_number in raster.bands:
            raster_data[raster.key] = associate_raster_file(
                features,
                raster.path,
                f"i_{raster.transform_id}",
                f"j_{raster.transform_id}",
                band_number,
            )

    raster_data = pandas.DataFrame(raster_data)
    features = pandas.concat([features, raster_data], axis="columns")

    return features


def associate_raster_file(
    df: pandas.DataFrame,
    fname: str,
    index_i: str = "index_i",
    index_j: str = "index_j",
    band_number: int = 1,
) -> pandas.Series:
    band_data = read_band_data(fname, band_number)
    raster_values = associate_raster(df, band_data, index_i, index_j)
    return raster_values


def read_band_data(
    fname: str,
    band_number: int = 1,
) -> numpy.ndarray:
    with rasterio.open(fname) as dataset:
        band_data: numpy.ndarray = dataset.read(band_number)
    return band_data


def extend_rasters_metadata(
    rasters: pandas.DataFrame,
) -> Tuple[pandas.DataFrame, List[Transform]]:
    transforms = []
    transform_ids = []
    raster_bands = []

    for raster in rasters.itertuples():
        logging.info("Reading metadata from raster %s", raster.path)
        transform, bands = read_raster_metadata(raster.path)

        # add transform to list if not present
        if transform not in transforms:
            transforms.append(transform)

        # record raster/transform details
        transform_id = transforms.index(transform)
        transform_ids.append(transform_id)
        raster_bands.append(bands)

    rasters["transform_id"] = transform_ids
    if "bands" not in rasters.columns:
        rasters["bands"] = raster_bands

    return rasters, transforms


def read_raster_metadata(path) -> Tuple[Transform, Tuple[int]]:
    with rasterio.open(path) as dataset:
        crs = dataset.crs
        width = dataset.width
        height = dataset.height
        affine_transform = tuple(dataset.transform)[
            :6
        ]  # trim to 6 - we expect the first two rows of 3x3 matrix
        bands = dataset.indexes
    transform = Transform(crs, width, height, affine_transform)
    return transform, bands


def read_features(path, layer=None):
    if path.suffix in (".parquet", ".geoparquet"):
        features = geopandas.read_parquet(path)
    else:
        if layer:
            features = geopandas.read_file(path, layer=layer)
        else:
            features = geopandas.read_file(path)
    return features
