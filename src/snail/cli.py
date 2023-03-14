import argparse
import logging
import os
import sys
from pathlib import Path

import geopandas
import numpy
import pandas
import rasterio
import xarray
from shapely.ops import linemerge

from snail.intersection import (
    Transform,
    apply_indices,
    associate_raster,
    split_linestrings,
    split_polygons,
)


def snail(args=None):
    parser = argparse.ArgumentParser(prog="snail")
    parser.add_argument("--verbose", "-v", action="count", default=0)
    subparsers = parser.add_subparsers(help="Run a command")

    parser_split = subparsers.add_parser(
        "split", help="Split vector features on a regular grid"
    )
    parser_split.add_argument(
        "-f",
        "--features",
        type=str,
        required=True,
        help="File with vector features to split",
    )
    parser_split.add_argument(
        "-l",
        "--layer",
        type=str,
        required=False,
        help="Layer in file with vector features to split",
    )
    parser_split.add_argument(
        "-r",
        "--raster",
        type=str,
        required=False,
        help="Raster file/s to use as definition of splitting grid",
    )
    parser_split.add_argument(
        "-t",
        "--transform",
        type=float,
        required=False,
        nargs=6,
        help="""Affine transform of splitting grid.

        For example, for a north-up image:
            {top-left x coordinate} {cell width} {zero} {top-left y coordinate} {cell height} {zero}
        """,
    )
    parser_split.add_argument(
        "-w",
        "--width",
        type=int,
        required=False,
        help="Width of splitting grid (number of columns)",
    )
    parser_split.add_argument(
        "-g",
        "--height",
        type=int,
        required=False,
        help="Height of splitting grid (number of rows)",
    )
    parser_split.add_argument(
        "-a",
        "--attribute",
        action="store_true",
        help="Attribute raster values to split output",
    )
    parser_split.add_argument(
        "-o",
        "--output",
        type=str,
        required=True,
        help="Output file",
    )
    parser_split.set_defaults(func=split)

    parser_process = subparsers.add_parser(
        "process", help="Split vectors and attribute raster values"
    )
    parser_process.add_argument(
        "-d",
        "--directory",
        type=str,
        help="Path to data directory for vector and raster paths",
    )
    parser_process.add_argument(
        "-fs",
        "--features",
        type=str,
        required=True,
        help="CSV file with vector layers",
    )
    parser_process.add_argument(
        "-rs",
        "--rasters",
        type=str,
        required=True,
        help="CSV file with raster layers",
    )
    parser_process.set_defaults(func=process)

    args = parser.parse_args(args)

    # Enable logging
    if args.verbose > 2:
        level = logging.DEBUG
    elif args.verbose > 1:
        level = logging.INFO
    elif args.verbose > 0:
        level = logging.WARNING
    else:
        level = logging.ERROR

    logging.basicConfig(format="%(asctime)s %(message)s", level=level)

    logging.info(args)

    # Call the subcommand function
    logging.info("Start.")
    args.func(args)
    logging.info("Done.")


def split(args):
    # raster variable name, might get set if reading a NetCDF
    varname = None

    if args.raster:
        _, ext = os.path.splitext(args.raster)
        if ext == "nc":
            ds = xarray.open_dataset(args.raster)
            affine_transform = list(ds.rio.transform())
            if args.raster_var:
                varname = args.raster_var
            else:
                varname = next(iter(list(ds.keys())))
            da = ds[varname]
            # TODO fix gross assumption about dimensions
            # time, lat, lon
            _, height, width = da.shape
        else:
            raster = rasterio.open(args.raster)
            crs = raster.crs
            width = raster.width
            height = raster.height
            affine_transform = list(raster.transform)
    else:
        crs = None
        width = args.width
        height = args.height
        affine_transform = args.transform
        if width is None or height is None or affine_transform is None:
            sys.exit(
                "Error: Expected either a raster file or transform, width and height of splitting grid"
            )
    transform = Transform(crs, width, height, affine_transform)
    logging.info(f"Splitting grid {transform=}")

    try:
        features = geopandas.read_file(args.features)
        geom_type = sample_geom_type(features)

        if "Point" in geom_type:
            splits = explode_multi(features)
        elif "LineString" in geom_type:
            splits = split_linestrings(features, transform)
        elif "Polygon" in geom_type:
            splits = split_polygons(features, transform)
        else:
            raise ValueError(
                f"Could not process vector data of type {geom_type}"
            )

        splits = apply_indices(splits, transform)
        if args.attribute and args.raster:
            splits[os.path.basename(args.raster)] = associate_raster_file(
                splits, args.raster, variable_name=varname
            )

        splits.to_file(args.output)

    finally:
        if args.raster:
            raster.close()


def join_dirname(path, dirname=False):
    if dirname:
        return os.path.join(dirname, path)
    return path


def format_key(row, colnames):
    parts = []
    for c in colnames:
        parts.append(f"{c}:{row.loc[c]}")
    key = "|".join(parts)
    return key


def process(args):
    # data directory
    dirname = args.directory

    # read transforms
    rasters = pandas.read_csv(args.rasters)
    rasters.path = rasters.path.apply(join_dirname, args=(dirname,))
    if "key" not in rasters.columns:
        colnames = sorted(set(rasters.columns) ^ {"path"})
        rasters["key"] = rasters.apply(format_key, args=(colnames,), axis=1)

    rasters, transforms = read_transforms(rasters)

    # read networks
    vector_layers = pandas.read_csv(args.features)
    vector_layers.path = vector_layers.path.apply(
        join_dirname, args=(dirname,)
    )
    if "output_path" not in vector_layers.columns:
        vector_layers["output_path"] = vector_layers.path.apply(
            lambda p: f"{p}.processed.parquet"
        )

    for vector_layer in vector_layers.itertuples():
        vector_path = Path(vector_layer.path)
        logging.info("Processing %s", vector_path.name)

        if vector_path.suffix in (".parquet", ".geoparquet"):
            features = geopandas.read_parquet(vector_path)
        else:
            features = geopandas.read_file(
                vector_path, layer=vector_layer.layer
            )

        geom_type = sample_geom_type(features)
        logging.info(f"{geom_type} Features CRS {features.crs}")

        if "Point" in geom_type:
            processed = process_nodes(features, transforms, rasters)
        elif "LineString" in geom_type:
            processed = process_edges(features, transforms, rasters)
        elif "Polygon" in geom_type:
            processed = process_areas(features, transforms, rasters)
        else:
            raise ValueError(
                f"Could not process vector data of type {geom_type}"
            )

        processed.to_parquet(vector_layer.output_path)


def process_nodes(nodes, transforms, rasters):
    # handle multipoints
    # nodes = explode_multi(nodes)

    return process_features(nodes, transforms, rasters, lambda df, _: df)


def process_edges(edges, transforms, rasters):
    # handle multilinestrings
    edges.geometry = edges.geometry.apply(try_merge)
    edges = explode_multi(edges)

    return process_features(edges, transforms, rasters, split_linestrings)


def process_areas(areas, transforms, rasters):
    # handle multipolygons
    areas = explode_multi(areas)

    return process_features(areas, transforms, rasters, split_polygons)


def process_features(features, transforms, rasters, split_func):
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
        raster_data[raster.key] = associate_raster_file(
            features,
            raster.path,
            f"i_{raster.transform_id}",
            f"j_{raster.transform_id}",
        )

        # if allowing out-of-range splits, assign nodata value
        # TODO derive NODATA value
        # TODO flag to allow out-of-range
        # TODO push some of this into core library code?
        nodata_value = numpy.nan
        nodata_mask = features[
            (f"i_{raster.transform_id}" < 0) | (f"j_{raster.transform_id}" < 0)
        ]
        raster_data.loc[nodata_mask, raster.key] = nodata_value

    raster_data = pandas.DataFrame(raster_data)
    features = pandas.concat([features, raster_data], axis="columns")

    return features


def associate_raster_file(
    df: pandas.DataFrame,
    fname: str,
    index_i: str = "index_i",
    index_j: str = "index_j",
    band_number: int = 1,
    variable_name: str = None,
) -> pandas.Series:
    band_data = read_band_data(fname, band_number, variable_name)
    raster_values = associate_raster(df, band_data, index_i, index_j)
    return raster_values


def read_band_data(
    fname: str,
    band_number: int = 1,
    variable_name: str = None,
) -> numpy.ndarray:
    _, ext = os.path.splitext(fname)
    if ext == "nc":
        ds = xarray.open_dataset(fname)
        da = ds[variable_name]
        # TODO fix gross assumption about dimensions
        # (time, lat, lon), so pick first time slice
        band_data = da.data[0]
    else:
        with rasterio.open(fname) as dataset:
            band_data: numpy.ndarray = dataset.read(band_number)
    return band_data


def read_transforms(rasters):
    transforms = []
    transform_ids = []

    for raster in rasters.itertuples():
        logging.info("Reading metadata from raster %s", raster.path)
        with rasterio.open(raster.path) as dataset:
            crs = dataset.crs
            width = dataset.width
            height = dataset.height
            transform = Transform(crs, width, height, tuple(dataset.transform))

        # add transform to list if not present
        if transform not in transforms:
            transforms.append(transform)

        # record raster/transform details
        transform_id = transforms.index(transform)
        transform_ids.append(transform_id)

    rasters["transform_id"] = transform_ids
    return rasters, transforms


def sample_geom_type(df: geopandas.GeoDataFrame) -> str:
    return df.iloc[0].geometry.geom_type


def try_merge(geom):
    if geom.geom_type == "MultiLineString":
        geom = linemerge(geom)
    return geom


def explode_multi(df):
    items = []
    geoms = []
    for item in df.itertuples(index=False):
        if item.geometry.geom_type in (
            "MultiPoint",
            "MultiLineString",
            "MultiPolygon",
        ):
            for part in item.geometry:
                items.append(item._asdict())
                geoms.append(part)
        else:
            items.append(item._asdict())
            geoms.append(item.geometry)

    df = geopandas.GeoDataFrame(items, crs=df.crs, geometry=geoms)
    return df
