import argparse
import logging
import os
import sys
from pathlib import Path

import geopandas
import pandas

from snail.intersection import (
    Transform,
    apply_indices,
    prepare_linestrings,
    prepare_polygons,
    prepare_points,
    split_features_for_rasters,
    split_linestrings,
    split_polygons,
    split_points,
)
from snail.io import (
    associate_raster_file,
    associate_raster_files,
    read_transform,
    read_transforms,
)


def snail(args=None):
    """snail command"""
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
        "-c",
        "--column",
        type=str,
        required=False,
        help="Column name to use when attributing raster values, defaults to raster filename",
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

    if args.verbose > 0:
        logformat = "%(asctime)s %(levelname)s %(message)s"
    else:
        logformat = "%(message)s"

    logging.basicConfig(
        format=logformat,
        level=level,
    )

    logging.debug("Called with %s", args)

    # Call the subcommand function
    logging.info("Start.")
    args.func(args)
    logging.info("Done.")


def split(args):
    """snail split command"""
    if args.raster:
        transform = read_transform(args.raster)
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

    features = geopandas.read_file(args.features)
    features_crs = features.crs
    geom_type = _sample_geom_type(features)

    if "Point" in geom_type:
        prepared = prepare_points(features)
        splits = split_points(prepared)
    elif "LineString" in geom_type:
        prepared = prepare_linestrings(features)
        splits = split_linestrings(prepared, transform)
    elif "Polygon" in geom_type:
        prepared = prepare_polygons(features)
        splits = split_polygons(prepared, transform)
    else:
        raise ValueError("Could not process vector data of type %s", geom_type)

    splits = apply_indices(splits, transform)

    if args.attribute and args.raster:
        if args.column:
            key = args.key
        else:
            key = os.path.basename(args.raster)

        logging.info(
            f"Attributing raster values, output in column %s from %s",
            key,
            args.raster,
        )
        splits[key] = associate_raster_file(splits, args.raster)

    splits.set_crs(features_crs, inplace=True)
    splits.to_file(args.output)


def process(args):
    """snail process command"""
    # data directory
    dirname = args.directory

    # read transforms
    try:
        rasters = pandas.read_csv(args.rasters)
    except FileNotFoundError:
        logging.error("Rasters file not found: %s", args.rasters)
        sys.exit()

    rasters.path = rasters.path.apply(_join_dirname, args=(dirname,))
    if "key" not in rasters.columns:
        colnames = sorted(set(rasters.columns) ^ {"path"})
        rasters["key"] = rasters.apply(_format_key, args=(colnames,), axis=1)

    rasters, transforms = read_transforms(rasters)

    # read networks
    try:
        vector_layers = pandas.read_csv(args.features)
    except FileNotFoundError:
        logging.error("Features file not found: %s", args.features)
        sys.exit()

    vector_layers.path = vector_layers.path.apply(
        _join_dirname, args=(dirname,)
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
            if "layer" in vector_layers.columns:
                features = geopandas.read_file(
                    vector_path, layer=vector_layer.layer
                )
            else:
                features = geopandas.read_file(vector_path)

        geom_type = _sample_geom_type(features)
        logging.info("%s Features CRS %s", geom_type, features.crs)

        if "Point" in geom_type:
            prepared = prepare_points(features)
            split = split_features_for_rasters(
                prepared, transforms, split_points
            )
            with_data = associate_raster_files(split, rasters)
        elif "LineString" in geom_type:
            prepared = prepare_linestrings(features)
            split = split_features_for_rasters(
                prepared, transforms, split_linestrings
            )
            with_data = associate_raster_files(split, rasters)
        elif "Polygon" in geom_type:
            prepared = prepare_polygons(features)
            split = split_features_for_rasters(
                prepared, transforms, split_polygons
            )
            with_data = associate_raster_files(split, rasters)
        else:
            raise ValueError(
                f"Could not process vector data of type {geom_type}"
            )

        with_data.to_parquet(vector_layer.output_path)


def _sample_geom_type(df: geopandas.GeoDataFrame) -> str:
    return df.iloc[0].geometry.geom_type


def _join_dirname(path, dirname=False):
    if dirname:
        return os.path.join(dirname, path)
    return path


def _format_key(row, colnames):
    parts = []
    for c in colnames:
        parts.append(f"{c}:{row.loc[c]}")
    key = "|".join(parts)
    return key
