import argparse
import logging
import os
import sys
from pathlib import Path

import geopandas
import pandas

from snail.intersection import (
    GridDefinition,
    apply_indices,
    get_raster_values_for_splits,
    prepare_linestrings,
    prepare_polygons,
    prepare_points,
    split_features_for_rasters,
    split_linestrings,
    split_polygons,
    split_points,
    split_polygons_experimental,
)
from snail.io import (
    read_raster_band_data,
    associate_raster_files,
    read_features,
    read_raster_metadata,
    extend_rasters_metadata,
)


def snail(args=None):
    """snail command"""
    parser = argparse.ArgumentParser(prog="snail")
    parser.add_argument("--verbose", "-v", action="count", default=0)
    parser.add_argument("-x", "--experimental", action="store_true")
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

        For example:
            {cell width} {zero} {top-left x coordinate} {zero} {cell height} {top-left y coordinate}
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
        "-b",
        "--band",
        type=int,
        required=False,
        nargs="+",
        help="Raster file band/s to use if attributing values",
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
        grid, all_bands = read_raster_metadata(args.raster)
    else:
        crs = None
        width = args.width
        height = args.height
        affine_transform = args.transform
        if width is None or height is None or affine_transform is None:
            sys.exit(
                "Error: Expected either a raster file or transform, width and height of splitting grid"
            )
        grid = GridDefinition(crs, width, height, affine_transform)
    logging.info(f"Splitting {grid=}")

    features = read_features(Path(args.features))
    features_crs = features.crs
    geom_type = _sample_geom_type(features)

    if "Point" in geom_type:
        logging.info("Preparing points")
        prepared = prepare_points(features)
        logging.info("Splitting points")
        splits = split_features_for_rasters(prepared, [grid], split_points)
    elif "LineString" in geom_type:
        logging.info("Preparing linestrings")
        prepared = prepare_linestrings(features)
        logging.info("Splitting linestrings")
        splits = split_features_for_rasters(
            prepared, [grid], split_linestrings
        )
    elif "Polygon" in geom_type:
        logging.info("Preparing polygons")
        prepared = prepare_polygons(features)
        if args.experimental:
            logging.info("Splitting polygons (experimental)")
            splits = split_features_for_rasters(
                prepared, [grid], split_polygons_experimental
            )
        else:
            logging.info("Splitting polygons")
            splits = split_features_for_rasters(
                prepared, [grid], split_polygons
            )
    else:
        raise ValueError("Could not process vector data of type %s", geom_type)

    logging.info("Applying indices")
    splits = apply_indices(splits, grid)

    if args.attribute and args.raster:
        if args.band:
            bands = args.band
        else:
            bands = all_bands

        if args.column:
            key = args.column
        else:
            key = os.path.basename(args.raster)

        for band_index in bands:
            if len(bands) == 1:
                band_key = key
            else:
                band_key = f"{key}_{band_index}"

            logging.info(
                "Attributing raster values, output in column %s from %s band %s",
                band_key,
                args.raster,
                band_index,
            )
            band_data = read_raster_band_data(
                args.raster, band_number=int(band_index)
            )
            splits[key] = get_raster_values_for_splits(splits, band_data)

    splits.set_crs(features_crs, inplace=True)
    splits.to_file(args.output)


def process(args):
    """snail process command"""
    # data directory
    dirname = args.directory

    # read rasters and transforms
    rasters = _read_csv_or_quit(args.rasters)

    # fix up path relative to dirname
    rasters.path = rasters.path.apply(_join_dirname, args=(dirname,))

    # generate "key" from metadata columns
    if "key" not in rasters.columns:
        colnames = sorted(set(rasters.columns) - {"path", "bands"})
        rasters["key"] = rasters.apply(_format_key, args=(colnames,), axis=1)

    # parse "1,2,3" band indices to tuple if present
    if "bands" in rasters.columns:
        rasters.bands = rasters.bands.apply(
            lambda c: tuple(int(b) for b in str(c).split(","))
        )

    rasters, transforms = extend_rasters_metadata(rasters)

    # read networks
    vector_layers = _read_csv_or_quit(args.features)

    vector_layers.path = vector_layers.path.apply(
        _join_dirname, args=(dirname,)
    )
    if "output_path" not in vector_layers.columns:
        vector_layers["output_path"] = vector_layers.path.apply(
            lambda p: f"{p}.processed.parquet"
        )

    for vector_layer in vector_layers.itertuples():
        _process_layer(vector_layer, transforms, rasters, args.experimental)


def _process_layer(vector_layer, transforms, rasters, experimental=False):
    vector_path = Path(vector_layer.path)
    layer = getattr(vector_layer, "layer", None)
    logging.info("Processing %s", vector_path.name)

    features = read_features(vector_path, layer)
    geom_type = _sample_geom_type(features)
    logging.info("%s Features CRS %s", geom_type, features.crs)

    if "Point" in geom_type:
        prepared = prepare_points(features)
        split = split_features_for_rasters(prepared, transforms, split_points)
        with_data = associate_raster_files(split, rasters)
    elif "LineString" in geom_type:
        prepared = prepare_linestrings(features)
        split = split_features_for_rasters(
            prepared, transforms, split_linestrings
        )
        with_data = associate_raster_files(split, rasters)
    elif "Polygon" in geom_type:
        prepared = prepare_polygons(features)
        if experimental:
            logging.info("Split polygons (experimental)")
            split = split_features_for_rasters(
                prepared, transforms, split_polygons_experimental
            )
        else:
            split = split_features_for_rasters(
                prepared, transforms, split_polygons
            )
        with_data = associate_raster_files(split, rasters)
    else:
        raise ValueError(f"Could not process vector data of type {geom_type}")

    with_data.to_parquet(vector_layer.output_path)


def _read_csv_or_quit(path) -> pandas.DataFrame:
    try:
        df = pandas.read_csv(path)
    except FileNotFoundError:
        logging.error("File not found: %s", path)
        sys.exit()
    return df


def _sample_geom_type(df: geopandas.GeoDataFrame) -> str:
    return df.iloc[0].geometry.geom_type


def _join_dirname(path, dirname=False):
    if dirname:
        return os.path.join(dirname, path)
    return path


def _format_key(row, colnames):
    if colnames:
        # stitch together from metadata columns
        parts = []
        for c in colnames:
            parts.append(f"{c}:{row.loc[c]}")
        key = "|".join(parts)
    else:
        # fall back to path as key
        key = row.loc["path"]
    return key
