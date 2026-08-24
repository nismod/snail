import argparse
import logging
import os
import sys
from pathlib import Path

import pandas

from snail.intersection import GridDefinition
from snail.io import (
    read_features,
    read_layer_names,
    read_raster_metadata,
    write_features,
)
from snail.overlay import (
    overlay_raster,
    overlay_rasters,
    parse_bands,
    split_features,
)

# Module-level logger
logger = logging.getLogger(__name__)


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
        "--all-layers",
        action="store_true",
        help="Split all layers in the vector file, writing each to a layer of a GeoPackage output",
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
        help="Column name to use when attributing raster values, defaults to "
        "raster filename stem. Values from a multi-band raster are attributed "
        "in one column per band, named '{column}_band_{n}'",
    )
    parser_split.add_argument(
        "--lazy-rasters",
        action="store_true",
        help=("Read raster bands lazily with xarray/dask when attributing values."),
    )
    parser_split.add_argument(
        "-o",
        "--output",
        type=str,
        required=True,
        help="Output file (any geopandas-supported format, or GeoParquet if "
        "the extension is '.parquet' or '.geoparquet')",
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
    parser_process.add_argument(
        "--lazy-rasters",
        action="store_true",
        help="Read raster bands lazily with xarray/dask during processing.",
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

    logger.debug("Called with %s", args)

    # Call the subcommand function
    logger.info("Start.")
    try:
        args.func(args)
    except AttributeError:
        parser.print_help()
    logger.info("Done.")


def split(args):
    """snail split command"""
    if args.layer and args.all_layers:
        sys.exit("Error: Expected either --layer or --all-layers, not both")

    if args.raster:
        grid, _ = read_raster_metadata(args.raster)
    else:
        width = args.width
        height = args.height
        affine_transform = args.transform
        if width is None or height is None or affine_transform is None:
            sys.exit(
                "Error: Expected either a raster file or transform, width and height of splitting grid"
            )
        grid = None  # read CRS from features
    logger.info(f"Splitting {grid=}")

    if args.all_layers:
        if Path(args.output).suffix.lower() != ".gpkg":
            sys.exit(
                "Error: Expected a GeoPackage (.gpkg) output to hold multiple layers with --all-layers"
            )
        layers = read_layer_names(args.features)
    else:
        layers = [args.layer]

    for layer in layers:
        features = read_features(Path(args.features), layer)
        if grid is None:
            layer_grid = GridDefinition(
                features.crs, width, height, tuple(affine_transform)
            )
        else:
            layer_grid = grid

        if args.attribute and args.raster:
            splits = overlay_raster(
                features,
                args.raster,
                bands=args.band,
                column=args.column,
                experimental=args.experimental,
                lazy=args.lazy_rasters,
            )
        else:
            splits = split_features(features, layer_grid, args.experimental)

        if args.all_layers:
            write_features(splits, args.output, layer=layer)
        else:
            write_features(splits, args.output)


def process(args):
    """snail process command"""
    # data directory
    dirname = args.directory

    # read rasters table
    rasters = _read_csv_or_quit(args.rasters)

    # fix up path relative to dirname
    rasters.path = rasters.path.apply(_join_dirname, args=(dirname,))

    # parse "1,2,3" band indices to tuple if present
    if "bands" in rasters.columns:
        rasters.bands = rasters.bands.apply(parse_bands)

    # read features table
    vector_layers = _read_csv_or_quit(args.features)
    vector_layers.path = vector_layers.path.apply(_join_dirname, args=(dirname,))

    # expand any wildcard ("*") layer rows to a row per layer in the file
    vector_layers = _expand_layers(vector_layers)

    # fill in default output paths where not specified
    rasters_name = Path(args.rasters).stem
    vector_layers = _fill_output_paths(vector_layers, rasters_name)

    for vector_layer in vector_layers.itertuples():
        _process_layer(
            vector_layer,
            rasters,
            experimental=args.experimental,
            lazy=args.lazy_rasters,
        )


def _process_layer(
    vector_layer,
    rasters,
    *,
    experimental: bool = False,
    lazy: bool = False,
):
    vector_path = Path(vector_layer.path)
    layer = getattr(vector_layer, "layer", None)
    if _is_missing(layer):
        layer = None
    logger.info("Processing %s layer %s", vector_path.name, layer)

    features = read_features(vector_path, layer)
    logger.info("Features CRS %s", features.crs)

    with_data = overlay_rasters(features, rasters, experimental=experimental, lazy=lazy)
    write_features(with_data, vector_layer.output_path)


def _expand_layers(vector_layers: pandas.DataFrame) -> pandas.DataFrame:
    """Expand rows with a wildcard ("*") layer to one row per layer in the file"""
    if "layer" not in vector_layers.columns:
        return vector_layers
    rows = []
    for row in vector_layers.to_dict("records"):
        layer = row.get("layer")
        if isinstance(layer, str) and layer.strip() == "*":
            for layer_name in read_layer_names(row["path"]):
                expanded = dict(row)
                expanded["layer"] = layer_name
                # an explicit output path must vary by layer, insert the layer name
                if not _is_missing(row.get("output_path")):
                    expanded["output_path"] = _insert_layer_in_path(
                        row["output_path"], layer_name
                    )
                rows.append(expanded)
        else:
            rows.append(row)
    return pandas.DataFrame(rows)


def _insert_layer_in_path(path, layer: str) -> str:
    path = Path(path)
    return str(path.with_name(f"{path.stem}_{layer}{path.suffix}"))


def _fill_output_paths(
    vector_layers: pandas.DataFrame, rasters_name: str
) -> pandas.DataFrame:
    if "output_path" not in vector_layers.columns:
        vector_layers["output_path"] = None
    vector_layers["output_path"] = [
        (
            _default_output_path(row.get("path"), row.get("layer"), rasters_name)
            if _is_missing(row.get("output_path"))
            else row["output_path"]
        )
        for row in vector_layers.to_dict("records")
    ]
    return vector_layers


def _default_output_path(vector_path, layer, rasters_name: str) -> str:
    """Default output path, "{features_stem}_{layer}__{rasters_stem}.parquet"
    next to the input features file (layer part omitted if not specified)"""
    path = Path(vector_path)
    if _is_missing(layer):
        name = f"{path.stem}__{rasters_name}.parquet"
    else:
        name = f"{path.stem}_{layer}__{rasters_name}.parquet"
    return str(path.parent / name)


def _is_missing(value) -> bool:
    return value is None or (isinstance(value, float) and pandas.isna(value))


def _read_csv_or_quit(path) -> pandas.DataFrame:
    try:
        df = pandas.read_csv(path)
    except FileNotFoundError:
        logger.error("File not found: %s", path)
        sys.exit()
    return df


def _join_dirname(path, dirname=False):
    if dirname:
        return os.path.join(dirname, path)
    return path
