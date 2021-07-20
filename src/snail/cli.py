import argparse

import geopandas as gpd
import rasterio

from snail.intersections import split


def parse_arguments(arguments):
    parser = argparse.ArgumentParser(description="the parser")
    parser.add_argument(
        "-r",
        "--raster",
        type=str,
        help="The path to the raster data file",
        required=True,
    )
    parser.add_argument(
        "-v",
        "--vector",
        type=str,
        help="The path to the vector data file",
        required=True,
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        help="The path to the output vector data file",
        required=True,
    )

    args = parser.parse_args(arguments)
    return args


def main(arguments=None):
    args = parse_arguments(arguments)

    raster_data = rasterio.open(args.raster)
    vector_data = gpd.read_file(args.vector)

    for linestring in vector_data["geometry"].values:
        splits = split(
            linestring,
            raster_data.width,
            raster_data.height,
            list(raster_data.transform),
        )
    print("Working with:")
    print(f"  Raster dataset: {args.raster}")
    print(f"  Vector dataset: {args.vector}")
    print(f"  Output vector dataset: {args.output}")
