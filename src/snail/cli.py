import argparse

import geopandas as gpd
import rasterio

from snail.snail_intersections import split, raster2split


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
    parser.add_argument(
        "--bands",
        type=int,
        help="Indices of raster bands to be read",
        nargs="+",
        required=False,
    )

    args = parser.parse_args(arguments)
    return args


def main(arguments=None):
    args = parse_arguments(arguments)

    raster_data = rasterio.open(args.raster)
    vector_data = gpd.read_file(args.vector)

    new_gdf = split(vector_data, raster_data)
    new_gdf.to_file(args.output)


def snail_raster2split(arguments=None):
    args = parse_arguments(arguments)

    raster_data = rasterio.open(args.raster)
    vector_data = gpd.read_file(args.vector)

    new_gdf = raster2split(vector_data, raster_data, args.bands)
    new_gdf.to_file(args.output)
