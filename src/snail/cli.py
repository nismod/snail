import argparse
from os.path import splitext

from shapely.geometry import LineString, MultiLineString, Polygon
import geopandas as gpd
import rasterio
from pandas import read_csv
from igraph import Graph

from snail.multi_intersections import (
    split_linestrings,
    split_polygons,
    raster2split,
)
from snail.routing import shortest_paths


def parse_arguments(arguments):
    parser = argparse.ArgumentParser(description="the parser")
    parser.add_argument(
        "-r",
        "--raster",
        type=str,
        help="The path to the raster data file",
        nargs="+",
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
        "--band",
        type=int,
        help="Indices of raster band to be read",
        required=False,
    )
    parser.add_argument(
        "--extremities",
        type=str,
        help="Path to csv file for shortest paths extremities",
        required=False,
    )

    args = parser.parse_args(arguments)
    return args


def snail_split(arguments=None):
    args = parse_arguments(arguments)

    raster_data = rasterio.open(args.raster)
    vector_data = gpd.read_file(args.vector)

    geom = vector_data.iloc[0].geometry
    if type(geom) is LineString:
        new_gdf = split_linestrings(vector_data, raster_data)
    elif type(geom) is Polygon:
        new_gdf = split_polygons(vector_data, raster_data)
    else:
        raise ValueError(
            f"Could not process vector data of type {type(geom)}, expected Polygon or LineString"
        )

    new_gdf.to_file(args.output)


def snail_raster2split(arguments=None):
    args = parse_arguments(arguments)

    if isinstance(args.raster, str):
        args.raster = list(args.raster)

    with rasterio.open(args.raster[0]) as dataset:
        raster_width = dataset.width
        raster_height = dataset.height
        raster_transform = list(dataset.transform)
    # Make key: filename dict with filename (without ext) as key
    rasters = {splitext(k)[0]: v for (k, v) in zip(args.raster, args.raster)}
    vector_data = gpd.read_file(args.vector)

    new_gdf = raster2split(
        vector_data,
        rasters,
        width=raster_width,
        height=raster_height,
        transform=raster_transform,
        band_number=args.band,
        inplace=True,
    )
    new_gdf.to_file(args.output)


def snail_shortest_paths(arguments=None):
    args = parse_arguments(arguments)
    # Read vector data and create graph object
    vector_data = gpd.read_file(args.vector)
    # We assume the name of columns in geodataframe
    edges = vector_data.loc[:, ["from_node", "to_node", "length_km"]]
    graph = Graph.DataFrame(edges, directed=False)
    # "extremities" argument must be the path to a csv file giving
    # start and end points of shortest paths
    # Ex:
    # from,to
    # node_3,node_243
    # node_4576,node_943
    # ...
    extrm = read_csv(args.extremities)
    extremities, paths = shortest_paths(
        extrm.sources.tolist(),
        extrm.destinations.tolist(),
        graph,
        weight="length_km",
    )
    # Assemble output dataframe containing origin and end nodes,
    # length of shortest path, and geometry as a multi-LineString.
    # A path is a list of edge ids, each one
    # mapping back to the original vector data dataframe. We first
    # build a sub geodataframe that consists of just the geometries
    # making the path.
    lengths = []
    geoms = []
    for path in paths:
        sub_gdf = vector_data.iloc[path, :]
        lengths.append(sub_gdf.length_km.sum())
        geoms.append(MultiLineString([lstr for lstr in sub_gdf.geometry]))
    gpd.GeoDataFrame(
        {
            "from_node": extrm.sources.to_list(),
            "to_node": extrm.destinations.to_list(),
            "length_km": lengths,
            "geometry": geoms,
        }
    ).to_file(args.output)
