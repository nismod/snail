"""Generate darthmouth events
"""
import os
import sys
import configparser
import csv
import pandas as pd
import geopandas as gpd
import subprocess
import numpy as np
import igraph as ig
from shapely.geometry import Point
from geopy.distance import vincenty
from boltons.iterutils import pairwise
from snkit import Network
from snkit.network import *
from tqdm import tqdm


def line_length(line, ellipsoid="WGS-84"):
    """Length of a line in meters, given in geographic coordinates.

    Adapted from https://gis.stackexchange.com/questions/4022/looking-for-a-pythonic-way-to-calculate-the-length-of-a-wkt-linestring#answer-115285

    Args:
        line: a shapely LineString object with WGS-84 coordinates.

        ellipsoid: string name of an ellipsoid that `geopy` understands (see http://geopy.readthedocs.io/en/latest/#module-geopy.distance).

    Returns:
        Length of line in kilometers.
    """
    if line.geometryType() == "MultiLineString":
        return sum(line_length(segment) for segment in line)

    return sum(
        vincenty(
            tuple(reversed(a)), tuple(reversed(b)), ellipsoid=ellipsoid
        ).kilometers
        for a, b in pairwise(line.coords)
    )


def create_network_from_edges(
    edge_file, node_edge_prefix, projection, distance=20
):
    edges = gpd.read_file(edge_file).to_crs(epsg=projection)
    edges.columns = map(str.lower, edges.columns)
    if "id" in edges.columns.values.tolist():
        edges.rename(columns={"id": "e_id"}, inplace=True)

    # create network
    network = Network(edges=edges)
    print("* Done with network creation")

    # add nodes at endpoints
    network = add_endpoints(network)
    print("* Done with adding endpoints")

    network = link_nodes_to_edges_within(network, distance)
    print("* Done with linking nodes to edges")
    # add topology
    network = add_topology(
        add_ids(
            network,
            edge_prefix="{}e".format(node_edge_prefix),
            node_prefix="{}n".format(node_edge_prefix),
        )
    )
    print("* Done with network topology")

    network.edges.rename(
        columns={"from_id": "from_node", "to_id": "to_node", "id": "edge_id"},
        inplace=True,
    )
    network.nodes.rename(columns={"id": "node_id"}, inplace=True)

    return network


def get_nearest_node(x, sindex_input_nodes, input_nodes, id_column):
    """Get nearest node in a dataframe

    Parameters
    ----------
    x
        row of dataframe
    sindex_nodes
        spatial index of dataframe of nodes in the network
    nodes
        dataframe of nodes in the network
    id_column
        name of column of id of closest node

    Returns
    -------
    Nearest node to geometry of row
    """
    return input_nodes.loc[list(sindex_input_nodes.nearest(x.bounds[:2]))][
        id_column
    ].values[0]


def extract_gdf_values_containing_nodes(
    x, sindex_input_gdf, input_gdf, column_name
):
    a = input_gdf.loc[list(input_gdf.geometry.contains(x.geometry))]
    if len(a.index) > 0:
        return a[column_name].values[0]
    else:
        return get_nearest_node(
            x.geometry, sindex_input_gdf, input_gdf, column_name
        )


def convert_tif_to_csv_gdf(
    filepath,
    filename,
    point_id_column,
    value_column,
    projection={"init": "epsg:4326"},
):
    outCSVName = os.path.join(
        filepath, "{}.csv".format(filename.split(".tif")[0])
    )
    # if filename.endswith('.tif'):
    #     outCSVName = os.path.join(filepath, '{}.csv'.format(filename[:-4]))
    # elif filename.endswith('.tiff'):
    #     outCSVName = os.path.join(filepath, '{}.csv'.format(filename[:-5]))

    subprocess.run(
        ["gdal2xyz.py", "-csv", os.path.join(filepath, filename), outCSVName]
    )

    # Load points and convert to geodataframe with coordinates
    load_points = pd.read_csv(
        outCSVName, header=None, names=["x", "y", value_column], index_col=None
    ).fillna(0)
    load_points = load_points[load_points[value_column] > 0]
    load_points[point_id_column] = load_points.index.values.tolist()

    geometry = [Point(xy) for xy in zip(load_points.x, load_points.y)]
    # load_points = load_points.drop(['x', 'y'], axis=1)
    gdf = gpd.GeoDataFrame(load_points, crs=projection, geometry=geometry)

    del load_points

    return gdf


def network_od_path_estimations(
    graph,
    source,
    target,
    id_column,
    distance_criteria,
    time_criteria,
    cost_criteria,
):
    """Estimate the paths, distances, times, and costs for given OD pair

    Parameters
    ---------
    graph
        igraph network structure
    source
        String/Float/Integer name of Origin node ID
    source
        String/Float/Integer name of Destination node ID
    id_column : str
        name of edge ID column in network
    distance_criteria : str
        name of distance criteria to be used
    time_criteria : str
        name of time criteria to be used
    cost_criteria : str
        name of generalised cost criteria to be used

    Returns
    -------
    edge_path_list : list[list]
        nested lists of Strings/Floats/Integers of edge ID's in routes
    path_dist_list : list[float]
        estimated distances of routes
    path_time_list : list[float]
        estimated times of routes
    path_gcost_list : list[float]
        estimated generalised costs of routes

    """
    paths = graph.get_shortest_paths(
        source, target, weights=cost_criteria, output="epath"
    )

    edge_path_list = []
    path_dist_list = []
    path_time_list = []
    path_gcost_list = []

    for path in paths:
        edge_path = []
        path_dist = 0
        path_time = 0
        path_gcost = 0
        if path:
            for n in path:
                edge_path.append(graph.es[n][id_column])
                path_dist += graph.es[n][distance_criteria]
                path_time += graph.es[n][time_criteria]
                path_gcost += graph.es[n][cost_criteria]

        edge_path_list.append(edge_path)
        path_dist_list.append(path_dist)
        path_time_list.append(path_time)
        path_gcost_list.append(path_gcost)

    return edge_path_list, path_dist_list, path_time_list, path_gcost_list


def network_od_paths_assembly(
    points_dataframe,
    graph,
    graph_id,
    origin_column,
    destination_column,
    distance_criteria,
    time_criteria,
    cost_criteria,
):
    """Assemble estimates of OD paths, distances, times, costs and tonnages on networks

    Parameters
    ----------
    points_dataframe : pandas.DataFrame
        OD nodes and their tonnages
    graph
        igraph network structure
    origin_column : str
        name of origin column in points dataframe
    destination_column : str
        name of destination column in points dataframe
    distance_criteria : str
        name of distance column in igraph network
    time_criteria : str
        name of time column in igraph network
    cost_criteria : str
        name of generalised cost column in igraph network

    Returns
    -------
    save_paths_df : pandas.DataFrame
        - origin - String node ID of Origin
        - destination - String node ID of Destination
        - edge_path - List of string of edge ID's for paths
        - distance - Float values of estimated distance for paths with minimum generalised cost flows
        - time - Float values of estimated time for paths with minimum generalised cost flows
        - gcost - Float values of estimated generalised cost for paths with minimum generalised cost flows
    """
    save_paths = []
    points_dataframe = points_dataframe.set_index(origin_column)
    origins = list(set(points_dataframe.index.values.tolist()))
    for origin in origins:
        try:
            destinations = points_dataframe.loc[
                [origin], destination_column
            ].values.tolist()

            (
                get_path,
                get_dist,
                get_time,
                get_gcost,
            ) = network_od_path_estimations(
                graph,
                origin,
                destinations,
                graph_id,
                distance_criteria,
                time_criteria,
                cost_criteria,
            )

            save_paths += list(
                zip(
                    [origin] * len(destinations),
                    destinations,
                    get_path,
                    get_dist,
                    get_time,
                    get_gcost,
                )
            )
            print("done with {0}".format(origin))
        except:
            print("* no path between {}-{}".format(origin, destinations))

    if cost_criteria == time_criteria or cost_criteria == distance_criteria:
        cost_criteria = "gcost"

    cols = [
        origin_column,
        destination_column,
        "edge_path",
        distance_criteria,
        time_criteria,
        cost_criteria,
    ]
    save_paths_df = pd.DataFrame(save_paths, columns=cols)
    del save_paths

    return save_paths_df
