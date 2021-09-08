import random
import time
import argparse

import numpy as np
import pandana
from pandas import DataFrame
from igraph import Graph
import geopandas as gpd


def str_to_int(str_list):
    return [int(string[6:]) for string in str_list]


def get_input_list(source, dest):
    """Return the input lists for many-to-many shortest path calculation
    with pandana.Network.shortest_paths
    >>> list1, list2 = get_input_list(["a", "b", "c"], ["d", "e", "f", "g"])
    >>> list1
    ["a", "a", "a", "a", "b", "b", "b", "b", "c", "c", "c", "c"]
    >>> list2
    ["d", "e", "f", "g", "d", "e", "f", "g", "d", "e", "f", "g"]
    """
    nodes_a = []
    nodes_b = []
    for source_node in source:
        nodes_a.extend([source_node] * len(dest))
        nodes_b.extend(dest)
    return nodes_a, nodes_b


def shortest_paths_igraph(geodf, source_ensbl, dest_ensbl, path_type):
    edges = geodf.loc[:, ["from_node", "to_node", "length_km"]]
    g = Graph.DataFrame(edges, directed=False)
    dest = g.vs.select(name_in=dest_ensbl)

    shortest_paths = []
    for source_id in source_ensbl:
        start = g.vs.find(name=source_id)
        sp = g.get_shortest_paths(
            start, dest, weights="length_km", output=path_type
        )
        shortest_paths.append(sp)

    return shortest_paths


def shortest_paths_pandana(
    geodf, nodes_geodf, source_ensbl, dest_ensbl, path_type
):
    list_of_coords = [
        (point.x, point.y) for point in nodes_geodf.loc[:, "geometry"]
    ]
    nodes = DataFrame(list_of_coords, columns=["x", "y"])

    # Transform source and destination columns into lists of integers
    # (node ids)
    from_node = str_to_int(gdf["from_node"])
    to_node = str_to_int(gdf["to_node"])

    edges = DataFrame(
        {"from": from_node, "to": to_node, "weight": gdf["length_km"].values}
    )
    net = pandana.Network(
        nodes["x"], nodes["y"], edges["from"], edges["to"], edges[["weight"]]
    )
    nodes_a, nodes_b = get_input_list(
        str_to_int(source_ensbl), str_to_int(dest_ensbl)
    )
    return net.shortest_paths(nodes_a, nodes_b, imp_name="weight")


def reconstruct_epath(vpath, gdf):
    epath = []
    for fromnode, tonode in zip(vpath[:-1], vpath[1:]):
        from_id = "roadn_" + str(fromnode)
        to_id = "roadn_" + str(tonode)
        road = gdf.index[
            (gdf["from_node"] == from_id) & (gdf["to_node"] == to_id)
        ].tolist()
        if len(road) == 0:
            road = gdf.index[
                (gdf["from_node"] == to_id) & (gdf["to_node"] == from_id)
            ].tolist()
        epath.extend(road)
    return epath


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--vpath", action="store_true")
    parser.add_argument("--epath", action="store_true")
    parser.add_argument("--sizemin", type=int, default=10)
    parser.add_argument("--sizemax", type=int, default=1000)
    parser.add_argument("--npoints", type=int, default=5)
    parser.add_argument("--nreps", type=int, default=10)

    args = parser.parse_args()

    path_type = "vpath" if args.vpath else "epath"

    gdf = gpd.read_file("jamaica_roads.gpkg")
    nodes_gdf = gpd.read_file("jamaica_roads.gpkg", layer="nodes")

    igraph_times = np.zeros(args.nreps, args.npoints)
    pandana_times = np.zeros(args.nreps, args.npoints)
    pandana_reconstruct_times = np.zeros(args.nreps, args.npoints)
    for j, size in enumerate(
        np.logspace(args.sizemin, args.sizemax, args.npoints)
    ):
        # Sample source and destination ensembles
        for irep in range(args.nreps):
            source_ensbl = random.sample(list(nodes_gdf["node_id"]), size)
            dest_ensbl = random.sample(list(nodes_gdf["node_id"]), size)

            tic = time.time()
            path = shortest_paths_igraph(
                gdf, source_ensbl, dest_ensbl, path_type
            )
            toc = time.time()
            igraph_times[irep, j] = toc - tic

            tic.time()
            vpath = shortest_paths_pandana(
                gdf, nodes_gdf, source_ensbl, dest_ensbl
            )
            toc.time()
            pandana_times[irep, j] = toc - tic

            if path_type == "epath":
                tic.time()
                epath = reconstruct_epath(vpath, gdf)
                toc.time()
                pandana_reconstruct_times[irep, j] = toc - tic
