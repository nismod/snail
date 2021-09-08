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
    geodf, nodes_geodf, source_ensbl, dest_ensbl
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


parser = argparse.ArgumentParser(description="")
parser.add_argument("--vpath", action="store_true")
parser.add_argument("--epath", action="store_true")
parser.add_argument("--sizes", nargs="+")
parser.add_argument("--nreps", type=int, default=2)

args = parser.parse_args(["--vpath", "--sizes", "2", "5", "10"])

path_type = "vpath" if args.vpath else "epath"

gdf = gpd.read_file("jamaica_roads.gpkg")
nodes_gdf = gpd.read_file("jamaica_roads.gpkg", layer="nodes")

igraph_times = np.zeros((args.nreps, len(args.sizes)))
pandana_times = np.zeros((args.nreps, len(args.sizes)))
pandana_reconstruct_times = np.zeros((args.nreps, args.npoints))
for j, size in enumerate(args.sizes):
    print(f"Size {j+1} of {len(args.sizes)}")
    # Sample source and destination ensembles
    for irep in range(args.nreps):
        print(f"    Rep {irep + 1} of {args.nreps}")
        # Some nodes in the "nodes" layer don't seem to have roads
        # start of end at them. So picking from from_node and to_node
        # columns.
        source_ensbl = random.sample(list(gdf["from_node"]), int(size))
        dest_ensbl = random.sample(list(gdf["to_node"]), int(size))

        print("        Timing igraph")
        tic = time.time()
        path = shortest_paths_igraph(
            gdf, source_ensbl, dest_ensbl, path_type
        )
        toc = time.time()
        igraph_times[irep, j] = toc - tic

        print("        Timing pandana")
        tic = time.time()
        vpaths = shortest_paths_pandana(
            gdf, nodes_gdf, source_ensbl, dest_ensbl
        )
        toc = time.time()
        pandana_times[irep, j] = toc - tic

        if path_type == "epath":
            print("        Timing pandana reconstrction step")
            tic = time.time()
            epaths = [reconstruct_epath(vpath, gdf) for vpath in vpaths]
            toc = time.time()
            pandana_reconstruct_times[irep, j] = toc - tic

igraph_min_times = np.min(igraph_times, axis=0)
pandana_min_times = np.min(pandana_times, axis=0)
pandana_reconstruct_min_times = np.min(pandana_reconstruct_times, axis=0)
