import random
import time
import argparse

import numpy as np
import pandana
from pandas import DataFrame
from igraph import Graph
import geopandas as gpd
from prettytable import PrettyTable


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


def shortest_paths_igraph(geodf, source_ensbl, dest_ensbl):
    """Perform many-to-many shortest path calculation with python-igraph.

    Positional arguments:
    geodf -- A geodataframe describing the edges. Must contain
    columns "from_node", "to_node" and "length_km".
    source_ensbl: A list of node ids for starting nodes.
    dest_ensbl: A list of node ids for destination nodes.

    Returns:
    A list of shortest paths (themselves list of edges).
    """
    edges = geodf.loc[:, ["from_node", "to_node", "length_km"]]
    g = Graph.DataFrame(edges, directed=False)
    dest = g.vs.select(name_in=dest_ensbl)

    shortest_paths = []
    for source_id in source_ensbl:
        start = g.vs.find(name=source_id)
        sp = g.get_shortest_paths(
            start, dest, weights="length_km", output="epath"
        )
        shortest_paths.append(sp)

    return shortest_paths


def shortest_paths_pandana(geodf, nodes_geodf, source_ensbl, dest_ensbl):
    """Perform many-to-many shortest path calculation with pandana.

    Positional arguments:
    geodf -- A geodataframe describing the edges. Must contain
    columns "from_node", "to_node" and "length_km".
    nodes_geodf -- A geodataframe describing the nodes.
    source_ensbl -- A list of node ids for starting nodes.
    dest_ensbl -- A list of node ids for destination nodes.

    Returns:
    A list of shortest paths (themselves list of vertices).
    """
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
    """Compute edge path from a vertex path.

    Positional arguments:
    vpath -- A list of node ids
    gdf -- A geodataframe describing the edges. Must contain
    columns "from_node", "to_node" with nodes as "roadn_<nodeid>"

    Returns:
    A list of edge IDs.
    """
    g = Graph.DataFrame(gdf.loc[:, ["from_node", "to_node"]], directed=False)
    igraph_nodes = [g.vs.find(name="roadn_" + str(i)).index for i in vpath]
    epath = g.get_eids(path=igraph_nodes, directed=False)
    return epath


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--sizes", nargs="+")
    parser.add_argument("--nreps", type=int, default=10)
    parser.add_argument("-r", "--reconstr", action="store_true")

    args = parser.parse_args()

    gdf = gpd.read_file("jamaica_roads.gpkg")
    nodes_gdf = gpd.read_file("jamaica_roads.gpkg", layer="nodes")

    igraph_times = np.zeros((args.nreps, len(args.sizes)))
    pandana_times = np.zeros((args.nreps, len(args.sizes)))
    pandana_reconstruct_times = np.zeros((args.nreps, len(args.sizes)))

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
            path = shortest_paths_igraph(gdf, source_ensbl, dest_ensbl)
            toc = time.time()
            igraph_times[irep, j] = toc - tic

            print("        Timing pandana")
            tic = time.time()
            vpaths = shortest_paths_pandana(
                gdf, nodes_gdf, source_ensbl, dest_ensbl
            )
            toc = time.time()
            pandana_times[irep, j] = toc - tic

            if args.reconstr:
                print("        Timing pandana reconstruction step")
                tic = time.time()
                epaths = [reconstruct_epath(vpath, gdf) for vpath in vpaths]
                toc = time.time()
                pandana_reconstruct_times[irep, j] = toc - tic

    tbl = PrettyTable()
    tbl.add_column("Size", args.sizes)
    tbl.add_column("Igraph (avg)", np.mean(igraph_times, axis=0))
    tbl.add_column("Igraph (std)", np.std(igraph_times, axis=0))
    tbl.add_column("pandana (avg)", np.mean(pandana_times, axis=0))
    tbl.add_column("pandana (std)", np.std(pandana_times, axis=0))
    if args.reconstr:
        tbl.add_column(
            "reconstr (avg)", np.mean(pandana_reconstruct_times, axis=0)
        )
        tbl.add_column(
            "reconstr (std)", np.std(pandana_reconstruct_times, axis=0)
        )

    from prettytable import MARKDOWN

    tbl.float_format = "1.2"
    tbl.set_style(MARKDOWN)
    with open("table.txt", "w") as f:
        f.write(tbl.get_string())
        f.write("\n")
    print(tbl)
