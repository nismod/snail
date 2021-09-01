import random

import pandana
from pandas import DataFrame
import numpy as np
from igraph import Graph
import geopandas as gpd


def str_to_int(str_list):
    return [int(string[6:]) for string in str_list]


gdf = gpd.read_file("jamaica_roads.gpkg")
nodes_gdf = gpd.read_file("jamaica_roads.gpkg", layer="nodes")

# Sample source and desitnation ensembles
Ns = 10
Nt = 10
source_ensbl = random.sample(list(nodes_gdf["node_id"]), Ns)
dest_ensbl = random.sample(list(nodes_gdf["node_id"]), Nt)

# ########## python-igraph ##########

edges = gdf.loc[:, ["from_node", "to_node", "length_km"]]
g = Graph.DataFrame(edges, directed=False)
dest = g.vs.select(name_in=dest_ensbl)

for source_id in source_ensbl:
    start = g.vs.find(name=source_id)
    sp_igraph = g.get_shortest_paths(
        start, dest, weights="length_km", output="vpath"
    )

# ########## pandana ##########

list_of_coords = [(point.x, point.y) for point in nodes_gdf.loc[:, "geometry"]]
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
        nodes_a.extend([source_node]*len(dest))
        nodes_b.extend(dest)
    return nodes_a, nodes_b


nodes_a, nodes_b = get_input_list(
    str_to_int(source_ensbl), str_to_int(dest_ensbl)
)
sp_pandana = net.shortest_paths(nodes_a, nodes_b, imp_name="weight")
