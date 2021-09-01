import random

import pandana
from pandas import DataFrame
import numpy as np
from igraph import Graph
import geopandas as gpd

gdf = gpd.read_file("jamaica_roads.gpkg")
nodes_gdf = gpd.read_file("jamaica_roads.gpkg", layer="nodes")

# Sample source and desitnation ensembles
Ns = 10
Nt = 10
source_ensbl = random.sample(nodes_gdf["node_id"], Ns)
dest_ensbl = random.sample(nodes_gdf["node_id"], Nt)

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
from_node = [int(node_id[6:]) for node_id in gdf["from_node"]]
to_node = [int(node_id[6:]) for node_id in gdf["to_node"]]

edges = DataFrame(
    {"from": from_node, "to": to_node, "weight": gdf["length_km"].values}
)
net = pandana.Network(
    nodes["x"], nodes["y"], edges["from"], edges["to"], edges[["weight"]]
)

sp_pandana = net.shortest_paths(source_ensbl * len(dest_ensbl), dest_ensbl)
