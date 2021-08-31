import random

import pandana
from pandas import DataFrame
import numpy as np
from igraph import Graph
import geopandas as gpd

gdf = gpd.read_file("jamaica_roads.gpkg")

# Transform source and destination columns into lists of integers
# (node ids)
from_node = [int(node_id[6:]) for node_id in gdf["from_node"]]
to_node = [int(node_id[6:]) for node_id in gdf["to_node"]]

# Array of all node ids
node_ids, geodf_idxs = np.unique(from_node + to_node, return_index=True)

# Sample source and desitnation ensembles
Ns = 10
Nt = 10
source_ensbl = random.sample(list(node_ids), Ns)
dest_ensbl = random.sample(list(node_ids), Nt)

# ########## python-igraph ##########

edges = gdf.loc[:, ["from_node", "to_node", "length_km"]]
g = Graph.DataFrame(edges, directed=False)
dest = g.vs.select(name_in=["roadn_" + str(i) for i in dest_ensbl])

for source_id in source_ensbl:
    start = g.vs.find(name="roadn_" + str(source_id))
    sp_igraph = g.get_shortest_paths(
        start, dest, weights="length_km", output="vpath"
    )


# ########## pandana ##########

# Build arrays of coordinate values
x = []
y = []
for geodf_idx in geodf_idxs:
    linestr = gdf.loc[geodf_idx % len(from_node), "geometry"]
    coords = linestr.coords[int(-1 * (geodf_idx // len(from_node)))]
    x.append(coords[0])
    y.append(coords[1])

nodes = DataFrame({"x": x, "y": y}, index=node_ids)
edges = DataFrame(
    {"from": from_node, "to": to_node, "weight": gdf["length_km"].values}
)
net = pandana.Network(
    nodes["x"], nodes["y"], edges["from"], edges["to"], edges[["weight"]]
)

sp_pandana = net.shortest_paths(source_ensbl * len(dest_ensbl), dest_ensbl)
