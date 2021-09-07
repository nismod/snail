import pandana
import geopandas as gpd
from pandas import DataFrame
import matplotlib.pyplot as plt
import numpy as np

gdf = gpd.read_file("jamaica_roads.gpkg")

from_node = [int(node_id[6:]) for node_id in gdf["from_node"]]
to_node = [int(node_id[6:]) for node_id in gdf["to_node"]]

nids, idxs = np.unique(from_node + to_node, return_index=True)
x = []
y = []
for nid, idx in zip(nids, idxs):
    linestr = gdf.loc[idx % len(from_node), "geometry"]
    coords = linestr.coords[int(-1 * (idx // len(from_node)))]
    x.append(coords[0])
    y.append(coords[1])

nodes = DataFrame({"x": x, "y": y}, index=nids)
edges = DataFrame(
    {"from": from_node, "to": to_node, "weight": gdf["length_km"].values}
)
net = pandana.Network(
    nodes["x"], nodes["y"], edges["from"], edges["to"], edges[["weight"]]
)

sp = net.shortest_path(0, 15981, imp_name="weight")
epath = []
for fromnode, tonode in zip(sp[:-1], sp[1:]):
    from_id = "roadn_" + str(fromnode)
    to_id = "roadn_" + str(tonode)
    road = gdf.index[(gdf["from_node"] == from_id) & (gdf["to_node"] == to_id)].tolist()
    if len(road) == 0:
        road = gdf.index[(gdf["from_node"] == to_id) & (gdf["to_node"] == from_id)].tolist()
    epath.extend(road)

