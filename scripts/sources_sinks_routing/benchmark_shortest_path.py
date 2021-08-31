from igraph import Graph
import geopandas as gpd
import matplotlib.pyplot as plt

gdf = gpd.read_file("jamaica_roads.gpkg")
edges = gdf.loc[:, ["from_node", "to_node", "length_km"]]

g = Graph.DataFrame(edges, directed=False)

start = g.vs.find(name="roadn_0")
finish = g.vs.find(name="roadn_15891")

# Return list of list of edge IDs
# Return vpath for comparison with pandana
sp = g.get_shortest_paths(
    start.index, finish.index, weights="length_km", output="vpath"
)

base = gdf.plot()
for path in sp:
    # Select linestrings according to edge IDs making the path
    # edge ID are 1to1 with row id in dataframe
    sub_gdf = gdf.iloc[path, :]
    sub_gdf.plot(ax=base, color="green")
plt.show()
