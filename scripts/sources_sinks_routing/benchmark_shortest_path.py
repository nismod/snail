import random
import timeit

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


def shortest_paths_igraph(geodf, source_ensbl, dest_ensbl):
    edges = geodf.loc[:, ["from_node", "to_node", "length_km"]]
    g = Graph.DataFrame(edges, directed=False)
    dest = g.vs.select(name_in=dest_ensbl)

    shortest_paths = []
    for source_id in source_ensbl:
        start = g.vs.find(name=source_id)
        sp = g.get_shortest_paths(
            start, dest, weights="length_km", output="vpath"
        )
        shortest_paths.append(sp)

    return shortest_paths


def shortest_paths_pandana(geodf, nodes_geodf, source_ensbl, dest_ensbl):
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


if __name__ == "__main__":
    gdf = gpd.read_file("jamaica_roads.gpkg")
    nodes_gdf = gpd.read_file("jamaica_roads.gpkg", layer="nodes")

    # Sample source and destination ensembles
    Ns = 10
    Nt = 10
    source_ensbl = random.sample(list(nodes_gdf["node_id"]), Ns)
    dest_ensbl = random.sample(list(nodes_gdf["node_id"]), Nt)

    sp_pandana = shortest_paths_pandana(gdf, nodes_gdf, source_ensbl, dest_ensbl)
    sp_igraph = shortest_paths_igraph(gdf, source_ensbl, dest_ensbl)

    stmt_igraph = "shortest_paths_igraph(gdf, source_ensbl, dest_ensbl)"
    stmt_pandana = "shortest_paths_pandana(gdf, nodes_gdf, source_ensbl, dest_ensbl)"

    igraph_time = timeit.timeit(stmt=stmt_igraph, globals=globals(), number=10)
    pandana_time = timeit.timeit(stmt=stmt_pandana, globals=globals(), number=10)

    print(f"igraph took {igraph_time}s")
    print(f"pandana took {pandana_time}s")
