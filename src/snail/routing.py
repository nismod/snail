def shortest_paths(sources, destinations, graph, weight):
    """Compute all shortest paths from an ensemble of sources
    to an ensemble of destinations.

    Positional arguments:
    sources -- list of source node ids (string or int).
    destinations -- list of destination node ids (string or int).
    graph: igraph.Graph instance representing the network.
    weight -- Edge attribute according to which paths should be
    weighted (string)

    Returns:
    A list of tuples (source, destination)
    A list of list of edge ids corresponding to shortest
    paths.  For each (source, destination) pair, their is either 0, 1
    or several shortest paths.
    """
    all_shortest_paths = []
    for source_id in sources:
        source_node = graph.vs.find(name=source_id)
        source_shortest_paths = graph.get_shortest_paths(
            source_node, destinations, weights=weight, output="epath"
        )
        all_shortest_paths.extend(source_shortest_paths)

    return all_shortest_paths
