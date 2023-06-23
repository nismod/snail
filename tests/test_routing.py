from igraph import Graph
from snail.routing import shortest_paths


def test_shortest_paths():
    """
            e4
        0--4
    e0  |  |
        1  | e3
    e1  |  |
        2--3
            e2
    """
    g = Graph.Ring(n=5, circular=True)
    g.vs["name"] = ["node_" + str(i) for i in range(5)]
    g.es["length_km"] = [1, 1, 1, 3, 1]
    actual = shortest_paths(
        ["node_0", "node_2"], ["node_4", "node_3"], g, "length_km"
    )
    expected = [
        [4],  # 0 to 4, along edge 4
        [0, 1, 2],  # 0 to 3, avoids long edge 3
        [1, 0, 4],  # 2 to 4, avoids long edge 3
        [2],  # 2 to 3, along edge 2
    ]
    assert actual == expected
