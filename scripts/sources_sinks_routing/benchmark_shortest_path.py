from igraph import Graph, plot

N = 10
M = 10
g = Graph.Lattice(dim=[N, M], circular=False)

sp = g.get_shortest_paths(20, 35)
layout = g.layout_grid()

for path in sp:
    g.vs[path]["color"] = "blue"
plot(g, layout=layout)
