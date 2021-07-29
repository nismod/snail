from shapely.geometry.polygon import LinearRing
from shapely.geometry import LineString

import matplotlib.pyplot as plt

from snail.intersections import split as split_one_geom

BLUE = "#6699cc"
GRAY = "#999999"
RED = '#ff3333'


def plot_coords(ax, ob, color=GRAY, zorder=1, alpha=1):
    x, y = ob.xy
    ax.plot(x, y, "o", color=color, zorder=zorder, alpha=alpha)


def plot_line(ax, ob, color=BLUE, zorder=1, linewidth=3, alpha=1):
    x, y = ob.xy
    ax.plot(
        x,
        y,
        color=color,
        linewidth=linewidth,
        solid_capstyle="round",
        zorder=zorder,
        alpha=alpha,
    )


nrows = 5
ncols = 3
points = [(1.5, 0.5), (2.5, 1.5), (2.5, 3.5), (1.5, 3.25), (0.5, 3.5), (0.5, 1.5)]
ring = LinearRing(points)
splits = split_one_geom(ring, nrows, ncols, [1, 0, 0, 0, 1, 0])
for split in splits:
    print(list(split.coords))

inter_points = LineString([split.coords[0] for split in splits])

ax = plt.subplot()
plot_coords(ax, ring)
plot_line(ax, ring)
for i in range(nrows):
    plt.axhline(i, 0, ncols-1)
for i in range(ncols):
    plt.axvline(i, 0, nrows-1)
plot_coords(ax, inter_points, color=RED)

plt.show()
