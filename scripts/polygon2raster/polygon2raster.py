from shapely.geometry.polygon import LinearRing, orient
from shapely.geometry import LineString, Polygon
from shapely.ops import polygonize

import matplotlib.pyplot as plt

from snail.intersections import split as split_one_geom

BLUE = "#6699cc"
GRAY = "#999999"
RED = "#ff3333"
GREEN = "#339933"


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


def get_crossings(splits):
    return [split.coords[0] for split in splits]


nrows = 5
ncols = 3
points = [(1.5, 0.25), (2.5, 1.5), (2.5, 3.5), (1.5, 2.25), (0.5, 3.5), (0.5, 1.5)]
ring = orient(Polygon(points), -1)
splits = split_one_geom(ring.exterior, nrows, ncols, [1, 0, 0, 0, 1, 0])
b_box = ring.bounds

maxy = b_box[-1]
miny = b_box[1]
maxx = b_box[-2]
minx = b_box[0]

inter_points = LineString(get_crossings(splits))

inner_grid_lines = []
for y in range(int(miny) + 1, int(maxy) + 1):
    p = [coord for coord in inter_points.coords if coord[1] == y]
    line2s = list(zip(p, p[1:]))[::2]
    for line2 in line2s:
        local_splits = split_one_geom(
            LineString(line2), nrows, ncols, [1, 0, 0, 0, 1, 0]
        )
        inner_grid_lines.extend(local_splits)

for x in range(int(minx) + 1, int(maxx) + 1):
    p = [coord for coord in inter_points.coords if coord[0] == x]
    line2s = list(zip(p, p[1:]))[::2]
    for line2 in line2s:
        local_splits = split_one_geom(
            LineString(line2), nrows, ncols, [1, 0, 0, 0, 1, 0]
        )
        inner_grid_lines.extend(local_splits)

polygons = list(polygonize(splits + inner_grid_lines))

ax = plt.subplot()
plot_coords(ax, ring.exterior)
plot_line(ax, ring.exterior)
for i in range(nrows):
    plt.axhline(i, 0, ncols - 1)
for i in range(ncols):
    plt.axvline(i, 0, nrows - 1)
plot_coords(ax, inter_points, color=RED)

for line in inner_grid_lines:
    plot_line(ax, line, color=GREEN)

plt.show()
