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


def split_along_gridlines(
    exterior_crossings, min_level=0, max_level=0, direction="horizontal"
):
    x_or_y = {"horizontal": 1, "vertical": 0}
    gridline_splits = []
    for level in range(min_level, max_level + 1):
        # Split horizontal grid lines inside polygon according to
        # intersections with vertical grid lines.
        crossings_on_gridline = list(
            filter(
                # Returns True if crossing point lies on gridline
                lambda coord: coord[x_or_y[direction]] == level,
                exterior_crossings,
            )
        )
        gridline_segments = [
            LineString(coord_pair)
            for coord_pair in zip(
                crossings_on_gridline, crossings_on_gridline[1:]
            )
        ]
        # Only every other gridline segments (between two consecutive
        # crossings) is contained in the polygon
        for gridline_segment in gridline_segments[::2]:
            splits = split_one_geom(
                gridline_segment, nrows, ncols, [1, 0, 0, 0, 1, 0]
            )
            gridline_splits.extend(splits)
    return gridline_splits


nrows = 5
ncols = 3
points = [
    (1.5, 0.25),
    (2.5, 1.5),
    (2.5, 3.5),
    (1.5, 2.25),
    (0.5, 3.5),
    (0.5, 1.5),
]
ring = orient(Polygon(points), 1)
minx, miny, maxx, maxy = ring.bounds

exterior_splits = split_one_geom(
    ring.exterior, nrows, ncols, [1, 0, 0, 0, 1, 0]
)
exterior_crossings = [split.coords[0] for split in exterior_splits]

horiz_splits = split_along_gridlines(
    exterior_crossings,
    min_level=int(miny) + 1,
    max_level=int(maxy),
    direction="horizontal",
)

vert_splits = split_along_gridlines(
    exterior_crossings,
    min_level=int(minx) + 1,
    max_level=int(maxx),
    direction="vertical",
)

polygons = list(polygonize(exterior_splits + horiz_splits + vert_splits))

ax = plt.subplot()
for p in polygons:
    plot_line(ax, p.exterior)
plt.show()
