from shapely.geometry.polygon import LinearRing
import matplotlib.pyplot as plt

from snail.intersections import split as split_one_geom

BLUE = "#6699cc"
GRAY = "#999999"


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


ax = plt.subplot()
points = [(2.5, 1.5), (3.5, 2), (3.5, 5), (2.5, 4.5), (1.5, 5), (1.5, 2)]
ring = LinearRing(points)
plot_coords(ax, ring)
plot_line(ax, ring)
plt.grid(True)
plt.show()
