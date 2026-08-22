import math
import os
import random

import geopandas as gpd
import shapely
import numpy as np
import pytest
from hilbertcurve.hilbertcurve import HilbertCurve
from numpy.testing import assert_array_equal
from rasterio.crs import CRS
from shapely import box
from shapely.geometry import LineString, Point, Polygon
from shapely.geometry.polygon import LinearRing, orient

from snail.intersection import (
    aggregate_values_to_grid,
    GridDefinition,
    split_linestrings,
    split_polygons,
    split_polygons_experimental,
    generate_grid_boxes,
)
from snail.core.intersections import split_linestring as core_split_linestring


@pytest.fixture
def linestrings():
    test_linestrings = [
        LineString([(0.5, 0.5), (0.75, 0.5), (1.5, 0.5), (1.5, 1.5)]),
        LineString([(0.5, 0.5), (0.75, 0.5), (1.5, 1.5)]),
    ]
    gdf = gpd.GeoDataFrame({"col1": ["name1", "name2"], "geometry": test_linestrings})
    return gdf


@pytest.fixture
def linestrings_split():
    expected_splits = [
        LineString([(0.5, 0.5), (0.75, 0.5), (1.0, 0.5)]),
        LineString([(1.0, 0.5), (1.5, 0.5), (1.5, 1.0)]),
        LineString([(1.5, 1.0), (1.5, 1.5)]),
    ] + [
        LineString([(0.5, 0.5), (0.75, 0.5), (1.0, 0.8333)]),
        LineString([(1.0, 0.8333), (1.125, 1.0)]),
        LineString([(1.125, 1.0), (1.5, 1.5)]),
    ]
    expected_gdf = gpd.GeoDataFrame(
        {"col1": ["name1"] * 3 + ["name2"] * 3, "geometry": expected_splits},
        index=[0] * 3 + [1] * 3,
    )
    return expected_gdf


@pytest.fixture
def polygon():
    test_linearing = LinearRing(
        [
            (1.5, 0.25),
            (2.5, 1.5),
            (2.5, 3.5),
            (1.5, 2.25),
            (0.5, 3.5),
            (0.5, 1.5),
        ]
    )
    counter_clockwise = 1
    test_polygon = orient(Polygon(test_linearing), counter_clockwise)
    return gpd.GeoDataFrame({"col1": ["name1"], "geometry": [test_polygon]})


@pytest.fixture
def polygon_split():
    rings = [
        [
            (0.9, 1.0),
            (0.5, 1.5),
            (0.5, 2.0),
            (1.0, 2.0),
            (1.0, 1.0),
            (0.9, 1.0),
        ],
        [
            (0.9, 3.0),
            (1.0, 2.875),
            (1.0, 2.0),
            (0.5, 2.0),
            (0.5, 3.0),
            (0.9, 3.0),
        ],
        [(1.0, 0.875), (0.9, 1.0), (1.0, 1.0), (1.0, 0.875)],
        [(0.5, 3.0), (0.5, 3.5), (0.9, 3.0), (0.5, 3.0)],
        [
            (2.0, 0.875),
            (1.5, 0.25),
            (1.0, 0.875),
            (1.0, 1.0),
            (2.0, 1.0),
            (2.0, 0.875),
        ],
        [
            (1.0, 2.875),
            (1.5, 2.25),
            (2.0, 2.875),
            (2.0, 2.0),
            (1.0, 2.0),
            (1.0, 2.875),
        ],
        [(2.0, 1.0), (1.0, 1.0), (1.0, 2.0), (2.0, 2.0), (2.0, 1.0)],
        [(2.1, 1.0), (2.0, 0.875), (2.0, 1.0), (2.1, 1.0)],
        [
            (2.5, 2.0),
            (2.5, 1.5),
            (2.1, 1.0),
            (2.0, 1.0),
            (2.0, 2.0),
            (2.5, 2.0),
        ],
        [
            (2.5, 3.0),
            (2.5, 2.0),
            (2.0, 2.0),
            (2.0, 2.875),
            (2.1, 3.0),
            (2.5, 3.0),
        ],
        [(2.1, 3.0), (2.5, 3.5), (2.5, 3.0), (2.1, 3.0)],
    ]
    expected_polygons = [Polygon(ring) for ring in rings]
    expected_idx = ["name1"] * len(rings)
    expected_gdf = gpd.GeoDataFrame(
        {"col1": expected_idx, "geometry": expected_polygons}
    )
    expected_gdf["index"] = 0
    return expected_gdf.set_index("index")


@pytest.fixture
def grid():
    return GridDefinition(crs=None, width=4, height=4, transform=(1, 0, 0, 0, 1, 0))


def test_grid_from_extent(grid):
    actual = GridDefinition.from_extent(
        xmin=0, ymin=0, xmax=4, ymax=4, cell_width=1, cell_height=1, crs=None
    )
    assert actual == grid


def test_grid_from_raster():
    fname = os.path.join(
        os.path.dirname(__file__),
        "integration",
        "range.tif",
    )
    actual = GridDefinition.from_raster(fname)
    expected = GridDefinition(
        crs=CRS.from_epsg(4326),
        width=23,
        height=14,
        transform=(
            0.008333333347826087,
            0.0,
            -1.341666667,
            0.0,
            -0.008333333285714315,
            51.808333333,
        ),
    )
    assert actual == expected


class TestSnailIntersections:
    def test_split_linestrings(self, grid, linestrings, linestrings_split):
        actual = split_linestrings(linestrings, grid)
        expected_gdf = linestrings_split

        # Assertions

        # Ideally we'd like to use geopandas.assert_geodataframe_equal to
        # to compare both expected and actual geodfs, but this function offers
        # little control over tolerance. When using option "check_less_precise",
        # it uses GeoSeries.geom_equals_exact under the hood, which has an kwarg
        # "tolerance". But assert_geodataframe_equal does not recognise kwarg "tolerance".
        assert (
            actual["geometry"]
            .geom_equals_exact(expected_gdf["geometry"], tolerance=1e-3)
            .values.all()
        )
        assert_array_equal(actual["col1"].values, expected_gdf["col1"].values)

    def test_split_polygons(self, grid, polygon, polygon_split):
        actual = sort_polygons(split_polygons(polygon, grid))
        expected = sort_polygons(polygon_split)

        for i in range(len(actual)):
            actual_geom = actual.iloc[i, 1]
            expected_geom = expected.iloc[i, 1]
            assert actual_geom.equals(expected_geom)
        assert_array_equal(actual["col1"].values, expected["col1"].values)

    def test_split_polygons_experimental(self, grid, polygon, polygon_split):
        actual = sort_polygons(split_polygons_experimental(polygon, grid))
        expected = sort_polygons(polygon_split)

        assert len(actual) == len(expected)
        for i in range(len(actual)):
            actual_geom = actual.iloc[i, 1]
            expected_geom = expected.iloc[i, 1]
            assert actual_geom.equals(expected_geom)
        assert_array_equal(actual["col1"].values, expected["col1"].values)

    def test_split_polygons_experimental_with_hole(self, grid):
        polygon_with_hole = Polygon(
            [(0.5, 0.5), (2.5, 0.5), (2.5, 2.5), (0.5, 2.5)],
            [[(1.25, 1.25), (1.75, 1.25), (1.75, 1.75), (1.25, 1.75)]],
        )
        gdf = gpd.GeoDataFrame({"col1": ["name1"], "geometry": [polygon_with_hole]})
        expected = sort_polygons(split_polygons(gdf.copy(), grid))
        actual = sort_polygons(split_polygons_experimental(gdf.copy(), grid))

        assert len(actual) == 9
        assert actual.geometry.area.sum() == pytest.approx(polygon_with_hole.area)
        assert len(actual) == len(expected)
        for i in range(len(actual)):
            assert actual.iloc[i, 1].equals(expected.iloc[i, 1])


def test_box_geom_bounds():
    """Values take from tests/integration/range.tif"""
    grid = GridDefinition(
        crs=CRS.from_epsg(4326),
        width=23,
        height=14,
        transform=(
            0.008333333347826087,
            0.0,
            -1.341666667,
            0.0,
            -0.008333333285714315,
            51.808333333,
        ),
    )
    box_geoms = generate_grid_boxes(grid)
    minb = box_geoms.bounds.min()
    maxb = box_geoms.bounds.max()

    atol = 1e-4
    assert abs(minb.minx - -1.3416667) < atol
    assert abs(minb.miny - 51.6916667) < atol
    assert abs(maxb.maxx - -1.1500000) < atol
    assert abs(maxb.maxy - 51.8083333) < atol


def sort_polygons(df):
    iterations = 6  # all coords must be <= (2**p - 1) ; 2**6 - 1 == 63
    ndimensions = 2
    hilbert_curve = HilbertCurve(iterations, ndimensions)
    points = df.geometry.centroid
    coords = np.array(list(zip(points.x.values.tolist(), points.y.values.tolist())))
    int_coords = (coords * 10).astype(int)
    distances = hilbert_curve.distances_from_points(int_coords)
    df["hilbert_distance"] = distances
    return df.sort_values(by="hilbert_distance").drop(columns="hilbert_distance")


def test_aggregate_values_to_grid_sum():
    grid = GridDefinition(
        crs=CRS.from_epsg(4326),
        width=3,
        height=2,
        transform=(1.0, 0.0, 0.0, 0.0, 1.0, 0.0),
    )
    splits = gpd.GeoDataFrame(
        {
            "index_i": [0, 0, 2, 1, -1],
            "index_j": [0, 0, 1, 1, 0],
            "length_km": [1.0, 2.0, 4.1, 0.0, 99.0],
        },
        geometry=[Point(0, 0)] * 5,
        crs=grid.crs,
    )
    # values are summed, out-of-bounds ignored, dtype preserved
    aggregated = aggregate_values_to_grid(splits, "length_km", grid)
    expected = np.array([[3.0, 0.0, 0.0], [0.0, 0.0, 4.1]])
    assert_array_equal(aggregated, expected)

    # negative fill value works okay
    aggregated = aggregate_values_to_grid(splits, "length_km", grid, fill_value=-99)
    expected = np.array([[3.0, -99, -99], [-99, 0.0, 4.1]])
    assert_array_equal(aggregated, expected)

    # nan fill value
    aggregated = aggregate_values_to_grid(splits, "length_km", grid, fill_value=np.nan)
    expected = np.array([[3.0, np.nan, np.nan], [np.nan, 0.0, 4.1]])
    assert_array_equal(aggregated, expected)

    # cast to integer dtype
    aggregated = aggregate_values_to_grid(splits, "length_km", grid, dtype="int")
    expected = np.array([[3, 0, 0], [0, 0, 4]], dtype="int64")
    assert_array_equal(aggregated, expected)


def test_core_split_linestring_zero_length_on_corner():
    transform = (1.0, 0.0, 0.0, 0.0, 1.0, 0.0)
    parts = core_split_linestring(
        LineString([(0.0, 0.0), (1.0, 1.0), (2.0, 0.0)]),
        2,
        2,
        transform,
    )
    lengths = [geom.length for geom in parts]
    assert all(length > 0 for length in lengths)


def _random_test_polygons(seed, count):
    """Polygons chosen to stress grid splitting: some with holes, and some
    with coordinates rounded onto grid lines to provoke degenerate cases
    (vertices and edges exactly on a cell border)"""
    rng = random.Random(seed)
    polygons = []
    while len(polygons) < count:
        centre = Point(rng.uniform(1, 9), rng.uniform(1, 9))
        radius = rng.uniform(0.05, 3.0)
        resolution = rng.choice([1, 2, 4, 8])
        geom = centre.buffer(radius, resolution=resolution)
        if rng.random() < 0.4:
            geom = geom.difference(centre.buffer(radius * 0.4, resolution=resolution))
        if rng.random() < 0.5:
            # snap coordinates onto a coarse grid, to land vertices and
            # edges exactly on cell borders, then drop the precision model
            # it leaves behind: GEOS applies that model to later overlay
            # operations, which would make the comparisons below disagree
            # with themselves - intersecting such a geometry with each of
            # its cells loses the slivers thinner than the precision
            geom = shapely.set_precision(shapely.set_precision(geom, 0.1), 0)
        if geom.geom_type == "Polygon" and geom.is_valid and not geom.is_empty:
            polygons.append(geom)
    return polygons


@pytest.mark.parametrize(
    "transform",
    [
        (1, 0, 0, 0, 1, 0),  # y increasing north
        (1, 0, 0, 0, -1, 10),  # y increasing south, as for a north-up raster
        (0.5, 0, -1, 0, -0.5, 11),  # offset, fractional cell size
    ],
)
def test_split_polygons_experimental_random(transform):
    """Splitting must conserve area and yield valid pieces, each within a
    single cell, however the polygon falls on the grid"""
    grid = GridDefinition(crs=None, width=40, height=40, transform=transform)
    cell_width, cell_height = abs(transform[0]), abs(transform[4])

    polygons = _random_test_polygons(seed=20220309, count=200)
    features = gpd.GeoDataFrame({"col1": range(len(polygons)), "geometry": polygons})
    splits = split_polygons_experimental(features, grid)

    assert splits.geometry.is_valid.all()
    assert (splits.geometry.geom_type == "Polygon").all()

    # each polygon's pieces account for exactly its area
    areas = splits.geometry.area.groupby(splits["col1"]).sum()
    for i, polygon in enumerate(polygons):
        assert areas[i] == pytest.approx(polygon.area, rel=1e-9), (
            f"polygon {i} not conserved: {polygon.wkt}"
        )

    # every piece lies within one cell, so is no larger than one
    bounds = splits.geometry.bounds
    assert ((bounds.maxx - bounds.minx) <= cell_width + 1e-9).all()
    assert ((bounds.maxy - bounds.miny) <= cell_height + 1e-9).all()


def test_split_polygons_experimental_matches_cells():
    """Each piece must be exactly what the polygon has in its own cell.

    Comparing cell by cell against a direct intersection with that cell's
    box, rather than comparing totals, catches a gap in one cell paid for
    by an overlap in another.
    """
    grid = GridDefinition(crs=None, width=30, height=30, transform=(1, 0, 0, 0, 1, 0))

    for polygon in _random_test_polygons(seed=99, count=40):
        features = gpd.GeoDataFrame({"col1": ["name1"], "geometry": [polygon]})
        splits = split_polygons_experimental(features, grid)

        # a piece lies within one cell, but its vertices may sit on that
        # cell's border, so the middle of its extent identifies the cell
        bounds = splits.geometry.bounds
        i = np.floor((bounds.minx.to_numpy() + bounds.maxx.to_numpy()) / 2).astype(int)
        j = np.floor((bounds.miny.to_numpy() + bounds.maxy.to_numpy()) / 2).astype(int)

        by_cell = {}
        for cell, area in zip(zip(i.tolist(), j.tolist()), splits.geometry.area):
            by_cell[cell] = by_cell.get(cell, 0.0) + area

        minx, miny, maxx, maxy = polygon.bounds
        for ci in range(math.floor(minx), math.ceil(maxx)):
            for cj in range(math.floor(miny), math.ceil(maxy)):
                expected = polygon.intersection(box(ci, cj, ci + 1, cj + 1)).area
                actual = by_cell.pop((ci, cj), 0.0)
                assert actual == pytest.approx(expected, abs=1e-9), (
                    f"cell ({ci}, {cj}) has {actual}, expected {expected}, "
                    f"for {polygon.wkt}"
                )
        assert not by_cell, f"pieces outside the polygon's cells: {sorted(by_cell)}"


def test_split_linestrings_random():
    """Splitting a linestring must conserve its length, and every piece must
    lie within one cell"""
    grid = GridDefinition(crs=None, width=40, height=40, transform=(1, 0, 0, 0, 1, 0))
    rng = random.Random(4242)

    lines = []
    for _ in range(200):
        x, y = rng.uniform(2, 35), rng.uniform(2, 35)
        points = [(x, y)]
        for _ in range(rng.randint(1, 8)):
            x += rng.uniform(-4, 4)
            y += rng.uniform(-4, 4)
            points.append((x, y))
        lines.append(LineString(points))

    features = gpd.GeoDataFrame({"col1": range(len(lines)), "geometry": lines})
    splits = split_linestrings(features, grid)

    assert (splits.geometry.length > 0).all()
    lengths = splits.geometry.length.groupby(splits["col1"]).sum()
    for i, line in enumerate(lines):
        assert lengths[i] == pytest.approx(line.length, rel=1e-9)

    bounds = splits.geometry.bounds
    assert ((bounds.maxx - bounds.minx) <= 1 + 1e-9).all()
    assert ((bounds.maxy - bounds.miny) <= 1 + 1e-9).all()
