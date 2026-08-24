"""Benchmark polygon splitting: split_polygons vs split_polygons_experimental

Compares the default shapely/GEOS overlay implementation against the C++
line-scan implementation on a few representative workloads, checking that
both conserve the total polygon area.

Run:
    python scripts/benchmark_split.py

See extension/benchmarks/benchmark_split.cpp for a benchmark of the C++
splitting core alone (excluding Python conversion costs).
"""

import random
import time

import geopandas
from shapely.geometry import Point, Polygon

from snail.intersection import (
    GridDefinition,
    split_polygons,
    split_polygons_experimental,
)


def buildings_workload(n=500, seed=42):
    """Small quadrilaterals (~1/20 cell size) scattered over a unit grid,
    some sitting exactly on grid lines"""
    random.seed(seed)
    geoms = []
    for _ in range(n):
        cx, cy = random.uniform(5, 195), random.uniform(5, 195)
        w, h = random.uniform(0.02, 0.08), random.uniform(0.02, 0.08)
        if random.random() < 0.2:
            # snap a corner onto a grid line
            cx = round(cx)
        geoms.append(
            Polygon(
                [
                    (cx - w, cy - h),
                    (cx + w, cy - h),
                    (cx + w, cy + h),
                    (cx - w, cy + h),
                ]
            )
        )
    grid = GridDefinition(crs=None, width=200, height=200, transform=(1, 0, 0, 0, 1, 0))
    return geoms, grid


def circles_workload(n=50, seed=42):
    """Medium polygons spanning ~20x20 cells each"""
    random.seed(seed)
    geoms = [
        Point(random.uniform(15, 185), random.uniform(15, 185)).buffer(
            random.uniform(8, 12), resolution=16
        )
        for _ in range(n)
    ]
    grid = GridDefinition(crs=None, width=200, height=200, transform=(1, 0, 0, 0, 1, 0))
    return geoms, grid


def large_workload():
    """One large polygon spanning ~90x90 cells"""
    geoms = [Point(50, 50).buffer(45, resolution=64)]
    grid = GridDefinition(crs=None, width=100, height=100, transform=(1, 0, 0, 0, 1, 0))
    return geoms, grid


def run(split_func, geoms, grid, repetitions):
    gdf = geopandas.GeoDataFrame({"col1": range(len(geoms)), "geometry": geoms})
    best = float("inf")
    for _ in range(repetitions):
        start = time.perf_counter()
        splits = split_func(gdf.copy(), grid)
        best = min(best, time.perf_counter() - start)
    return best, splits


def main():
    workloads = [
        ("500 small buildings", *buildings_workload(), 3),
        ("50 medium circles", *circles_workload(), 3),
        ("1 large circle", *large_workload(), 3),
    ]
    print(
        f"{'workload':<22} {'overlay':>10} {'experimental':>13} {'speedup':>8}   pieces"
    )
    for name, geoms, grid, repetitions in workloads:
        t_overlay, s_overlay = run(split_polygons, geoms, grid, repetitions)
        t_experimental, s_experimental = run(
            split_polygons_experimental, geoms, grid, repetitions
        )
        expected = sum(g.area for g in geoms)
        for label, splits in (("overlay", s_overlay), ("experimental", s_experimental)):
            total = splits.geometry.area.sum()
            if abs(total - expected) > 1e-6 * expected:
                raise AssertionError(
                    f"{name}/{label}: area {total} != expected {expected}"
                )
        print(
            f"{name:<22} {t_overlay * 1000:>8.1f}ms {t_experimental * 1000:>11.1f}ms"
            f" {t_overlay / t_experimental:>7.1f}x   {len(s_overlay)} / {len(s_experimental)}"
        )


if __name__ == "__main__":
    main()
