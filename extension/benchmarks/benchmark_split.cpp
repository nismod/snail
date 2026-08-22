// Benchmark the C++ polygon splitting core
// (snail::operations::splitPolygonGrid)
//
// Build and run:
//   cmake -Bbuild ./extension && cmake --build build && ./build/run_benchmarks
//
// Times the split alone - Python conversion costs are excluded. See
// scripts/benchmark_split.py for the Python-level comparison against the
// shapely overlay implementation.

#include <chrono>
#include <cmath>
#include <cstdio>
#include <vector>

#include "geometry.hpp"
#include "grid.hpp"
#include "operations.hpp"

using snail::geometry::Coord;
using linestr = std::vector<Coord>;

/// A closed ring approximating a circle
static linestr circle(double cx, double cy, double radius, int segments) {
  linestr ring;
  ring.reserve(segments + 1);
  for (int i = 0; i < segments; i++) {
    double angle = 2.0 * snail::utils::PI * i / segments;
    ring.push_back(
        Coord(cx + radius * std::cos(angle), cy + radius * std::sin(angle)));
  }
  ring.push_back(ring.front());
  return ring;
}

static void benchmark(const char *name, const std::vector<linestr> &rings,
                      const snail::grid::Grid &grid, int repetitions) {
  // splitPolygonGridPieces is what the Python extension calls; the
  // per-piece splitPolygonGrid is a convenience wrapper over it
  std::size_t pieces =
      snail::operations::splitPolygonGridPieces(rings, grid).size();
  auto start = std::chrono::steady_clock::now();
  for (int i = 0; i < repetitions; i++) {
    snail::operations::splitPolygonGridPieces(rings, grid);
  }
  auto end = std::chrono::steady_clock::now();
  double micros =
      std::chrono::duration<double, std::micro>(end - start).count() /
      repetitions;
  std::printf("%-32s %10.2f us/split  %6zu pieces  (x%d)\n", name, micros,
              pieces, repetitions);
}

int main() {
  // A small building with vertices close to grid lines, from the error
  // cases attached to https://github.com/nismod/snail/issues/45
  snail::grid::Grid building_grid(
      698, 252,
      snail::transform::Affine(0.0030999999999999925, 0.0, -78.34655, 0.0,
                               -0.0030999999999999934, 18.52365));
  linestr building = {{-77.280182457, 17.97282044099908},
                      {-77.28015690000001, 17.97282209999907},
                      {-77.28015000000001, 17.9727272999991},
                      {-77.28013888, 17.97272804099908},
                      {-77.28013888, 17.97270079499907},
                      {-77.28014880000001, 17.97270009999908},
                      {-77.2801487, 17.9726981999991},
                      {-77.28018086100001, 17.97269609699906},
                      {-77.280182457, 17.97282044099908}};

  // Circles of increasing size on a unit grid
  snail::grid::Grid unit_grid(100, 100, snail::transform::Affine());
  linestr small_circle = circle(50.5, 50.5, 2.2, 32);
  linestr medium_circle = circle(50.5, 50.5, 10.2, 64);
  linestr large_circle = circle(50.0, 50.0, 45.0, 256);

  // A large circle with a large hole: boundary-heavy, few interior cells
  std::vector<linestr> annulus = {large_circle, circle(50.0, 50.0, 42.0, 256)};

  benchmark("building (2 cells)", {building}, building_grid, 200000);
  benchmark("small circle (~5x5 cells)", {small_circle}, unit_grid, 50000);
  benchmark("medium circle (~20x20 cells)", {medium_circle}, unit_grid, 20000);
  benchmark("large circle (~90x90 cells)", {large_circle}, unit_grid, 2000);
  benchmark("annulus (~90x90, wide hole)", annulus, unit_grid, 2000);

  return 0;
}
