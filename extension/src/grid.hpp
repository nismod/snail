#ifndef GRID_H
#define GRID_H

#include <limits>
#include <tuple>

#include "geometry.hpp"
#include "transform.hpp"
#include "utils.hpp"

namespace snail {
namespace grid {

using transform::Affine;

/// Structure defining a raster grid.
struct Grid {
  /// number of columns
  size_t ncols;
  /// number of rows
  size_t nrows;
  /// grid to world transform - provided (derived from x, y offset and cellsize)
  Affine grid_to_world;
  /// world to grid transform - calculated
  Affine world_to_grid;
  /// Upper right corner
  geometry::Coord ur;
  /// Lower left corner
  geometry::Coord ll;

  Grid() : ncols{0}, nrows{0}, grid_to_world{Affine()}, ur{0.0, 0.0}, ll{0.0, 0.0} {
    world_to_grid = ~grid_to_world;
  };

  Grid(size_t ncols, size_t nrows, Affine grid_to_world)
      : ncols{ncols}, nrows{nrows}, grid_to_world{grid_to_world} {
    world_to_grid = ~grid_to_world;
    ll = grid_to_world * geometry::Coord(0.0, 0.0);
    ur = grid_to_world * geometry::Coord(static_cast<double>(ncols), static_cast<double>(nrows));
  };

  /// Calculate hashed index in raster.
  size_t cellIndex(const geometry::Coord p) const {
    auto offset = cellIndices(p);
    return std::get<0>(offset) + std::get<1>(offset) * ncols;
  }

  /// Recover i, j index in raster.
  std::tuple<int, int> cellIndices(const geometry::Coord p,
                                   double epsilon = std::numeric_limits<double>::epsilon()) const {
    // Note on epsilon: nudge point slightly in the x and y direction towards
    // the cell centre
    // - this should allow for some tolerance in coordinate precision and avoid
    // off-by-one errors
    // TODO confirm and construct test case to demonstrate.
    auto offset = world_to_grid * (p + geometry::Coord(epsilon, epsilon));
    return std::make_tuple(floor(offset.x), floor(offset.y));
  }
};

} // namespace grid
} // namespace snail
#endif // GRID_H
