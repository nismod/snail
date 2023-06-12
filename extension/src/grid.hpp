#ifndef GRID_H
#define GRID_H

#include <limits>
#include <sstream>
#include <string>
#include <vector>

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
  /// 1D vector of doubles storing the data (conceptually a 2D grid)
  std::vector<double> data;

  Grid()
      : ncols{0}, nrows{0},
        grid_to_world{Affine()}, data{std::vector<double>()} {
    world_to_grid = ~grid_to_world;
  };

  Grid(size_t ncols, size_t nrows, Affine grid_to_world)
      : ncols{ncols}, nrows{nrows},
        grid_to_world{grid_to_world}, data{std::vector<double>()} {
    world_to_grid = ~grid_to_world;
  };

  Grid(size_t ncols, size_t nrows, Affine grid_to_world,
       std::vector<double> data)
      : ncols{ncols}, nrows{nrows}, grid_to_world{grid_to_world}, data{data} {
    world_to_grid = ~grid_to_world;
  };

  /// Calculate hashed index in raster.
  size_t cellIndex(const geometry::Coord p) const {
    auto offset = cellIndices(p);
    return std::get<0>(offset) + std::get<1>(offset) * ncols;
  }

  /// Recover i, j index in raster.
  std::tuple<int, int>
  cellIndices(const geometry::Coord p,
              double epsilon = std::numeric_limits<double>::epsilon()) const {
    // Note on epsilon: nudge point slightly in the x and y direction towards
    // the cell centre
    // - this should allow for some tolerance in coordinate precision and avoid
    // off-by-one errors
    // TODO confirm and construct test case to demonstrate.
    auto offset = world_to_grid * (p + geometry::Coord(epsilon, epsilon));
    return std::make_tuple(floor(offset.x), floor(offset.y));
  }

  /// Calculate the relative position of a point in a cell.
  geometry::Coord offsetInCell(const geometry::Coord p) const {
    // Retrieve the indices of the cell.
    auto cell_offset = cellIndices(p);

    // Calculate the LL coordinates of the cell.
    geometry::Coord cell = grid_to_world * geometry::Coord(cell_offset);

    // Return the difference between the two points.
    return p - cell;
  }

  /// Calculate the points at which a line-segment intersects the
  /// grid lines / graticules.
  std::vector<geometry::Coord>
  findIntersections(const geometry::Line line) const {
    // Create a vector of grid / graticule crossings to return to the caller.
    std::vector<geometry::Coord> crossings;

    // First, calculate the run and rise of the line.
    double run = (line.end.x - line.start.x);
    double rise = (line.end.y - line.start.y);

    // If the start and end point are the same, return early
    if (run == 0 && rise == 0) {
      return crossings;
    }

    // Append the start point of the line to the vector of crossings.
    crossings.push_back(line.start);

    // Calculate the length of the line segment being passed in for testing.
    double length = line.length();

    // Pull cell size out of grid transform
    double cellsize_x = grid_to_world.a;
    double cellsize_y = grid_to_world.e;

    // Determine horizontal and vertical heading, in terms of cellsize unit.
    // The double comparison is necessary in case of negative cell sizes, which
    // are allowed in the definition of grid transforms.
    bool east = ((run > 0) == (cellsize_x > 0));
    bool north = ((rise > 0) == (cellsize_y > 0));

    // We will step east or west, north or south, according to heading
    double step_y = north ? cellsize_y : -cellsize_y;
    double step_x = east ? cellsize_x : -cellsize_x;

    // Set up initial delta from the start of the line to the first crossing.
    geometry::Coord delta = offsetInCell(line.start);
    double dN = north ? cellsize_y - delta.y : -delta.y;
    double dE = east ? cellsize_x - delta.x : -delta.x;

    // Calculate the positions at which the line crosses the grid graticules.
    geometry::Coord pN(dN * run / rise, dN);
    geometry::Coord pE(dE, dE * rise / run);

    // Consider each possible crossing point along horizontal line
    if (rise == 0) {
      while (pE.length() <= length) {
        // std::cout << "  pE(" << pE.x << "," << pE.y << ") length "  <<
        // pE.length() << "\n";

        // Add the closest crossing point
        crossings.push_back(line.start + pE);
        // Update the distance to the next graticule
        dE += step_x;
        // Calculate the position of the next crossing
        pE = geometry::Coord(dE, 0);
      }

      // Consider each possible crossing point along vertical line
    } else if (run == 0) {
      while (pN.length() <= length) {

        // Add the closest crossing point
        crossings.push_back(line.start + pN);
        // Update the distance to the next graticule
        dN += step_y;
        // Calculate the position of the next crossing
        pN = geometry::Coord(0, dN);
      }

      // Consider each possible crossing point along line
    } else {
      while (pE.length() <= length || pN.length() <= length) {
        // Add the closest crossing point to the vector of grid /
        // graticule crossings.  If both crossing points overlap then we
        // update both and it doesn't matter which one we add to the
        // vector of crossings.
        if (pE == pN) {
          crossings.push_back(line.start + pN);
          // Update the distance to the next graticule.
          dE += step_x;
          dN += step_y;
          // Calculate the position of the next crossings in both directions
          pE = geometry::Coord(dE, dE * rise / run);
          pN = geometry::Coord(dN * run / rise, dN);
        } else if (pE.length() < pN.length()) {
          crossings.push_back(line.start + pE);
          // Update the distance to the next graticule.
          dE += step_x;
          // Calculate the position of the next crossing
          pE = geometry::Coord(dE, dE * rise / run);
        } else if (pN.length() < pE.length()) {
          crossings.push_back(line.start + pN);
          // Update the distance to the next graticule.
          dN += step_y;
          // Calculate the position of the next crossing
          pN = geometry::Coord(dN * run / rise, dN);
        } else {
          std::ostringstream msg;
          msg << "Unexpected points while splitting line:\n";
          msg << "  start(" << line.start.x << "," << line.start.y << ")\n";
          msg << "  end(" << line.end.x << "," << line.end.y << ")\n";
          msg << "  run: " << run << " rise: " << rise << "\n";
          msg << "  pE(" << pE.x << "," << pE.y << ") length " << pE.length()
              << "\n";
          msg << "  pN(" << pN.x << "," << pN.y << ") length " << pN.length()
              << "\n";
          utils::Exception(msg.str());
          break;
        }
      }
    }

    // Return the vector of crossing points that exist along the line.
    return crossings;
  }
};

} // namespace grid
} // namespace snail
#endif // GRID_H
