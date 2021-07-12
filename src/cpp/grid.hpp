#ifndef GRID_H
#define GRID_H

#include <string>
#include <vector>

#include "geom.hpp"
#include "transform.hpp"
#include "utils.hpp"

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
    :ncols{0}, nrows{0}, grid_to_world{Affine()}, data{std::vector<double>()} {
      world_to_grid = ~grid_to_world;
    };

  Grid(size_t ncols, size_t nrows, Affine grid_to_world)
    :ncols{ncols}, nrows{nrows}, grid_to_world{grid_to_world}, data{std::vector<double>()} {
      world_to_grid = ~grid_to_world;
    };

  Grid(size_t ncols, size_t nrows, Affine grid_to_world, std::vector<double> data)
    :ncols{ncols}, nrows{nrows}, grid_to_world{grid_to_world}, data{data} {
      world_to_grid = ~grid_to_world;
    };

  /// Calculate hashed index in raster.
  int cellIndex(const geometry::Vec2<double> p) const {
    auto offset = world_to_grid * p;
    return floor(offset.x) + floor(offset.y) * ncols;
  }

  /// Recover i, j index in raster.
  geometry::Vec2<int> cellIndices(const geometry::Vec2<double> p) const {
    auto offset = world_to_grid * p;
    return geometry::Vec2<int>(floor(offset.x), floor(offset.y));
  }

  /// Calculate the relative position of a point in a cell.
  geometry::Vec2<double> offsetInCell(const geometry::Vec2<double> p) const {
    // Retrieve the indices of the cell.
    geometry::Vec2<int> cell_offset = cellIndices(p);

    // Calculate the LL coordinates of the cell.
    geometry::Vec2<double> cell = grid_to_world * cell_offset;

    // Return the vector between the two points.
    return geometry::Vec2<double>(p.x - cell.x, p.y - cell.y);
  }

  /// Calculate the points at which a line-segment intersects the
  /// grid lines / graticules.
  std::vector<geometry::Vec2<double>>
  findIntersections(const geometry::Line2<double> line) const {
    // First, calculate the run and rise of the line.
    double run = (line.end.x - line.start.x);
    double rise = (line.end.y - line.start.y);

    // Calculate the length of the line segment being passed in for testing.
    double length = line.length();

    // Work out where the start-point of the line falls in the cell.
    geometry::Vec2<double> delta = offsetInCell(line.start);

    // Determine which cell boundaries the line will cross (N or S, E or W).
    int north = rise >= 0 ? 2 : 0;
    int east = run >= 0 ? 2 : 0;

    // And work out the appropriate delta from the start of the line to the
    // crossing boundary.
    double cellsize_x = grid_to_world.a;
    double cellsize_y = grid_to_world.e;
    double dN = north ? cellsize_y - delta.y : -delta.y;
    double dE = east ? cellsize_x - delta.x : -delta.x;

    // Calculate the positions at which the line crosses the grid graticules.
    geometry::Vec2<double> pN(dN * run / rise, dN);
    geometry::Vec2<double> pE(dE, dE * rise / run);

    // Create a vector of grid / graticule crossings to return to the caller.
    std::vector<geometry::Vec2<double>> crossings;

    // Append the start point of the line to the vector of crossings.
    crossings.push_back(line.start);

    // As long as there is a crossing point BEFORE the end of the line, we can
    // keep looping.
    while (pE.length() < length || pN.length() < length) {
      // Add the closest crossing point to the vector of grid / graticule
      // crossings.
      if (pE.length() < pN.length()) {
        crossings.push_back(line.start + pE);
        // Update the distance to the next graticule.
        dE += double(east - 1) * cellsize_x;
        // Calculate the position of the crossing point on the next grid /
        // graticule line.
        pE = geometry::Vec2<double>(dE, dE * rise / run);
      } else {
        crossings.push_back(line.start + pN);
        // Update the distance to the next graticule.
        dN += double(north - 1) * cellsize_y;
        // Calculate the position of the crossing point on the next grid /
        // graticule line.
        pN = geometry::Vec2<double>(dN * run / rise, dN);
      }
    }

    // Return the vector of grid / graticule crossing points that exist bbetween
    // the start and end of the line.
    return crossings;
  }

};

#endif // GRID_H
