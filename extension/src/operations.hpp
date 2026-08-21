#ifndef OPERATIONS_H
#define OPERATIONS_H

#include <vector>
#include "geometry.hpp"
#include "grid.hpp"

namespace snail {
namespace operations {

/// Find intersection points of a linestring with a raster grid
std::vector<std::vector<geometry::Coord>> findIntersectionsLineString(geometry::LineString, grid::Grid,
                                                                      bool bounded = false);

/// Split a valid polygon along the lines of a raster grid
std::vector<geometry::Polygon> splitPolygonGrid(const std::vector<std::vector<geometry::Coord>> &rings,
                                                const grid::Grid &grid);

bool pointInBounds(const geometry::Coord &pt, const grid::Grid &raster);

} // namespace operations
} // namespace snail

#endif // OPERATIONS_H
