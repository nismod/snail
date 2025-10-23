#ifndef OPERATIONS_H
#define OPERATIONS_H

#include <vector>
#include "geometry.hpp"
#include "grid.hpp"

namespace snail {
namespace operations {

enum class Direction { horizontal, vertical };

std::vector<std::vector<geometry::Coord>>
findIntersectionsLineString(geometry::LineString, grid::Grid,
                            bool bounded = false);
std::vector<std::vector<geometry::Coord>>
splitAlongGridlines(std::vector<geometry::Coord>, int, int, Direction,
                    grid::Grid);

bool pointInBounds(const geometry::Coord &pt, const grid::Grid &raster);

} // namespace operations
} // namespace snail

#endif // OPERATIONS_H
