#ifndef OPERATIONS_H
#define OPERATIONS_H

#include <vector>
#include "geometry.hpp"
#include "grid.hpp"

namespace snail {
namespace operations {

enum class Direction { horizontal, vertical };

std::vector<std::vector<geometry::Coord>>
    findIntersectionsLineString(geometry::LineString, grid::Grid);
std::vector<std::vector<geometry::Coord>>
splitAlongGridlines(std::vector<geometry::Coord>, int, int, Direction,
                    grid::Grid);

} // namespace operations
} // namespace snail

#endif // OPERATIONS_H
