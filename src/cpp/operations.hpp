#ifndef OPERATIONS_H
#define OPERATIONS_H

#include <vector>
#include "geometry.hpp"
#include "grid.hpp"

namespace snail {
namespace operations {

enum class Direction { horizontal, vertical };

std::vector<std::vector<geometry::Vec2<double>>>
    findIntersectionsLineString(geometry::LineString<double>, grid::Grid);
std::vector<std::vector<geometry::Vec2<double>>>
splitAlongGridlines(std::vector<geometry::Vec2<double>>, int, int, Direction,
                    grid::Grid);

} // namespace operations
} // namespace snail

#endif // OPERATIONS_H
