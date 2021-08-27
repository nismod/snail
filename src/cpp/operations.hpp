#include <vector>
#include "geometry.hpp"
#include "grid.hpp"

namespace snail {
namespace operations {

std::vector<std::vector<geometry::Vec2<double>>>
    findIntersectionsLineString(geometry::LineString<double>, grid::Grid);
std::vector<std::vector<geometry::Vec2<double>>>
splitAlongGridlines(std::vector<geometry::Vec2<double>>, int, int, std::string,
                    grid::Grid);

} // namespace operations
} // namespace snail
