#ifndef OPERATIONS_H
#define OPERATIONS_H

#include <vector>
#include "geometry.hpp"
#include "grid.hpp"

namespace snail {
namespace operations {

std::vector<std::vector<geometry::Coord>>
    findIntersectionsLineString(geometry::LineString, grid::Grid);

/// Split a valid polygon along the lines of a raster grid.
///
/// Rings are passed exterior first, then any interior rings (holes); each
/// ring may be closed or open (closure is applied) and in either winding
/// order (orientation is normalised internally). Returns one polygon piece
/// per covered region of a grid cell; every piece lies within a single cell.
std::vector<geometry::Polygon>
splitPolygonGrid(const std::vector<std::vector<geometry::Coord>> &rings,
                 const grid::Grid &grid);

} // namespace operations
} // namespace snail

#endif // OPERATIONS_H
