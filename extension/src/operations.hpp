#ifndef OPERATIONS_H
#define OPERATIONS_H

#include <vector>
#include "geometry.hpp"
#include "grid.hpp"

namespace snail {
namespace operations {

/// Test if a coordinate falls within grid bounds
bool pointInBounds(const geometry::Coord &pt, const grid::Grid &raster);

/// Linestring pieces held as flat arrays: every piece's points concatenated
/// in order, piece p spanning [offsets[p], offsets[p + 1]). The offsets
/// carry a closing entry, so there is one more of them than there are
/// pieces. As with PolygonPieces, this is the layout shapely consumes.
struct LinePieces {
  std::vector<geometry::Coord> coordinates;
  std::vector<std::size_t> offsets;

  LinePieces() : offsets{0} {}

  std::size_t size() const { return offsets.size() - 1; }

  /// Close off the piece whose points have just been appended
  void endPiece() { offsets.push_back(coordinates.size()); }
};

/// Split a linestring along the lines of a raster grid, so that each piece
/// lies within a single grid cell. Consecutive pieces meet at the point
/// where the line leaves one cell for the next, and together they are the
/// line: the split conserves its length and its endpoints.
LinePieces splitLineStringGrid(const std::vector<geometry::Coord> &coordinates,
                               const grid::Grid &grid, bool bounded = false);

/// Polygon pieces held as flat arrays: every ring's points concatenated in
/// order, with the rings of a polygon contiguous and its exterior ring
/// first. Ring r spans coordinates [ring_offsets[r], ring_offsets[r + 1]);
/// polygon p spans rings [polygon_offsets[p], polygon_offsets[p + 1]).
/// Both offset arrays carry a closing entry, so they are one longer than
/// the number of rings and of polygons respectively.
///
/// This is the layout shapely and GeoArrow consume directly, and it avoids
/// allocating per piece: a polygon split over a large grid is mostly whole
/// cells, and those are appended here without allocating at all.
struct PolygonPieces {
  std::vector<geometry::Coord> coordinates;
  std::vector<std::size_t> ring_offsets;
  std::vector<std::size_t> polygon_offsets;

  PolygonPieces() : ring_offsets{0}, polygon_offsets{0} {}

  std::size_t size() const { return polygon_offsets.size() - 1; }

  /// Close off the ring whose points have just been appended
  void endRing() { ring_offsets.push_back(coordinates.size()); }

  /// Close off the polygon whose rings have just been ended
  void endPolygon() { polygon_offsets.push_back(ring_offsets.size() - 1); }
};

/// Split a valid polygon along the lines of a raster grid.
///
/// Rings are passed exterior first, then any interior rings (holes); each
/// ring may be closed or open (closure is applied) and in either winding
/// order (orientation is normalised internally). Returns one polygon piece
/// per covered region of a grid cell; every piece lies within a single cell.
PolygonPieces
splitPolygonGridPieces(const std::vector<std::vector<geometry::Coord>> &rings,
                       const grid::Grid &grid);

/// As splitPolygonGridPieces, returning each piece as its own Polygon
std::vector<geometry::Polygon>
splitPolygonGrid(const std::vector<std::vector<geometry::Coord>> &rings,
                 const grid::Grid &grid);

} // namespace operations
} // namespace snail

#endif // OPERATIONS_H
