#ifndef OPERATIONS_H
#define OPERATIONS_H

#include <vector>
#include "geometry.hpp"
#include "grid.hpp"

namespace snail {
namespace operations {

/// Test if a coordinate falls within grid bounds
bool pointInBounds(const geometry::Coord &pt, const grid::Grid &raster);

/// A run of coordinates someone else owns.
///
/// Splitting reads its input once and keeps no reference to it, so it has no
/// need of a container. Taking a span instead lets a caller whose coordinates
/// already sit in a buffer - the interleaved xy buffer of a GeoArrow array,
/// which is laid out exactly as a vector of Coord would be - hand them over
/// without copying them into one first.
struct CoordSpan {
  const geometry::Coord *data = nullptr;
  std::size_t count = 0;

  CoordSpan() = default;
  CoordSpan(const geometry::Coord *data, std::size_t count)
      : data(data), count(count) {}
  /// implicit, so a caller that does hold a vector passes it unchanged
  CoordSpan(const std::vector<geometry::Coord> &coordinates)
      : data(coordinates.data()), count(coordinates.size()) {}

  std::size_t size() const { return count; }
  bool empty() const { return count == 0; }
  const geometry::Coord &operator[](std::size_t i) const { return data[i]; }
  const geometry::Coord *begin() const { return data; }
  const geometry::Coord *end() const { return data + count; }
};

/// Linestring pieces held as flat arrays: every piece's points concatenated
/// in order, piece p spanning [offsets[p], offsets[p + 1]). The offsets
/// carry a closing entry, so there is one more of them than there are
/// pieces. As with PolygonPieces, this is the layout shapely consumes.
///
/// Offsets are 32-bit signed because that is what a plain Arrow list takes:
/// holding them in the width they will be read at means a whole batch of
/// pieces can be handed to Arrow as it stands, rather than converted one at
/// a time on the way out.
struct LinePieces {
  std::vector<geometry::Coord> coordinates;
  std::vector<int32_t> offsets;

  LinePieces() : offsets{0} {}

  std::size_t size() const { return offsets.size() - 1; }

  /// Close off the piece whose points have just been appended
  void endPiece() { offsets.push_back(static_cast<int32_t>(coordinates.size())); }
};

/// Split a linestring along the lines of a raster grid, so that each piece
/// lies within a single grid cell. Consecutive pieces meet at the point
/// where the line leaves one cell for the next, and together they are the
/// line: the split conserves its length and its endpoints.
///
/// Pieces are *appended* to `out`, and their offsets are recorded against
/// whatever it already holds, so splitting a whole column of linestrings
/// into one `LinePieces` needs no concatenating afterwards. A line that
/// degenerates to fewer than two distinct points appends nothing.
void splitLineStringGrid(CoordSpan coordinates, const grid::Grid &grid,
                         bool bounded, LinePieces &out);

/// As above, for a caller that wants only this line's pieces
LinePieces splitLineStringGrid(CoordSpan coordinates, const grid::Grid &grid,
                               bool bounded = false);

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
/// Offsets are 32-bit signed for the same reason as LinePieces'.
struct PolygonPieces {
  std::vector<geometry::Coord> coordinates;
  std::vector<int32_t> ring_offsets;
  std::vector<int32_t> polygon_offsets;

  PolygonPieces() : ring_offsets{0}, polygon_offsets{0} {}

  std::size_t size() const { return polygon_offsets.size() - 1; }

  /// Close off the ring whose points have just been appended
  void endRing() {
    ring_offsets.push_back(static_cast<int32_t>(coordinates.size()));
  }

  /// Close off the polygon whose rings have just been ended
  void endPolygon() {
    polygon_offsets.push_back(static_cast<int32_t>(ring_offsets.size() - 1));
  }
};

/// Split a valid polygon along the lines of a raster grid.
///
/// Rings are passed exterior first, then any interior rings (holes); each
/// ring may be closed or open (closure is applied) and in either winding
/// order (orientation is normalised internally). Produces one polygon piece
/// per covered region of a grid cell; every piece lies within a single cell.
///
/// As with linestrings, pieces are *appended* to `out` with their offsets
/// recorded against what it already holds. A degenerate polygon appends
/// nothing.
void splitPolygonGridPieces(const std::vector<CoordSpan> &rings,
                            const grid::Grid &grid, PolygonPieces &out);

/// As above, for a caller that wants only this polygon's pieces
PolygonPieces splitPolygonGridPieces(const std::vector<CoordSpan> &rings,
                                     const grid::Grid &grid);

/// As above, taking rings that are already in their own vectors
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
