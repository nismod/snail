#include <algorithm> // sort, reverse, clamp
#include <cmath>     // floor, ceil, fabs, fmod, hypot, round
#include <cstddef>
#include <limits>
#include <optional>
#include <utility> // pair
#include <vector>

#include "geometry.hpp"
#include "grid.hpp"
#include "operations.hpp"

namespace snail::operations {

using geometry::Coord;
using geometry::Ring;


/// Does any part of a straight line segment intersect the grid extent, either
/// partially (i.e. crossing one or more of bounding box lines) or entirely.
bool segmentIntersectsGridBounds(const geometry::Line &line, const grid::Grid &raster) {
  // If either endpoint or both falls within the grid extent, the line segment
  // intersects
  auto xy0 = line.start;
  auto xy1 = line.end;
  if (pointInBounds(xy0, raster) || pointInBounds(xy1, raster)) {
    return true;
  }

  // Pick out grid bounding box coordinates
  auto ll = raster.ll;
  auto ur = raster.ur;
  const double xmin = ll.x;
  const double ymin = ll.y;
  const double xmax = ur.x;
  const double ymax = ur.y;

  // Pick out line endpoint coordinates
  const double x0 = xy0.x;
  const double y0 = xy0.y;
  const double x1 = xy1.x;
  const double y1 = xy1.y;
  const double line_xmin = std::min(x0, x1);
  const double line_xmax = std::max(x0, x1);
  const double line_ymin = std::min(y0, y1);
  const double line_ymax = std::max(y0, y1);

  auto crosses_x = [&](const double &xtest) {
    // Does line cross the vertical ray at xtest?
    if (line_xmin <= xtest && xtest <= line_xmax) {
      // If line is vertical, does it overlap y range?
      if (line_xmin == line_xmax) {
        return std::max(line_ymin, ymin) <= std::min(line_ymax, ymax);
      }
      // Else does the crossing point overlap y range?
      const double ycross = y0 + (xtest - x0) * (y1 - y0) / (x1 - x0);
      return ymin <= ycross && ycross <= ymax;
    }
    return false;
  };

  auto crosses_y = [&](const double &ytest) {
    // Does line cross the horizontal ray at ytest?
    if (line_ymin <= ytest && ytest <= line_ymax) {
      // If line is horizontal, does it overlap x range?
      if (line_ymin == line_ymax) {
        return std::max(line_xmin, xmin) <= std::min(line_xmax, xmax);
      }
      // Else does the crossing point overlap x range?
      const double xcross = x0 + (ytest - y0) * (x1 - x0) / (y1 - y0);
      return xmin <= xcross && xcross <= xmax;
    }
    return false;
  };

  return crosses_x(xmin) || crosses_x(xmax) || crosses_y(ymin) || crosses_y(ymax);
}

bool pointInBounds(const geometry::Coord &pt, const grid::Grid &raster) {
  auto ll = raster.ll;
  auto ur = raster.ur;

  return pt.x >= ll.x && pt.x <= ur.x && pt.y >= ll.y && pt.y <= ur.y;
};

using linestr = std::vector<geometry::Coord>;

// The machinery below splits a polygon along the lines of a raster grid.
// All of the work happens in grid coordinates, where the regular grid makes
// cell borders exact: each ring vertex is transformed to grid space once
// (and snapped onto a grid line where it lies within tolerance), so that a
// cell is the half-open unit square [i, i+1) x [j, j+1), the cell index of
// a point is simply floor, a point lies on a grid line exactly when a
// coordinate is an integer, crossing points sit exactly on integer
// boundaries and cell corners are exact integer pairs. Results are
// transformed back to world coordinates in a single output pass.
//
// The approach:
// - split each ring at every grid line crossing, breaking also at vertices
//   which lie on a grid line, giving ring pieces that each lie within a
//   single cell, with endpoints on the cell border
// - bucket the pieces by cell
// - within each cell, trace the boundary of each covered region: follow a
//   ring piece from where it enters the cell to where it leaves, then walk
//   along the cell border (passing corners) to the next piece, until the
//   ring closes (Weiler-Atherton clipping, specialised to a grid cell)
// - cells wholly inside the polygon (no boundary pieces) become full cell
//   boxes, found with a parity-safe scanline through each row of cell
//   centres
//
// N.B. grid transforms are assumed axis-aligned (no shear or rotation, i.e.
// transform b and d coefficients are zero), as elsewhere in this library.
namespace {

/// Grid-space coordinates within this distance of a grid line are snapped
/// onto it: comfortably above the noise of the world-to-grid transform (the
/// relative epsilon of world coordinates, in cell units), far below
/// coordinate data precision
constexpr double SNAP_TOL = 1e-9;

/// Tolerance when comparing positions along a cell border (the border
/// parameter runs from 0 to 4, one unit per cell edge)
constexpr double BORDER_PARAM_TOL = 1e-9;

/// Snap a grid-space coordinate onto the nearest grid line when it lies
/// within SNAP_TOL of it
double snapped(double v) {
  double r = std::round(v);
  return (std::fabs(v - r) < SNAP_TOL) ? r : v;
}

/// After snapping, a point lies on a grid line exactly when a coordinate is
/// an integer
bool isInteger(double v) { return v == std::floor(v); }

/// Twice the signed area of a closed ring (positive if counter-clockwise).
/// Coordinates are taken relative to the first point: without this, rings
/// that are small relative to their distance from the origin (e.g. buildings
/// in geographic coordinates) lose all precision to cancellation.
double ringSignedArea2(const Coord *ring, std::size_t count) {
  if (count == 0) {
    return 0;
  }
  const Coord origin = ring[0];
  double area2 = 0;
  for (std::size_t i = 0; i + 1 < count; i++) {
    area2 += (ring[i].x - origin.x) * (ring[i + 1].y - origin.y) -
             (ring[i + 1].x - origin.x) * (ring[i].y - origin.y);
  }
  return area2;
}

double ringSignedArea2(const linestr &ring) {
  return ringSignedArea2(ring.data(), ring.size());
}

/// Even-odd test for a point inside a closed ring. Points on the boundary
/// may be classified either way.
bool pointInRing(const Coord p, const Coord *ring, std::size_t count) {
  bool inside = false;
  for (std::size_t i = 0; i + 1 < count; i++) {
    const Coord a = ring[i];
    const Coord b = ring[i + 1];
    // Half-open rule: each edge counts one of its endpoints only, so
    // crossing counts stay consistent however edges meet at shared vertices.
    if ((a.y < p.y) != (b.y < p.y)) {
      double t = (p.y - a.y) / (b.y - a.y);
      if (a.x + t * (b.x - a.x) > p.x) {
        inside = !inside;
      }
    }
  }
  return inside;
}

/// A cell's position in the grid
struct CellIndex {
  long column = 0;
  long row = 0;

  bool operator==(const CellIndex &other) const {
    return column == other.column && row == other.row;
  }
  bool operator!=(const CellIndex &other) const { return !(*this == other); }
};

/// A piece of a polygon ring lying within a single grid cell, as a span of
/// the arena its points were written to. Owning nothing keeps a piece
/// cheap to sort and keeps the points of neighbouring pieces together in
/// memory, which is how they are walked.
struct Piece {
  std::size_t begin = 0;
  std::size_t end = 0;
  /// closed pieces are rings that never cross a grid line (or loops pinched
  /// at a single border point); open pieces are chains crossing the cell
  bool closed = false;
  /// twice the signed area (meaningful for closed pieces)
  double area2 = 0;
  /// where the chain meets the cell border, as positions along the walk of
  /// it (open pieces only)
  double enters_at = 0;
  double leaves_at = 0;
  bool used = false;
  /// the cell this piece lies in
  CellIndex cell{0, 0};

  std::size_t size() const { return end - begin; }
};

/// Order pieces by the row, then the column, of the cell they lie in, so
/// that the pieces of a cell end up adjacent and the cells themselves in
/// the order the interior scan sweeps them. Pieces of the same cell keep
/// the order the rings were walked in, which the trace depends on; ordering
/// by that as a last resort makes this a total order, so an ordinary sort
/// gives the same answer as a stable one without the buffer a stable sort
/// has to allocate.
bool rowMajorOrder(const Piece &a, const Piece &b) {
  if (a.cell.row != b.cell.row) {
    return a.cell.row < b.cell.row;
  }
  if (a.cell.column != b.cell.column) {
    return a.cell.column < b.cell.column;
  }
  return a.begin < b.begin;
}

/// A cell the polygon boundary passes through, and whether that boundary
/// crosses the horizontal line through the cell centre an odd number of
/// times - which is what tells the interior scan to change its
/// inside/outside status as it passes this cell
struct BoundaryCell {
  CellIndex cell;
  bool crossed;
};

/// A ring traced within a cell: where its points sit in the cell's arena,
/// and its signed area, whose sign says whether it bounds a covered region
/// (counter-clockwise) or a hole (clockwise)
struct CellRing {
  std::size_t begin;
  std::size_t end;
  double area2;
};

/// A point on a cell's border, at a position along the counter-clockwise
/// walk of it. Used both for where chains meet the border and for the
/// corners a walk passes.
struct BorderPoint {
  double reach;
  Coord point;
};

/// An interior ring of a cell, with a point just inside it - which decides
/// the region it belongs to - and whether that region has been found
struct CellHole {
  std::size_t ring;
  Coord inside;
  bool assigned;
};

/// Border geometry of a single grid cell: integer corner coordinates and a
/// counter-clockwise border parameterisation, in grid space
struct CellBorder {
  long column;
  long row;

  explicit CellBorder(const CellIndex index)
      : column(index.column), row(index.row) {}

  /// Grid coordinates of corner k (0 <= k < 4) along the counter-clockwise
  /// border walk (0,0) -> (1,0) -> (1,1) -> (0,1)
  Coord corner(int corner) const {
    static const int du[4] = {0, 1, 1, 0};
    static const int dv[4] = {0, 0, 1, 1};
    return Coord(column + du[corner], row + dv[corner]);
  }

  /// Direction of the counter-clockwise border walk at parameter t
  static Coord direction(double t) {
    switch (static_cast<int>(std::floor(t))) {
    case 0:
      return Coord(1, 0); // bottom edge, left to right
    case 1:
      return Coord(0, 1); // right edge, upwards
    case 2:
      return Coord(-1, 0); // top edge, right to left
    default:
      return Coord(0, -1); // left edge, downwards
    }
  }

  /// Position of a (near-border) grid-space point along the border walk, in
  /// [0, 4): the nearest point of the border, one unit per cell edge
  double parameter(const Coord g) const {
    const double u = std::clamp(g.x - column, 0.0, 1.0);
    const double v = std::clamp(g.y - row, 0.0, 1.0);
    double distance = v;
    double t = u; // bottom edge
    if (1 - u < distance) {
      distance = 1 - u;
      t = 1 + v; // right edge
    }
    if (1 - v < distance) {
      distance = 1 - v;
      t = 2 + (1 - u); // top edge
    }
    if (u < distance) {
      t = 3 + (1 - v); // left edge
    }
    return (t >= 4.0) ? t - 4.0 : t;
  }
};

/// The angle swept turning clockwise from direction `from` to direction
/// `to`, in (0, 2*PI]. Directions which double back exactly (a zero turn)
/// are reported as a full turn, so that retracing the way we came is only
/// ever chosen as a last resort.
double clockwiseTurn(const Coord from, const Coord to) {
  const double turn = std::atan2(from.y, from.x) - std::atan2(to.y, to.x);
  // both angles lie in (-PI, PI], so the difference needs at most one turn
  // added to bring it into range
  return (turn <= BORDER_PARAM_TOL) ? turn + 2 * utils::PI : turn;
}

/// Index of the longest segment of a piece
std::size_t longestSegment(const Coord *points, std::size_t count) {
  std::size_t longest = 0;
  double longest_length2 = -1.0;
  for (std::size_t i = 0; i + 1 < count; i++) {
    double dx = points[i + 1].x - points[i].x;
    double dy = points[i + 1].y - points[i].y;
    double length2 = dx * dx + dy * dy;
    if (length2 > longest_length2) {
      longest_length2 = length2;
      longest = i;
    }
  }
  return longest;
}

/// The cell that a ring piece belongs to. Pieces lie within a single cell,
/// with only their endpoints (and vertices which touch a grid line) on the
/// cell border, so any point with both coordinates non-integer names the
/// cell exactly. For pieces that run along a grid line, take the cell on
/// the requested side of the direction of travel: side_sign +1 for the
/// left, -1 for the right.
CellIndex pieceCell(const Coord *points, std::size_t count, double side_sign) {
  // Any segment midpoint strictly inside a cell. Midpoints come first
  // because a chain's endpoints lie on the cell border by construction, so
  // for the common two-point chain a scan of the vertices cannot succeed.
  for (std::size_t i = 0; i + 1 < count; i++) {
    Coord mid((points[i].x + points[i + 1].x) / 2,
              (points[i].y + points[i + 1].y) / 2);
    if (!isInteger(mid.x) && !isInteger(mid.y)) {
      return CellIndex{static_cast<long>(std::floor(mid.x)),
                       static_cast<long>(std::floor(mid.y))};
    }
  }
  // else any vertex strictly inside a cell
  for (std::size_t k = 0; k < count; k++) {
    const Coord &point = points[k];
    if (!isInteger(point.x) && !isInteger(point.y)) {
      return CellIndex{static_cast<long>(std::floor(point.x)),
                       static_cast<long>(std::floor(point.y))};
    }
  }
  // else the piece runs along a grid line: take the cell on the requested
  // side of the direction of travel
  const Coord front = points[0];
  const Coord back = points[count - 1];
  double dx = back.x - front.x;
  double dy = back.y - front.y;
  double nx = -dy * side_sign;
  double ny = dx * side_sign;
  long column, row;
  if (dx == 0) {
    column = static_cast<long>(front.x) - (nx > 0 ? 0 : 1);
  } else {
    column = static_cast<long>(std::floor(std::min(front.x, back.x)));
  }
  if (dy == 0) {
    row = static_cast<long>(front.y) - (ny > 0 ? 0 : 1);
  } else {
    row = static_cast<long>(std::floor(std::min(front.y, back.y)));
  }
  return CellIndex{column, row};
}

/// Append a whole cell as one ring. This is the path for a cell covered
/// except for a hole in it; runs of covered cells go through
/// appendCellBoxRun instead.
void appendCellBoxRing(const CellIndex index, PolygonPieces &pieces) {
  const double x0 = static_cast<double>(index.column);
  const double y0 = static_cast<double>(index.row);
  const double x1 = x0 + 1;
  const double y1 = y0 + 1;
  pieces.coordinates.push_back(Coord(x0, y0));
  pieces.coordinates.push_back(Coord(x1, y0));
  pieces.coordinates.push_back(Coord(x1, y1));
  pieces.coordinates.push_back(Coord(x0, y1));
  pieces.coordinates.push_back(Coord(x0, y0));
  pieces.endRing();
}

/// Append every cell in a row from column `from` up to (not including)
/// `to`, which is how whole covered cells arrive: in runs between the
/// cells the boundary passes through. Growing the output once for the run
/// and writing straight into it beats appending cell by cell.
void appendCellBoxRun(long from, long to, long row, PolygonPieces &pieces) {
  const std::size_t count = static_cast<std::size_t>(to - from);
  const std::size_t base = pieces.coordinates.size();
  pieces.coordinates.resize(base + count * 5);
  Coord *out = pieces.coordinates.data() + base;
  const double y0 = static_cast<double>(row);
  const double y1 = y0 + 1;
  for (long k = from; k < to; k++) {
    const double x0 = static_cast<double>(k);
    const double x1 = x0 + 1;
    out[0] = Coord(x0, y0);
    out[1] = Coord(x1, y0);
    out[2] = Coord(x1, y1);
    out[3] = Coord(x0, y1);
    out[4] = Coord(x0, y0);
    out += 5;
    pieces.ring_offsets.push_back(base +
                                  static_cast<std::size_t>(k - from + 1) * 5);
    pieces.polygon_offsets.push_back(pieces.ring_offsets.size() - 1);
  }
}

/// A point just inside a closed ring piece: the midpoint of its longest
/// segment, nudged towards the ring's own interior
Coord ringInsidePoint(const Coord *points, std::size_t count, double area2) {
  std::size_t seg = longestSegment(points, count);
  Coord a = points[seg];
  Coord b = points[seg + 1];
  Coord mid((a.x + b.x) / 2, (a.y + b.y) / 2);
  double dx = b.x - a.x;
  double dy = b.y - a.y;
  double length = std::hypot(dx, dy);
  if (length == 0) {
    return mid;
  }
  // the ring interior is on the left of travel if counter-clockwise
  double side = (area2 > 0) ? 1.0 : -1.0;
  double delta = 1e-6 * length;
  return Coord(mid.x - dy / length * side * delta,
               mid.y + dx / length * side * delta);
}

/// Call `emit` at each point where the segment from a to b crosses a grid
/// line, strictly between the two, in order along the segment. Both
/// coordinates of a crossing are exact: the crossed one is the integer
/// level itself, and the other is snapped when it lands on a grid line
/// too, so that a crossing through a cell corner is a single point.
///
/// The direction of travel is resolved once, up front, and the number of
/// crossings on each axis is known before the walk, so the loop is a merge
/// of two monotone sequences with no per-crossing sign tests.
template <class Emit>
void forEachCrossing(const Coord a, const Coord b, Emit &&emit) {
  const double dx = b.x - a.x;
  const double dy = b.y - a.y;

  // Levels strictly between a and b on each axis, counted up front. Going
  // up, they run from floor(a)+1 while below b; going down, from
  // ceil(a)-1 while above b. Using counts rather than a comparison per
  // step keeps the loop free of sign branches.
  const double sx = (dx >= 0) ? 1.0 : -1.0;
  const double sy = (dy >= 0) ? 1.0 : -1.0;
  double kx = (dx >= 0) ? std::floor(a.x) + 1 : std::ceil(a.x) - 1;
  double ky = (dy >= 0) ? std::floor(a.y) + 1 : std::ceil(a.y) - 1;
  long nx = (dx == 0)
                ? 0
                : static_cast<long>(std::max(
                      0.0, sx * (b.x - kx) + (isInteger(b.x) ? 0.0 : 1.0)));
  long ny = (dy == 0)
                ? 0
                : static_cast<long>(std::max(
                      0.0, sy * (b.y - ky) + (isInteger(b.y) ? 0.0 : 1.0)));

  // Divide rather than multiply by a reciprocal: the crossing positions
  // must come out bit for bit the same as they would from any other
  // segment meeting the same grid line, or neighbouring cells stop sharing
  // exactly the point between them.
  while (nx > 0 || ny > 0) {
    const double tx = (nx > 0) ? (kx - a.x) / dx : 2.0;
    const double ty = (ny > 0) ? (ky - a.y) / dy : 2.0;
    Coord crossing(0, 0);
    if (tx <= ty) {
      crossing = Coord(kx, snapped(a.y + tx * dy));
      kx += sx;
      nx--;
      // a crossing through a cell corner consumes a level on both axes
      if (ny > 0 && crossing.y == ky) {
        ky += sy;
        ny--;
      }
    } else {
      crossing = Coord(snapped(a.x + ty * dx), ky);
      ky += sy;
      ny--;
      if (nx > 0 && crossing.x == kx) {
        kx += sx;
        nx--;
      }
    }
    emit(crossing);
  }
}

/// Split a closed grid-space ring at every crossing with a grid line,
/// breaking also at any vertex which lies on a grid line (a crossing may
/// pass exactly through a vertex). Every returned piece lies within a
/// single cell. Breaking at a vertex which only touches a grid line is
/// harmless: the two chains reconnect seamlessly when the cell is
/// assembled.
void splitRing(const linestr &ring, std::vector<Coord> &arena,
               std::vector<Piece> &pieces) {
  std::size_t begin = arena.size();
  arena.push_back(ring.front());

  // close off the piece built so far, and start the next one at the point
  // the two share
  auto breakAt = [&](const Coord at) {
    if (!(arena.back() == at)) {
      arena.push_back(at);
    }
    if (arena.size() - begin >= 2) {
      Piece piece;
      piece.begin = begin;
      piece.end = arena.size();
      pieces.push_back(piece);
    }
    begin = arena.size();
    arena.push_back(at);
  };

  for (std::size_t i = 0; i + 1 < ring.size(); i++) {
    const Coord b = ring[i + 1];
    forEachCrossing(ring[i], b, breakAt);

    // the segment end: break here too if it lies on a grid line (unless it
    // is the ring's closing vertex, which ends the final piece anyway)
    if (i + 2 < ring.size() && (isInteger(b.x) || isInteger(b.y))) {
      breakAt(b);
    } else if (!(arena.back() == b)) {
      arena.push_back(b);
    }
  }
  if (arena.size() - begin >= 2) {
    Piece piece;
    piece.begin = begin;
    piece.end = arena.size();
    pieces.push_back(piece);
  } else {
    arena.erase(arena.begin() + begin, arena.end());
  }
}

/// Scratch space for a split, reused from cell to cell and from polygon to
/// polygon. Cells are small and there are many of them, so the buffers a
/// cell needs are worth keeping rather than allocating and freeing each
/// time; every member here is cleared before use, never freed.
struct Workspace {
  // per split
  std::vector<linestr> rings;
  /// the points of every ring piece, laid end to end
  std::vector<Coord> pieces;
  std::vector<Piece> bucketed;
  /// pieces of the ring currently being split, before they are bucketed
  std::vector<Piece> raw;
  std::vector<BoundaryCell> boundary;
  // per cell
  std::vector<Piece *> chains;
  std::vector<BorderPoint> endpoints;
  std::vector<Piece *> coincident;
  std::vector<BorderPoint> passed;
  /// traced rings, laid end to end, with a span and signed area each
  std::vector<Coord> arena;
  std::vector<CellRing> cell_rings;
  std::vector<CellHole> holes;
  /// scratch for the rare ring that pinches and has to be taken apart
  linestr pinch;
};

/// The first pair of equal vertices pinching a ring into two loops, if
/// there is one. The closing pair does not count: a ring meets itself
/// there by definition.
std::optional<std::pair<std::size_t, std::size_t>>
findPinch(const Coord *points, std::size_t count) {
  for (std::size_t i = 0; i + 1 < count; i++) {
    for (std::size_t j = i + 2; j < count; j++) {
      if (!(i == 0 && j == count - 1) && points[i] == points[j]) {
        return std::make_pair(i, j);
      }
    }
  }
  return std::nullopt;
}

/// Record a ring already sitting at the end of the cell's arena
void recordRing(std::size_t begin, Workspace &work) {
  work.cell_rings.push_back(CellRing{
      begin, work.arena.size(),
      ringSignedArea2(work.arena.data() + begin, work.arena.size() - begin)});
}

/// Copy a ring into the cell's arena, splitting it at any pinch vertex so
/// that regions meeting at a single point come out as separate rings
void appendSimpleRings(const Coord *points, std::size_t count,
                       Workspace &work) {
  if (count < 4) {
    return;
  }
  if (const auto pinch = findPinch(points, count)) {
    const auto [i, j] = *pinch;
    // the loop from i to j, and the ring with that loop taken out
    appendSimpleRings(points + i, j - i + 1, work);
    linestr rest(points, points + i + 1);
    rest.insert(rest.end(), points + j + 1, points + count);
    appendSimpleRings(rest.data(), rest.size(), work);
    return;
  }
  const std::size_t begin = work.arena.size();
  work.arena.insert(work.arena.end(), points, points + count);
  recordRing(begin, work);
}

/// Assemble the polygon pieces covering a single cell from the ring pieces
/// that were bucketed to it, appending to results. The polygon interior lies
/// on the left of each piece (rings oriented exterior counter-clockwise,
/// holes clockwise), and the cell border is walked counter-clockwise, so
/// each covered region is traced by alternately following pieces and border
/// arcs.
void assembleCell(const CellIndex index, Piece *first, Piece *last,
                  const Coord *points, Workspace &work,
                  PolygonPieces &results) {
  CellBorder border(index);

  work.chains.clear();
  work.endpoints.clear();
  work.arena.clear();
  work.cell_rings.clear();

  for (Piece *it = first; it != last; ++it) {
    Piece &piece = *it;
    if (!piece.closed) {
      work.chains.push_back(&piece);
    } else if (piece.area2 != 0) {
      // a ring that never leaves the cell stands as it is: a piece of its
      // own if counter-clockwise, an interior ring if clockwise
      appendSimpleRings(points + piece.begin, piece.size(), work);
    }
    // pieces with exactly zero area are degenerate - ignored
  }

  // Chain endpoints, so border walks can be noded wherever the polygon
  // boundary touches the border
  for (Piece *chain : work.chains) {
    work.endpoints.push_back({chain->enters_at, points[chain->begin]});
    work.endpoints.push_back({chain->leaves_at, points[chain->end - 1]});
  }

  // Weld points within floating-point noise of each other as rings are
  // built: a polygon vertex on a grid line and the computed crossing point
  // there may differ in the last few bits, and keeping both can make a ring
  // fold back on itself
  std::size_t ring_begin = 0;
  auto append_point = [&work, &ring_begin](const Coord &point) {
    if (work.arena.size() > ring_begin) {
      const Coord &back = work.arena.back();
      if (std::fabs(back.x - point.x) < SNAP_TOL &&
          std::fabs(back.y - point.y) < SNAP_TOL) {
        return;
      }
    }
    work.arena.push_back(point);
  };

  // Trace rings bounded by chains and border arcs
  for (Piece *start : work.chains) {
    if (start->used) {
      continue;
    }
    ring_begin = work.arena.size();
    Piece *current = start;
    // each iteration consumes a chain, so this terminates
    while (true) {
      current->used = true;
      for (std::size_t k = current->begin; k < current->end; k++) {
        append_point(points[k]);
      }

      // Find the next chain entry, walking counter-clockwise around the
      // border from the current exit; the ring closes when the nearest
      // entry is the starting chain's.
      const double at = current->leaves_at;
      Piece *next = nullptr;
      double best_reach = 5.0;
      double start_reach = 5.0;
      work.coincident.clear();
      for (Piece *candidate : work.chains) {
        if (candidate->used && candidate != start) {
          continue;
        }
        double reach = std::fmod(candidate->enters_at - at, 4.0);
        if (reach < 0) {
          reach += 4.0;
        }
        // an entry point coinciding with the exit is at no distance at all,
        // not a full lap
        if (reach > 4.0 - BORDER_PARAM_TOL) {
          reach = 0.0;
        }
        if (reach < BORDER_PARAM_TOL) {
          // This chain begins exactly where we are: the boundary touches
          // the cell border here without crossing it. Whether to follow the
          // chain or carry on along the border is decided below, by which
          // of them comes first turning clockwise away from the direction
          // we arrived from - the rule which keeps the covered area on our
          // left, as it is along every chain.
          work.coincident.push_back(candidate);
          if (candidate == start) {
            // the ring closes here, at no distance along the border
            start_reach = 0.0;
          }
          continue;
        }
        if (reach < best_reach) {
          best_reach = reach;
          next = candidate;
        }
        if (candidate == start) {
          start_reach = reach;
        }
      }
      // when closing the ring ties with entering another chain, close
      if (start_reach <= best_reach + BORDER_PARAM_TOL) {
        best_reach = start_reach;
        next = start;
      }

      if (!work.coincident.empty()) {
        // direction we arrived from, reversed
        const Coord *arriving = points + current->begin;
        const std::size_t arrived = current->size();
        Coord back(arriving[arrived - 2].x - arriving[arrived - 1].x,
                   arriving[arrived - 2].y - arriving[arrived - 1].y);
        // Carrying on along the border is the alternative to beat. A chain
        // ties with it when the chain runs along the border itself, and
        // then the chain must win: it is the same path, and leaving it
        // untraced would strand it as a piece of its own.
        double least_turn =
            clockwiseTurn(back, CellBorder::direction(at)) + BORDER_PARAM_TOL;
        for (Piece *candidate : work.coincident) {
          Coord away(
              points[candidate->begin + 1].x - points[candidate->begin].x,
              points[candidate->begin + 1].y - points[candidate->begin].y);
          double turn = clockwiseTurn(back, away);
          if (turn < least_turn) {
            least_turn = turn;
            next = candidate;
            best_reach = 0.0;
          }
        }
      }

      // Walk the border, adding any corners passed on the way, and noding
      // at any other chain endpoints passed - so that regions which pinch
      // together at a point on the border can be separated below
      work.passed.clear();
      double corner_at = std::floor(at) + 1.0;
      for (int step = 0; step < 4; step++) {
        const double reach = corner_at - at;
        if (reach >= best_reach - BORDER_PARAM_TOL) {
          break;
        }
        if (reach > BORDER_PARAM_TOL) {
          work.passed.push_back({reach, border.corner(static_cast<int>(
                                            std::fmod(corner_at, 4.0)))});
        }
        corner_at += 1.0;
      }
      for (const BorderPoint &endpoint : work.endpoints) {
        double reach = std::fmod(endpoint.reach - at, 4.0);
        if (reach < 0) {
          reach += 4.0;
        }
        if (reach > BORDER_PARAM_TOL && reach < best_reach - BORDER_PARAM_TOL) {
          work.passed.push_back({reach, endpoint.point});
        }
      }
      std::sort(work.passed.begin(), work.passed.end(),
                [](const BorderPoint &a, const BorderPoint &b) {
                  return a.reach < b.reach;
                });
      for (const BorderPoint &passed : work.passed) {
        append_point(passed.point);
      }

      if (next == start) {
        break;
      }
      current = next;
    }

    // close the ring; it is already in the arena, so record it where it
    // lies unless it pinches
    if (work.arena.size() > ring_begin) {
      const Coord front = work.arena[ring_begin];
      if (std::fabs(work.arena.back().x - front.x) < SNAP_TOL &&
          std::fabs(work.arena.back().y - front.y) < SNAP_TOL) {
        work.arena.back() = front;
      } else {
        work.arena.push_back(front);
      }
    }
    const std::size_t count = work.arena.size() - ring_begin;
    if (count < 4) {
      work.arena.erase(work.arena.begin() + ring_begin, work.arena.end());
    } else if (findPinch(work.arena.data() + ring_begin, count)) {
      // rare: take it back out and re-append it as simple rings, which
      // cannot be done in place because the pieces overlap the original
      work.pinch.assign(work.arena.begin() + ring_begin, work.arena.end());
      work.arena.erase(work.arena.begin() + ring_begin, work.arena.end());
      appendSimpleRings(work.pinch.data(), count, work);
    } else {
      recordRing(ring_begin, work);
    }
  }

  // Gather the interior rings, with the point that decides which region
  // each belongs to. Most cells have none, and then none of the work below
  // happens at all; where there are some, each such point is worked out
  // once rather than once per region tested against.
  work.holes.clear();
  for (const CellRing &ring : work.cell_rings) {
    if (ring.area2 < 0) {
      work.holes.push_back(
          CellHole{static_cast<std::size_t>(&ring - work.cell_rings.data()),
                   ringInsidePoint(work.arena.data() + ring.begin,
                                   ring.end - ring.begin, ring.area2),
                   false});
    }
  }

  // Every counter-clockwise ring is a covered region; every clockwise one
  // is a hole in whichever of them contains it
  for (const CellRing &face : work.cell_rings) {
    if (face.area2 <= 0) {
      continue;
    }
    results.coordinates.insert(results.coordinates.end(),
                               work.arena.begin() + face.begin,
                               work.arena.begin() + face.end);
    results.endRing();
    for (CellHole &hole : work.holes) {
      if (hole.assigned) {
        continue;
      }
      const CellRing &ring = work.cell_rings[hole.ring];
      if (!pointInRing(hole.inside, work.arena.data() + face.begin,
                       face.end - face.begin)) {
        continue;
      }
      // the regions of one cell are disjoint, so the region a hole is
      // inside is the only one it can belong to
      hole.assigned = true;
      results.coordinates.insert(results.coordinates.end(),
                                 work.arena.begin() + ring.begin,
                                 work.arena.begin() + ring.end);
      results.endRing();
    }
    results.endPolygon();
  }

  // A hole with no region around it means the rest of the cell is covered
  for (const CellHole &hole : work.holes) {
    if (hole.assigned) {
      continue;
    }
    const CellRing &ring = work.cell_rings[hole.ring];
    appendCellBoxRing(index, results);
    results.coordinates.insert(results.coordinates.end(),
                               work.arena.begin() + ring.begin,
                               work.arena.begin() + ring.end);
    results.endRing();
    results.endPolygon();
  }
}

} // namespace

namespace {
/// Scratch kept alive between splits. The batch entry points reuse one, so
/// that splitting a table of features does not rebuild these buffers per
/// feature; a single split gets a fresh one and pays that cost once.
Workspace &splitWorkspace() {
  static thread_local Workspace work;
  return work;
}
} // namespace

PolygonPieces splitPolygonGridPieces(const std::vector<CoordSpan> &rings_in,
                                     const grid::Grid &grid) {
  PolygonPieces pieces;
  splitPolygonGridPieces(rings_in, grid, pieces);
  return pieces;
}

PolygonPieces splitPolygonGridPieces(const std::vector<linestr> &rings_in,
                                     const grid::Grid &grid) {
  std::vector<CoordSpan> spans(rings_in.begin(), rings_in.end());
  return splitPolygonGridPieces(spans, grid);
}

void splitPolygonGridPieces(const std::vector<CoordSpan> &rings_in,
                            const grid::Grid &grid, PolygonPieces &results) {
  // Pieces are appended, so the passes at the end of this function - back to
  // world coordinates, and the mirror correction - must touch only what this
  // call adds, not what a previous polygon left behind.
  const std::size_t coordinate_base = results.coordinates.size();
  const std::size_t ring_base = results.ring_offsets.size() - 1;

  Workspace &work = splitWorkspace();
  std::vector<linestr> &rings = work.rings;
  std::vector<Piece> &bucketed = work.bucketed;
  std::vector<BoundaryCell> &boundary = work.boundary;
  // the ring buffers keep their capacity from the previous split; only
  // their contents are dropped
  for (linestr &ring : rings) {
    ring.clear();
  }
  std::size_t used_rings = 0;
  bucketed.clear();
  boundary.clear();
  // Transform each ring to grid coordinates - once, up front - snapping
  // vertices onto grid lines where they lie within tolerance; drop
  // consecutive duplicate points, close each ring, and orient rings so that
  // the polygon interior is always on the left of the direction of travel
  // in grid space (exterior counter-clockwise, holes clockwise).
  for (std::size_t r = 0; r < rings_in.size(); r++) {
    if (used_rings == rings.size()) {
      rings.emplace_back();
    }
    linestr &ring = rings[used_rings];
    ring.reserve(rings_in[r].size() + 1);
    for (const Coord &point : rings_in[r]) {
      Coord g = grid.world_to_grid * point;
      g.x = snapped(g.x);
      g.y = snapped(g.y);
      if (ring.empty() || !(ring.back() == g)) {
        ring.push_back(g);
      }
    }
    if (!ring.empty() && !(ring.front() == ring.back())) {
      ring.push_back(ring.front());
    }
    bool is_exterior = (r == 0);
    double area2 = ring.size() >= 4 ? ringSignedArea2(ring) : 0.0;
    if (area2 == 0.0) {
      if (is_exterior) {
        // degenerate polygon
        return;
      }
      // degenerate hole
      ring.clear();
      continue;
    }
    if (is_exterior ? (area2 < 0) : (area2 > 0)) {
      std::reverse(ring.begin(), ring.end());
    }
    used_rings++;
  }
  if (used_rings == 0) {
    return;
  }

  // Split each ring at every grid line crossing and bucket the resulting
  // pieces by the cell they lie in
  std::vector<Coord> &arena = work.pieces;
  arena.clear();
  std::vector<Piece> &raw = work.raw;
  for (std::size_t r = 0; r < used_rings; r++) {
    raw.clear();
    splitRing(rings[r], arena, raw);
    if (raw.empty()) {
      continue;
    }

    // When the ring starts partway through a cell, the last and first
    // pieces are two halves of the same chain: join them, by copying the
    // first onto the end of the last. When the ring happens to start on a
    // grid line, keep the break (chains must only meet the cell border at
    // their endpoints).
    const Coord junction = rings[r].front();
    if (raw.size() >= 2 && !isInteger(junction.x) && !isInteger(junction.y) &&
        arena[raw.back().end - 1] == arena[raw.front().begin]) {
      Piece &last = raw.back();
      const Piece first = raw.front();
      arena.insert(arena.end(), arena.begin() + first.begin + 1,
                   arena.begin() + first.end);
      last.end = arena.size();
      raw.front() = last;
      raw.pop_back();
    }

    for (Piece piece : raw) {
      const Coord *points = arena.data() + piece.begin;
      const std::size_t count = piece.size();
      piece.closed = std::fabs(points[0].x - points[count - 1].x) < SNAP_TOL &&
                     std::fabs(points[0].y - points[count - 1].y) < SNAP_TOL;
      if (piece.closed) {
        // snap closed
        arena[piece.end - 1] = points[0];
        if (count < 4) {
          continue;
        }
        piece.area2 = ringSignedArea2(points, count);
        // bucket by the side the piece's own interior is on
        double side = (piece.area2 < 0) ? -1.0 : 1.0;
        piece.cell = pieceCell(points, count, side);
      } else {
        // an open chain: the polygon interior is on the left
        piece.cell = pieceCell(points, count, 1.0);
        CellBorder border(piece.cell);
        piece.enters_at = border.parameter(points[0]);
        piece.leaves_at = border.parameter(points[count - 1]);
      }
      bucketed.push_back(piece);
    }
  }

  // Gather the pieces of each cell together, in the order the interior
  // scan below sweeps them. Sorting once beats inserting each piece into an
  // ordered container, and a stable sort keeps the pieces of a cell in the
  // order the rings were traversed. The pieces of a cell are then a
  // contiguous run.
  std::sort(bucketed.begin(), bucketed.end(), rowMajorOrder);

  // Assemble the pieces covering each cell crossed by a ring, noting as we
  // go whether the boundary in the cell crosses the horizontal line through
  // the cell centre an odd number of times: a chain does so exactly when
  // its endpoints lie on opposite sides of that line, and a closed piece
  // never does.
  for (std::size_t start = 0; start < bucketed.size();) {
    const CellIndex cell = bucketed[start].cell;
    const double level = cell.row + 0.5;
    std::size_t end = start;
    bool crossed = false;
    while (end < bucketed.size() && bucketed[end].cell == cell) {
      const Piece &piece = bucketed[end];
      if (!piece.closed) {
        crossed = crossed != ((arena[piece.begin].y < level) !=
                              (arena[piece.end - 1].y < level));
      }
      end++;
    }
    assembleCell(cell, bucketed.data() + start, bucketed.data() + end,
                 arena.data(), work, results);
    boundary.push_back(BoundaryCell{cell, crossed});
    start = end;
  }

  // Cells wholly inside the polygon become full cell boxes. Scan each row
  // of boundary cells in column order, tracking whether the row of cells in
  // between is inside the polygon: the status can only change across a cell
  // the boundary passes through, and it changes there exactly when that
  // cell's crossings are odd. The half-open crossing rule makes the parity
  // along a whole row even, so every row ends outside.
  // count the interior cells first, so the output is grown once rather
  // than repeatedly as they are appended: for a large polygon they are
  // most of the result
  std::size_t interior = 0;
  for (std::size_t start = 0; start < boundary.size();) {
    const long row = boundary[start].cell.row;
    bool inside = false;
    long previous_column = 0;
    std::size_t at = start;
    for (; at < boundary.size() && boundary[at].cell.row == row; at++) {
      const long column = boundary[at].cell.column;
      if (inside && column > previous_column + 1) {
        interior += static_cast<std::size_t>(column - previous_column - 1);
      }
      inside = inside != boundary[at].crossed;
      previous_column = column;
    }
    start = at;
  }
  // reserve() takes an exact capacity rather than growing geometrically, so
  // asking for just what this polygon needs would recopy the whole of an
  // accumulator that already holds other polygons - once per polygon, which
  // is quadratic over a batch. Ask for at least double what is there.
  auto reserveAtLeast = [](auto &vec, std::size_t needed) {
    if (needed > vec.capacity()) {
      vec.reserve(std::max(needed, vec.capacity() * 2));
    }
  };
  reserveAtLeast(results.coordinates, results.coordinates.size() + interior * 5);
  reserveAtLeast(results.ring_offsets, results.ring_offsets.size() + interior);
  reserveAtLeast(results.polygon_offsets,
                 results.polygon_offsets.size() + interior);

  for (std::size_t start = 0; start < boundary.size();) {
    const long row = boundary[start].cell.row;
    bool inside = false;
    long previous_column = 0;
    std::size_t at = start;
    for (; at < boundary.size() && boundary[at].cell.row == row; at++) {
      const long column = boundary[at].cell.column;
      if (inside && column > previous_column + 1) {
        appendCellBoxRun(previous_column + 1, column, row, results);
      }
      inside = inside != boundary[at].crossed;
      previous_column = column;
    }
    start = at;
  }

  // Transform results back to world coordinates. Where the grid axes mirror
  // world orientation (e.g. a north-up raster whose row index increases
  // southwards), reverse each ring so that emitted exteriors stay
  // counter-clockwise in world coordinates.
  bool mirrored = (grid.grid_to_world.a * grid.grid_to_world.e -
                   grid.grid_to_world.b * grid.grid_to_world.d) < 0;
  // Written out for the axis-aligned transform this whole file assumes,
  // rather than through the general form: half the arithmetic per point,
  // over a loop the compiler can then vectorise.
  const double ax = grid.grid_to_world.a;
  const double cx = grid.grid_to_world.c;
  const double ey = grid.grid_to_world.e;
  const double fy = grid.grid_to_world.f;
  for (std::size_t i = coordinate_base; i < results.coordinates.size(); i++) {
    Coord &point = results.coordinates[i];
    point.x = point.x * ax + cx;
    point.y = point.y * ey + fy;
  }
  if (mirrored) {
    for (std::size_t r = ring_base; r + 1 < results.ring_offsets.size(); r++) {
      std::reverse(results.coordinates.begin() + results.ring_offsets[r],
                   results.coordinates.begin() + results.ring_offsets[r + 1]);
    }
  }
}

std::vector<geometry::Polygon>
splitPolygonGrid(const std::vector<linestr> &rings_in, const grid::Grid &grid) {
  PolygonPieces pieces = splitPolygonGridPieces(rings_in, grid);
  std::vector<geometry::Polygon> polygons;
  polygons.reserve(pieces.size());
  for (std::size_t p = 0; p + 1 < pieces.polygon_offsets.size(); p++) {
    std::size_t first = pieces.polygon_offsets[p];
    std::size_t last = pieces.polygon_offsets[p + 1];
    geometry::Polygon polygon;
    for (std::size_t r = first; r < last; r++) {
      linestr ring(pieces.coordinates.begin() + pieces.ring_offsets[r],
                   pieces.coordinates.begin() + pieces.ring_offsets[r + 1]);
      if (r == first) {
        polygon.exterior = std::move(ring);
      } else {
        polygon.interiors.push_back(std::move(ring));
      }
    }
    polygons.push_back(std::move(polygon));
  }
  return polygons;
}

/// The cell a segment lies in. Its endpoints may sit on cell borders, so
/// the midpoint decides; a segment running along a grid line is taken to
/// belong to the cell above or to the right of it, matching how
/// grid::Grid::cellIndices resolves a point on a border.
static CellIndex segmentCell(const Coord a, const Coord b) {
  const double mx = (a.x + b.x) / 2;
  const double my = (a.y + b.y) / 2;
  return CellIndex{static_cast<long>(std::floor(mx)),
                   static_cast<long>(std::floor(my))};
}

LinePieces splitLineStringGrid(CoordSpan coordinates, const grid::Grid &grid,
                               bool bounded) {
  LinePieces pieces;
  splitLineStringGrid(coordinates, grid, bounded, pieces);
  return pieces;
}

void splitLineStringGrid(CoordSpan coordinates, const grid::Grid &grid,
                         bool bounded, LinePieces &pieces) {
  // Into grid coordinates once, snapping vertices onto grid lines where
  // they lie within tolerance, and dropping points repeated in place. Each
  // is remembered against the input vertex it came from, so that the
  // line's own points can be given back exactly as they arrived rather
  // than as the transform round-trips them.
  linestr line;
  std::vector<std::size_t> line_source;
  line.reserve(coordinates.size());
  line_source.reserve(coordinates.size());
  for (std::size_t i = 0; i < coordinates.size(); i++) {
    Coord g = grid.world_to_grid * coordinates[i];
    g.x = snapped(g.x);
    g.y = snapped(g.y);
    if (line.empty() || !(line.back() == g)) {
      line.push_back(g);
      line_source.push_back(i);
    }
  }
  if (line.size() < 2) {
    return;
  }

  // Every point at which the line may change cell: its own vertices, and
  // its crossings with the grid lines. A crossing has no input vertex
  // behind it, marked here by an out-of-range source.
  const std::size_t computed = coordinates.size();
  auto withinGrid = [&](const Coord &point) {
    return point.x >= 0.0 && point.x <= grid.ncols && point.y >= 0.0 &&
           point.y <= grid.nrows;
  };
  linestr nodes;
  std::vector<std::size_t> node_source;
  nodes.reserve(line.size() * 2);
  node_source.reserve(line.size() * 2);
  nodes.push_back(line.front());
  node_source.push_back(line_source.front());
  for (std::size_t i = 0; i + 1 < line.size(); i++) {
    const bool segment_intersects_grid =
        !bounded || segmentIntersectsGridBounds(
                        geometry::Line(coordinates[line_source[i]],
                                       coordinates[line_source[i + 1]]),
                        grid);
    forEachCrossing(line[i], line[i + 1], [&](const Coord crossing) {
      if (segment_intersects_grid &&
          (!bounded || withinGrid(crossing)) && !(nodes.back() == crossing)) {
        nodes.push_back(crossing);
        node_source.push_back(computed);
      }
    });
    if (!(nodes.back() == line[i + 1])) {
      nodes.push_back(line[i + 1]);
      node_source.push_back(line_source[i + 1]);
    }
  }
  if (nodes.size() < 2) {
    return;
  }

  // Run of consecutive spans sharing a cell becomes one piece. Taking the
  // cell span by span, rather than breaking at every node, keeps a vertex
  // which only touches a grid line from cutting the line in two.
  auto emit = [&](std::size_t from, std::size_t to) {
    for (std::size_t k = from; k <= to; k++) {
      pieces.coordinates.push_back(node_source[k] < computed
                                       ? coordinates[node_source[k]]
                                       : grid.grid_to_world * nodes[k]);
    }
    pieces.endPiece();
  };

  std::size_t start = 0;
  CellIndex current = segmentCell(nodes[0], nodes[1]);
  for (std::size_t k = 1; k + 1 < nodes.size(); k++) {
    CellIndex cell = segmentCell(nodes[k], nodes[k + 1]);
    if (cell != current && (!bounded || withinGrid(nodes[k]))) {
      emit(start, k);
      start = k;
    }
    current = cell;
  }
  emit(start, nodes.size() - 1);
}

} // namespace snail::operations
