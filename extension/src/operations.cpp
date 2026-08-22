#include <algorithm> /// sort, unique, reverse
#include <cmath>     /// floor, fabs, fmod, hypot, round
#include <cstddef>
#include <limits>
#include <utility> /// pair
#include <vector>

#include "geometry.hpp"
#include "grid.hpp"
#include "operations.hpp"

namespace snail {
namespace operations {

using geometry::Coord;
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
const double SNAP_TOL = 1e-9;

/// Tolerance when comparing positions along a cell border (the border
/// parameter runs from 0 to 4, one unit per cell edge)
const double BORDER_PARAM_TOL = 1e-9;

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

/// Cell column/row index
using CellIndex = std::pair<long, long>;

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
  /// border parameters of the first and last point (open pieces only)
  double t_in = 0;
  double t_out = 0;
  bool used = false;
  /// the cell this piece lies in
  CellIndex cell{0, 0};

  std::size_t size() const { return end - begin; }
};

/// Order pieces by the row, then the column, of the cell they lie in, so
/// that the pieces of a cell end up adjacent and the cells themselves in
/// the order the interior scan sweeps them
bool rowMajorOrder(const Piece &a, const Piece &b) {
  if (a.cell.second != b.cell.second) {
    return a.cell.second < b.cell.second;
  }
  return a.cell.first < b.cell.first;
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

/// Border geometry of a single grid cell: integer corner coordinates and a
/// counter-clockwise border parameterisation, in grid space
struct CellBorder {
  long ci;
  long cj;

  CellBorder(const CellIndex index) : ci(index.first), cj(index.second) {}

  /// Grid coordinates of corner k (0 <= k < 4) along the counter-clockwise
  /// border walk (0,0) -> (1,0) -> (1,1) -> (0,1)
  Coord corner(int k) const {
    static const int du[4] = {0, 1, 1, 0};
    static const int dv[4] = {0, 0, 1, 1};
    return Coord(ci + du[k], cj + dv[k]);
  }

  /// Direction of the counter-clockwise border walk at parameter t
  static Coord direction(double t) {
    switch ((int)std::floor(t)) {
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
    double u = std::min(std::max(g.x - ci, 0.0), 1.0);
    double v = std::min(std::max(g.y - cj, 0.0), 1.0);
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
  double turn = std::atan2(from.y, from.x) - std::atan2(to.y, to.x);
  while (turn <= BORDER_PARAM_TOL) {
    turn += 2 * utils::PI;
  }
  return turn;
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
  // any vertex strictly inside a cell
  for (std::size_t k = 0; k < count; k++) {
    const Coord &point = points[k];
    if (!isInteger(point.x) && !isInteger(point.y)) {
      return CellIndex((long)std::floor(point.x), (long)std::floor(point.y));
    }
  }
  // else any segment midpoint strictly inside a cell
  for (std::size_t i = 0; i + 1 < count; i++) {
    Coord mid((points[i].x + points[i + 1].x) / 2,
              (points[i].y + points[i + 1].y) / 2);
    if (!isInteger(mid.x) && !isInteger(mid.y)) {
      return CellIndex((long)std::floor(mid.x), (long)std::floor(mid.y));
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
  long ci, cj;
  if (dx == 0) {
    ci = (nx > 0) ? (long)front.x : (long)front.x - 1;
  } else {
    ci = (long)std::floor(std::min(front.x, back.x));
  }
  if (dy == 0) {
    cj = (ny > 0) ? (long)front.y : (long)front.y - 1;
  } else {
    cj = (long)std::floor(std::min(front.y, back.y));
  }
  return CellIndex(ci, cj);
}

/// A full cell as a polygon (counter-clockwise, closed), in grid space
geometry::Polygon cellBoxPolygon(const CellIndex index) {
  CellBorder border(index);
  linestr ring = {border.corner(0), border.corner(1), border.corner(2),
                  border.corner(3), border.corner(0)};
  return geometry::Polygon{ring, {}};
}

/// Append a whole cell as a polygon piece, without allocating
void appendCellBox(const CellIndex index, PolygonPieces &pieces) {
  CellBorder border(index);
  for (int k = 0; k < 4; k++) {
    pieces.coordinates.push_back(border.corner(k));
  }
  pieces.coordinates.push_back(border.corner(0));
  pieces.endRing();
  pieces.endPolygon();
}

/// Append an assembled polygon as a piece
void appendPolygon(const geometry::Polygon &polygon, PolygonPieces &pieces) {
  pieces.coordinates.insert(pieces.coordinates.end(), polygon.exterior.begin(),
                            polygon.exterior.end());
  pieces.endRing();
  for (const linestr &interior : polygon.interiors) {
    pieces.coordinates.insert(pieces.coordinates.end(), interior.begin(),
                              interior.end());
    pieces.endRing();
  }
  pieces.endPolygon();
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
  long nx =
      (dx == 0)
          ? 0
          : (long)std::max(0.0, sx * (b.x - kx) + (isInteger(b.x) ? 0.0 : 1.0));
  long ny =
      (dy == 0)
          ? 0
          : (long)std::max(0.0, sy * (b.y - ky) + (isInteger(b.y) ? 0.0 : 1.0));

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

/// Split a traced ring into simple rings at any repeated (pinch) vertices,
/// so regions that touch at a single point become separate rings
void splitAtRepeatedVertices(const linestr &ring, std::vector<linestr> &out) {
  std::size_t n = ring.size();
  for (std::size_t i = 0; i + 1 < n; i++) {
    for (std::size_t j = i + 2; j < n; j++) {
      if (i == 0 && j == n - 1) {
        // the closing vertex pair does not pinch the ring
        continue;
      }
      if (ring[i] == ring[j]) {
        linestr loop(ring.begin() + i, ring.begin() + j + 1);
        linestr rest(ring.begin(), ring.begin() + i + 1);
        rest.insert(rest.end(), ring.begin() + j + 1, ring.end());
        splitAtRepeatedVertices(loop, out);
        splitAtRepeatedVertices(rest, out);
        return;
      }
    }
  }
  out.push_back(ring);
}

/// Assemble the polygon pieces covering a single cell from the ring pieces
/// that were bucketed to it, appending to results. The polygon interior lies
/// on the left of each piece (rings oriented exterior counter-clockwise,
/// holes clockwise), and the cell border is walked counter-clockwise, so
/// each covered region is traced by alternately following pieces and border
/// arcs.
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
  std::vector<std::pair<double, Coord>> endpoints;
  std::vector<Piece *> coincident;
  std::vector<std::pair<double, Coord>> passed;
  /// traced rings, laid end to end, with a span and signed area each
  std::vector<Coord> arena;
  std::vector<CellRing> cell_rings;
  /// scratch for the rare ring that pinches and has to be taken apart
  linestr pinch;
};

/// Append a ring to the cell's arena, splitting it at any repeated (pinch)
/// vertex so that regions meeting at a single point come out separately.
/// A ring with no repeated vertex - almost always - is appended as it
/// stands, without copying it anywhere first.
void appendSimpleRings(const Coord *points, std::size_t count,
                       Workspace &work) {
  if (count < 4) {
    return;
  }
  for (std::size_t i = 0; i + 1 < count; i++) {
    for (std::size_t j = i + 2; j < count; j++) {
      if (i == 0 && j == count - 1) {
        // the closing vertex pair does not pinch the ring
        continue;
      }
      if (points[i] == points[j]) {
        // the loop from i to j, and the ring with that loop removed
        appendSimpleRings(points + i, j - i + 1, work);
        linestr rest(points, points + i + 1);
        rest.insert(rest.end(), points + j + 1, points + count);
        appendSimpleRings(rest.data(), rest.size(), work);
        return;
      }
    }
  }
  CellRing ring;
  ring.begin = work.arena.size();
  work.arena.insert(work.arena.end(), points, points + count);
  ring.end = work.arena.size();
  ring.area2 = ringSignedArea2(points, count);
  work.cell_rings.push_back(ring);
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
    work.endpoints.push_back(std::make_pair(chain->t_in, points[chain->begin]));
    work.endpoints.push_back(
        std::make_pair(chain->t_out, points[chain->end - 1]));
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
      double t = current->t_out;
      Piece *next = nullptr;
      double best_dt = 5.0;
      double start_dt = 5.0;
      work.coincident.clear();
      for (Piece *candidate : work.chains) {
        if (candidate->used && candidate != start) {
          continue;
        }
        double dt = std::fmod(candidate->t_in - t, 4.0);
        if (dt < 0) {
          dt += 4.0;
        }
        // an entry point coinciding with the exit is at distance zero, not
        // a full lap
        if (dt > 4.0 - BORDER_PARAM_TOL) {
          dt = 0.0;
        }
        if (dt < BORDER_PARAM_TOL) {
          // This chain begins exactly where we are: the boundary touches
          // the cell border here without crossing it. Whether to follow the
          // chain or carry on along the border is decided below, by which
          // of them comes first turning clockwise away from the direction
          // we arrived from - the rule which keeps the covered area on our
          // left, as it is along every chain.
          work.coincident.push_back(candidate);
          continue;
        }
        if (dt < best_dt) {
          best_dt = dt;
          next = candidate;
        }
        if (candidate == start) {
          start_dt = dt;
        }
      }
      // when closing the ring ties with entering another chain, close
      if (start_dt <= best_dt + BORDER_PARAM_TOL) {
        best_dt = start_dt;
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
            clockwiseTurn(back, CellBorder::direction(t)) + BORDER_PARAM_TOL;
        for (Piece *candidate : work.coincident) {
          Coord away(
              points[candidate->begin + 1].x - points[candidate->begin].x,
              points[candidate->begin + 1].y - points[candidate->begin].y);
          double turn = clockwiseTurn(back, away);
          if (turn < least_turn) {
            least_turn = turn;
            next = candidate;
            best_dt = 0.0;
          }
        }
      }

      // Walk the border, adding any corners passed on the way, and noding
      // at any other chain endpoints passed - so that regions which pinch
      // together at a point on the border can be separated below
      work.passed.clear();
      double k = std::floor(t) + 1.0;
      for (int step = 0; step < 4; step++) {
        double d_corner = k - t;
        if (d_corner >= best_dt - BORDER_PARAM_TOL) {
          break;
        }
        if (d_corner > BORDER_PARAM_TOL) {
          work.passed.push_back(
              std::make_pair(d_corner, border.corner((int)std::fmod(k, 4.0))));
        }
        k += 1.0;
      }
      for (const std::pair<double, Coord> &endpoint : work.endpoints) {
        double dt = std::fmod(endpoint.first - t, 4.0);
        if (dt < 0) {
          dt += 4.0;
        }
        if (dt > BORDER_PARAM_TOL && dt < best_dt - BORDER_PARAM_TOL) {
          work.passed.push_back(std::make_pair(dt, endpoint.second));
        }
      }
      std::sort(
          work.passed.begin(), work.passed.end(),
          [](const std::pair<double, Coord> &a,
             const std::pair<double, Coord> &b) { return a.first < b.first; });
      for (const std::pair<double, Coord> &point : work.passed) {
        append_point(point.second);
      }

      if (next == start) {
        break;
      }
      current = next;
    }

    // close the ring, then take it out of the arena and put it back split
    // at any pinch point
    if (work.arena.size() > ring_begin) {
      const Coord front = work.arena[ring_begin];
      if (std::fabs(work.arena.back().x - front.x) < SNAP_TOL &&
          std::fabs(work.arena.back().y - front.y) < SNAP_TOL) {
        work.arena.back() = front;
      } else {
        work.arena.push_back(front);
      }
    }
    std::size_t count = work.arena.size() - ring_begin;
    work.pinch.assign(work.arena.begin() + ring_begin, work.arena.end());
    work.arena.erase(work.arena.begin() + ring_begin, work.arena.end());
    appendSimpleRings(work.pinch.data(), count, work);
  }

  // Every counter-clockwise ring is a covered region; every clockwise one
  // is a hole in whichever of them contains it
  for (std::size_t f = 0; f < work.cell_rings.size(); f++) {
    const CellRing &face = work.cell_rings[f];
    if (face.area2 <= 0) {
      continue;
    }
    results.coordinates.insert(results.coordinates.end(),
                               work.arena.begin() + face.begin,
                               work.arena.begin() + face.end);
    results.endRing();
    for (std::size_t h = 0; h < work.cell_rings.size(); h++) {
      const CellRing &hole = work.cell_rings[h];
      if (hole.area2 >= 0) {
        continue;
      }
      Coord inside = ringInsidePoint(work.arena.data() + hole.begin,
                                     hole.end - hole.begin, hole.area2);
      if (!pointInRing(inside, work.arena.data() + face.begin,
                       face.end - face.begin)) {
        continue;
      }
      // a hole belongs to the smallest region containing it; with the
      // regions of one cell disjoint, the first containing one is it
      results.coordinates.insert(results.coordinates.end(),
                                 work.arena.begin() + hole.begin,
                                 work.arena.begin() + hole.end);
      results.endRing();
    }
    results.endPolygon();
  }

  // A hole with no region around it means the rest of the cell is covered
  for (const CellRing &hole : work.cell_rings) {
    if (hole.area2 >= 0) {
      continue;
    }
    bool contained = false;
    Coord inside = ringInsidePoint(work.arena.data() + hole.begin,
                                   hole.end - hole.begin, hole.area2);
    for (const CellRing &face : work.cell_rings) {
      if (face.area2 > 0 && pointInRing(inside, work.arena.data() + face.begin,
                                        face.end - face.begin)) {
        contained = true;
        break;
      }
    }
    if (contained) {
      continue;
    }
    CellBorder box(index);
    for (int c = 0; c < 4; c++) {
      results.coordinates.push_back(box.corner(c));
    }
    results.coordinates.push_back(box.corner(0));
    results.endRing();
    results.coordinates.insert(results.coordinates.end(),
                               work.arena.begin() + hole.begin,
                               work.arena.begin() + hole.end);
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

PolygonPieces splitPolygonGridPieces(const std::vector<linestr> &rings_in,
                                     const grid::Grid &grid) {
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
        return PolygonPieces();
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
    return PolygonPieces();
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
        piece.t_in = border.parameter(points[0]);
        piece.t_out = border.parameter(points[count - 1]);
      }
      bucketed.push_back(piece);
    }
  }

  // Gather the pieces of each cell together, in the order the interior
  // scan below sweeps them. Sorting once beats inserting each piece into an
  // ordered container, and a stable sort keeps the pieces of a cell in the
  // order the rings were traversed. The pieces of a cell are then a
  // contiguous run.
  std::stable_sort(bucketed.begin(), bucketed.end(), rowMajorOrder);

  // Assemble the pieces covering each cell crossed by a ring, noting as we
  // go whether the boundary in the cell crosses the horizontal line through
  // the cell centre an odd number of times: a chain does so exactly when
  // its endpoints lie on opposite sides of that line, and a closed piece
  // never does.
  PolygonPieces results;
  for (std::size_t start = 0; start < bucketed.size();) {
    const CellIndex cell = bucketed[start].cell;
    const double level = cell.second + 0.5;
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
  for (std::size_t start = 0; start < boundary.size();) {
    const long j = boundary[start].cell.second;
    bool inside = false;
    long previous_i = 0;
    std::size_t at = start;
    for (; at < boundary.size() && boundary[at].cell.second == j; at++) {
      const long i = boundary[at].cell.first;
      if (inside) {
        for (long k = previous_i + 1; k < i; k++) {
          appendCellBox(CellIndex(k, j), results);
        }
      }
      inside = inside != boundary[at].crossed;
      previous_i = i;
    }
    start = at;
  }

  // Transform results back to world coordinates. Where the grid axes mirror
  // world orientation (e.g. a north-up raster whose row index increases
  // southwards), reverse each ring so that emitted exteriors stay
  // counter-clockwise in world coordinates.
  bool mirrored = (grid.grid_to_world.a * grid.grid_to_world.e -
                   grid.grid_to_world.b * grid.grid_to_world.d) < 0;
  for (Coord &point : results.coordinates) {
    point = grid.grid_to_world * point;
  }
  if (mirrored) {
    for (std::size_t r = 0; r + 1 < results.ring_offsets.size(); r++) {
      std::reverse(results.coordinates.begin() + results.ring_offsets[r],
                   results.coordinates.begin() + results.ring_offsets[r + 1]);
    }
  }
  return results;
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
  return CellIndex((long)std::floor(mx), (long)std::floor(my));
}

LinePieces splitLineStringGrid(const linestr &coordinates,
                               const grid::Grid &grid) {
  LinePieces pieces;

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
    return pieces;
  }

  // Every point at which the line may change cell: its own vertices, and
  // its crossings with the grid lines. A crossing has no input vertex
  // behind it, marked here by an out-of-range source.
  const std::size_t computed = coordinates.size();
  linestr nodes;
  std::vector<std::size_t> node_source;
  nodes.reserve(line.size() * 2);
  node_source.reserve(line.size() * 2);
  nodes.push_back(line.front());
  node_source.push_back(line_source.front());
  for (std::size_t i = 0; i + 1 < line.size(); i++) {
    forEachCrossing(line[i], line[i + 1], [&](const Coord crossing) {
      if (!(nodes.back() == crossing)) {
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
    return pieces;
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
    if (cell != current) {
      emit(start, k);
      start = k;
      current = cell;
    }
  }
  emit(start, nodes.size() - 1);

  return pieces;
}

} // namespace operations
} // namespace snail
