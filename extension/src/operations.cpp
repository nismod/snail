#include <algorithm> /// sort, unique, reverse
#include <cmath>     /// floor, fabs, fmod, hypot, round
#include <cstddef>
#include <limits>
#include <map>
#include <utility> /// pair
#include <vector>

#include "geometry.hpp"
#include "grid.hpp"
#include "operations.hpp"

namespace snail {
namespace operations {

using geometry::Coord;
using linestr = std::vector<geometry::Coord>;

/// Piecewise decomposition of a linestring according to intersection points
std::vector<linestr> split_linestr(linestr linestring, linestr intersections) {
  // Add line start point
  linestring.push_back(intersections.at(0));
  // Loop over each intersection, and add a new feature for each
  std::vector<linestr> splits;
  for (std::size_t j = 1; j < intersections.size(); j++) {
    // Add the crossing point to the cleaned features geometry.
    linestring.push_back(intersections.at(j));
    splits.push_back(linestring);
    linestring.clear();
    linestring.push_back(intersections.at(j));
  }
  return (splits);
}

/// Find intersection points of a linestring with a raster grid
std::vector<linestr>
findIntersectionsLineString(geometry::LineString linestring,
                            grid::Grid raster) {
  linestr coords = linestring.coordinates;

  std::vector<linestr> allsplits;
  linestr linestr_piece;
  for (std::size_t i = 0; i < coords.size() - 1; i++) {
    geometry::Line line(coords.at(i), coords.at(i + 1));

    // If the line starts and ends in different cells, it needs to be cleaned.
    if (raster.cellIndex(line.start) != raster.cellIndex(line.end)) {
      linestr intersections = raster.findIntersections(line);
      if (intersections.size() == 1) {
        // The segment changes cell, but no crossing point was found within
        // it: the grid line it crosses passes exactly through the start or
        // the end of the segment (the crossing at zero or full segment
        // length was lost to rounding). Find which endpoint sits on the
        // boundary between the two cells, and break the running piece
        // there.
        std::tuple<int, int> start_cell = raster.cellIndices(line.start);
        std::tuple<int, int> end_cell = raster.cellIndices(line.end);
        geometry::Coord g_start = raster.world_to_grid * line.start;
        geometry::Coord g_end = raster.world_to_grid * line.end;
        double d_start = std::numeric_limits<double>::max();
        double d_end = std::numeric_limits<double>::max();
        if (std::get<0>(start_cell) != std::get<0>(end_cell)) {
          double boundary =
              std::max(std::get<0>(start_cell), std::get<0>(end_cell));
          d_start = std::min(d_start, std::fabs(g_start.x - boundary));
          d_end = std::min(d_end, std::fabs(g_end.x - boundary));
        }
        if (std::get<1>(start_cell) != std::get<1>(end_cell)) {
          double boundary =
              std::max(std::get<1>(start_cell), std::get<1>(end_cell));
          d_start = std::min(d_start, std::fabs(g_start.y - boundary));
          d_end = std::min(d_end, std::fabs(g_end.y - boundary));
        }
        if (d_start <= d_end) {
          // break at the start point: close any running piece there and
          // carry on from it
          if (!linestr_piece.empty()) {
            if (!(linestr_piece.back() == line.start)) {
              linestr_piece.push_back(line.start);
            }
            if (linestr_piece.size() >= 2) {
              allsplits.push_back(linestr_piece);
            }
          }
          linestr_piece = {line.start};
        } else {
          // break at the end point, treating it as the crossing
          if (linestr_piece.empty() || !(linestr_piece.back() == line.start)) {
            linestr_piece.push_back(line.start);
          }
          linestr_piece.push_back(line.end);
          allsplits.push_back(linestr_piece);
          linestr_piece = {};
        }
      } else {
        std::vector<linestr> splits =
            split_linestr(linestr_piece, intersections);
        allsplits.insert(allsplits.end(), splits.begin(), splits.end());
        if (line.end == intersections.back()) {
          linestr_piece = {};
        } else {
          linestr_piece = {intersections.back()};
        }
      }
    } else {
      linestr_piece.push_back(coords.at(i));
    }
  }

  if (linestr_piece.size() > 0) {
    linestr_piece.push_back(coords.back());
    allsplits.push_back(linestr_piece);
  }

  return (allsplits);
}

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
double ringSignedArea2(const linestr &ring) {
  if (ring.empty()) {
    return 0;
  }
  const Coord origin = ring.front();
  double area2 = 0;
  for (std::size_t i = 0; i + 1 < ring.size(); i++) {
    area2 += (ring[i].x - origin.x) * (ring[i + 1].y - origin.y) -
             (ring[i + 1].x - origin.x) * (ring[i].y - origin.y);
  }
  return area2;
}

/// Even-odd test for a point inside a closed ring. Points on the boundary
/// may be classified either way.
bool pointInRing(const Coord p, const linestr &ring) {
  bool inside = false;
  for (std::size_t i = 0; i + 1 < ring.size(); i++) {
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

/// A piece of a polygon ring lying within a single grid cell
struct Piece {
  linestr points;
  /// closed pieces are rings that never cross a grid line (or loops pinched
  /// at a single border point); open pieces are chains crossing the cell
  bool closed = false;
  /// twice the signed area (meaningful for closed pieces)
  double area2 = 0;
  /// border parameters of the first and last point (open pieces only)
  double t_in = 0;
  double t_out = 0;
  bool used = false;
};

/// The ring pieces bucketed to a single cell, plus the parity of their
/// crossings of the horizontal line through the cell's centre - tracked so
/// the interior scan can update its inside/outside status from the cells it
/// encounters, without testing points against the whole set of rings
struct CellPieces {
  std::vector<Piece> pieces;
  /// true if the boundary within this cell crosses the row-centre line an
  /// odd number of times
  bool crossed = false;
};

/// Order cells by row, then column, so the interior scan can sweep along
/// each row of cells
struct RowMajorOrder {
  bool operator()(const CellIndex &a, const CellIndex &b) const {
    if (a.second != b.second) {
      return a.second < b.second;
    }
    return a.first < b.first;
  }
};

using CellMap = std::map<CellIndex, CellPieces, RowMajorOrder>;

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
std::size_t longestSegment(const linestr &points) {
  std::size_t longest = 0;
  double longest_length2 = -1.0;
  for (std::size_t i = 0; i + 1 < points.size(); i++) {
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
CellIndex pieceCell(const linestr &points, double side_sign) {
  // any vertex strictly inside a cell
  for (const Coord &point : points) {
    if (!isInteger(point.x) && !isInteger(point.y)) {
      return CellIndex((long)std::floor(point.x), (long)std::floor(point.y));
    }
  }
  // else any segment midpoint strictly inside a cell
  for (std::size_t i = 0; i + 1 < points.size(); i++) {
    Coord mid((points[i].x + points[i + 1].x) / 2,
              (points[i].y + points[i + 1].y) / 2);
    if (!isInteger(mid.x) && !isInteger(mid.y)) {
      return CellIndex((long)std::floor(mid.x), (long)std::floor(mid.y));
    }
  }
  // else the piece runs along a grid line: take the cell on the requested
  // side of the direction of travel
  const Coord front = points.front();
  const Coord back = points.back();
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

/// A point just inside a closed ring piece: the midpoint of its longest
/// segment, nudged towards the ring's own interior
Coord ringInsidePoint(const linestr &points, double area2) {
  std::size_t seg = longestSegment(points);
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

/// Split a closed grid-space ring at every crossing with a grid line,
/// breaking also at any vertex which lies on a grid line (a crossing may
/// pass exactly through a vertex). Every returned piece lies within a
/// single cell. Breaking at a vertex which only touches a grid line is
/// harmless: the two chains reconnect seamlessly when the cell is
/// assembled.
std::vector<linestr> splitRing(const linestr &ring) {
  std::vector<linestr> pieces;
  linestr piece = {ring.front()};
  for (std::size_t i = 0; i + 1 < ring.size(); i++) {
    const Coord a = ring[i];
    const Coord b = ring[i + 1];
    double dx = b.x - a.x;
    double dy = b.y - a.y;

    // Crossings with the integer x and y levels strictly between a and b,
    // in order of position along the segment
    double sx = (dx > 0) ? 1.0 : -1.0;
    double sy = (dy > 0) ? 1.0 : -1.0;
    double kx = (sx > 0) ? std::floor(a.x) + 1 : std::ceil(a.x) - 1;
    double ky = (sy > 0) ? std::floor(a.y) + 1 : std::ceil(a.y) - 1;
    bool has_x = (dx != 0) && (sx > 0 ? kx < b.x : kx > b.x);
    bool has_y = (dy != 0) && (sy > 0 ? ky < b.y : ky > b.y);
    while (has_x || has_y) {
      double tx = has_x ? (kx - a.x) / dx : 2.0;
      double ty = has_y ? (ky - a.y) / dy : 2.0;
      Coord crossing(0, 0);
      if (tx <= ty) {
        crossing = Coord(kx, snapped(a.y + tx * dy));
        kx += sx;
        has_x = (sx > 0) ? kx < b.x : kx > b.x;
        // a crossing through a cell corner consumes both levels
        if (has_y && crossing.y == ky) {
          ky += sy;
          has_y = (sy > 0) ? ky < b.y : ky > b.y;
        }
      } else {
        crossing = Coord(snapped(a.x + ty * dx), ky);
        ky += sy;
        has_y = (sy > 0) ? ky < b.y : ky > b.y;
        if (has_x && crossing.x == kx) {
          kx += sx;
          has_x = (sx > 0) ? kx < b.x : kx > b.x;
        }
      }
      if (!(piece.back() == crossing)) {
        piece.push_back(crossing);
      }
      if (piece.size() >= 2) {
        pieces.push_back(std::move(piece));
      }
      piece = {crossing};
    }

    // the segment end: break here too if it lies on a grid line (unless it
    // is the ring's closing vertex, which ends the final piece anyway)
    if (!(piece.back() == b)) {
      piece.push_back(b);
    }
    if (i + 2 < ring.size() && (isInteger(b.x) || isInteger(b.y))) {
      if (piece.size() >= 2) {
        pieces.push_back(std::move(piece));
      }
      piece = {b};
    }
  }
  if (piece.size() >= 2) {
    pieces.push_back(std::move(piece));
  }
  return pieces;
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
void assembleCell(const CellIndex index, std::vector<Piece> &pieces,
                  std::vector<geometry::Polygon> &results) {
  CellBorder border(index);

  std::vector<Piece *> chains;
  std::vector<linestr> holes;
  std::vector<geometry::Polygon> faces;
  for (Piece &piece : pieces) {
    if (!piece.closed) {
      chains.push_back(&piece);
    } else if (piece.area2 > 0) {
      // a ring that never leaves the cell is a piece in itself
      faces.push_back(geometry::Polygon{piece.points, {}});
    } else if (piece.area2 < 0) {
      // an interior ring that never leaves the cell
      holes.push_back(piece.points);
    }
    // pieces with exactly zero area are degenerate - ignored
  }

  // Chain endpoints, so border walks can be noded wherever the polygon
  // boundary touches the border
  std::vector<std::pair<double, Coord>> endpoints;
  for (Piece *chain : chains) {
    endpoints.push_back(std::make_pair(chain->t_in, chain->points.front()));
    endpoints.push_back(std::make_pair(chain->t_out, chain->points.back()));
  }

  // Weld points within floating-point noise of each other as rings are
  // built: a polygon vertex on a grid line and the computed crossing point
  // there may differ in the last few bits, and keeping both can make a ring
  // fold back on itself
  auto append_point = [](linestr &ring, const Coord &point) {
    if (!ring.empty() && std::fabs(ring.back().x - point.x) < SNAP_TOL &&
        std::fabs(ring.back().y - point.y) < SNAP_TOL) {
      return;
    }
    ring.push_back(point);
  };

  // Trace faces bounded by chains and border arcs
  for (Piece *start : chains) {
    if (start->used) {
      continue;
    }
    linestr ring;
    Piece *current = start;
    // each iteration consumes a chain, so this terminates
    while (true) {
      current->used = true;
      for (const Coord &point : current->points) {
        append_point(ring, point);
      }

      // Find the next chain entry, walking counter-clockwise around the
      // border from the current exit; the ring closes when the nearest
      // entry is the starting chain's.
      double t = current->t_out;
      Piece *next = nullptr;
      double best_dt = 5.0;
      double start_dt = 5.0;
      std::vector<Piece *> coincident;
      for (Piece *candidate : chains) {
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
          coincident.push_back(candidate);
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

      if (!coincident.empty()) {
        // direction we arrived from, reversed
        const linestr &arriving = current->points;
        Coord back(arriving[arriving.size() - 2].x - arriving.back().x,
                   arriving[arriving.size() - 2].y - arriving.back().y);
        // Carrying on along the border is the alternative to beat. A chain
        // ties with it when the chain runs along the border itself, and
        // then the chain must win: it is the same path, and leaving it
        // untraced would strand it as a piece of its own.
        double least_turn =
            clockwiseTurn(back, CellBorder::direction(t)) + BORDER_PARAM_TOL;
        for (Piece *candidate : coincident) {
          Coord away(candidate->points[1].x - candidate->points[0].x,
                     candidate->points[1].y - candidate->points[0].y);
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
      std::vector<std::pair<double, Coord>> passed;
      double k = std::floor(t) + 1.0;
      for (int step = 0; step < 4; step++) {
        double d_corner = k - t;
        if (d_corner >= best_dt - BORDER_PARAM_TOL) {
          break;
        }
        if (d_corner > BORDER_PARAM_TOL) {
          passed.push_back(
              std::make_pair(d_corner, border.corner((int)std::fmod(k, 4.0))));
        }
        k += 1.0;
      }
      for (const std::pair<double, Coord> &endpoint : endpoints) {
        double dt = std::fmod(endpoint.first - t, 4.0);
        if (dt < 0) {
          dt += 4.0;
        }
        if (dt > BORDER_PARAM_TOL && dt < best_dt - BORDER_PARAM_TOL) {
          passed.push_back(std::make_pair(dt, endpoint.second));
        }
      }
      std::sort(
          passed.begin(), passed.end(),
          [](const std::pair<double, Coord> &a,
             const std::pair<double, Coord> &b) { return a.first < b.first; });
      for (const std::pair<double, Coord> &point : passed) {
        append_point(ring, point.second);
      }

      if (next == start) {
        break;
      }
      current = next;
    }

    // close the ring
    if (!ring.empty()) {
      if (std::fabs(ring.back().x - ring.front().x) < SNAP_TOL &&
          std::fabs(ring.back().y - ring.front().y) < SNAP_TOL) {
        ring.back() = ring.front();
      } else {
        ring.push_back(ring.front());
      }
    }
    if (ring.size() >= 4) {
      // separate any regions that meet at a single pinch point
      std::vector<linestr> simple_rings;
      splitAtRepeatedVertices(ring, simple_rings);
      for (const linestr &simple : simple_rings) {
        if (simple.size() < 4) {
          continue;
        }
        double area2 = ringSignedArea2(simple);
        if (area2 > 0) {
          faces.push_back(geometry::Polygon{simple, {}});
        } else if (area2 < 0) {
          // An interior ring which meets the cell border only where it
          // touches it (tangentially), so that it was split into chains
          // rather than staying a closed piece: traced clockwise, it bounds
          // a hole rather than a covered region.
          holes.push_back(simple);
        }
      }
    }
  }

  // Attach each interior ring to the face that contains it
  for (const linestr &hole : holes) {
    Coord inside = ringInsidePoint(hole, ringSignedArea2(hole));
    std::size_t target = faces.size();
    for (std::size_t i = 0; i < faces.size(); i++) {
      if (pointInRing(inside, faces[i].exterior)) {
        target = i;
        break;
      }
    }
    if (target == faces.size()) {
      if (faces.empty()) {
        // a hole in an otherwise fully-covered cell
        faces.push_back(cellBoxPolygon(index));
        target = 0;
      } else {
        // fallback (floating-point edge case): the largest face
        double largest = -1.0;
        for (std::size_t i = 0; i < faces.size(); i++) {
          double area2 = ringSignedArea2(faces[i].exterior);
          if (area2 > largest) {
            largest = area2;
            target = i;
          }
        }
      }
    }
    faces[target].interiors.push_back(hole);
  }

  results.insert(results.end(), faces.begin(), faces.end());
}

} // namespace

std::vector<geometry::Polygon>
splitPolygonGrid(const std::vector<linestr> &rings_in, const grid::Grid &grid) {
  // Transform each ring to grid coordinates - once, up front - snapping
  // vertices onto grid lines where they lie within tolerance; drop
  // consecutive duplicate points, close each ring, and orient rings so that
  // the polygon interior is always on the left of the direction of travel
  // in grid space (exterior counter-clockwise, holes clockwise).
  std::vector<linestr> rings;
  for (std::size_t r = 0; r < rings_in.size(); r++) {
    linestr ring;
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
        return {};
      }
      // degenerate hole
      continue;
    }
    if (is_exterior ? (area2 < 0) : (area2 > 0)) {
      std::reverse(ring.begin(), ring.end());
    }
    rings.push_back(std::move(ring));
  }
  if (rings.empty()) {
    return {};
  }

  // Split each ring at every grid line crossing and bucket the resulting
  // pieces by the cell they lie in
  CellMap cell_pieces;
  for (const linestr &ring : rings) {
    std::vector<linestr> pieces = splitRing(ring);
    if (pieces.empty()) {
      continue;
    }

    // When the ring starts partway through a cell, the last and first
    // pieces are two halves of the same chain: merge them. When the ring
    // happens to start on a grid line, keep the break (chains must only
    // meet the cell border at their endpoints).
    const Coord junction = ring.front();
    if (pieces.size() >= 2 && !isInteger(junction.x) &&
        !isInteger(junction.y) &&
        pieces.back().back() == pieces.front().front()) {
      linestr merged = std::move(pieces.back());
      merged.insert(merged.end(), pieces.front().begin() + 1,
                    pieces.front().end());
      pieces.front() = std::move(merged);
      pieces.pop_back();
    }

    for (linestr &points : pieces) {
      Piece piece;
      piece.points = std::move(points);
      bool nearly_closed =
          std::fabs(piece.points.front().x - piece.points.back().x) <
              SNAP_TOL &&
          std::fabs(piece.points.front().y - piece.points.back().y) < SNAP_TOL;
      piece.closed = nearly_closed;
      if (piece.closed) {
        // snap closed
        piece.points.back() = piece.points.front();
        if (piece.points.size() < 4) {
          continue;
        }
        piece.area2 = ringSignedArea2(piece.points);
        // bucket by the side the piece's own interior is on; being closed,
        // it crosses the row-centre line an even number of times
        double side = (piece.area2 < 0) ? -1.0 : 1.0;
        CellIndex cell = pieceCell(piece.points, side);
        cell_pieces[cell].pieces.push_back(std::move(piece));
      } else {
        // an open chain: the polygon interior is on the left
        CellIndex cell = pieceCell(piece.points, 1.0);
        CellBorder border(cell);
        piece.t_in = border.parameter(piece.points.front());
        piece.t_out = border.parameter(piece.points.back());
        CellPieces &content = cell_pieces[cell];
        // the chain crosses the row-centre line an odd number of times
        // exactly when its endpoints lie on opposite sides of it
        double level = cell.second + 0.5;
        content.crossed =
            content.crossed != ((piece.points.front().y < level) !=
                                (piece.points.back().y < level));
        content.pieces.push_back(std::move(piece));
      }
    }
  }

  // Assemble the pieces covering each cell crossed by a ring
  std::vector<geometry::Polygon> results;
  for (auto &entry : cell_pieces) {
    assembleCell(entry.first, entry.second.pieces, results);
  }

  // Cells wholly inside the polygon become full cell boxes. Scan each row
  // of boundary cells in column order, tracking whether the row of cells in
  // between is inside the polygon: the status can only change across a cell
  // crossed by the boundary, and it flips there exactly when the boundary
  // pieces in the cell cross the row's centre line an odd number of times
  // (recorded per cell during bucketing). The half-open crossing rule makes
  // the parity along each whole row even, so every row ends outside.
  for (auto it = cell_pieces.begin(); it != cell_pieces.end();) {
    long j = it->first.second;
    bool inside = false;
    long previous_i = 0;
    for (; it != cell_pieces.end() && it->first.second == j; ++it) {
      long i = it->first.first;
      if (inside) {
        for (long k = previous_i + 1; k < i; k++) {
          results.push_back(cellBoxPolygon(CellIndex(k, j)));
        }
      }
      inside = inside != it->second.crossed;
      previous_i = i;
    }
  }

  // Transform results back to world coordinates. Where the grid axes mirror
  // world orientation (e.g. a north-up raster whose row index increases
  // southwards), reverse each ring so that emitted exteriors stay
  // counter-clockwise in world coordinates.
  bool mirrored = (grid.grid_to_world.a * grid.grid_to_world.e -
                   grid.grid_to_world.b * grid.grid_to_world.d) < 0;
  for (geometry::Polygon &result : results) {
    for (Coord &point : result.exterior) {
      point = grid.grid_to_world * point;
    }
    if (mirrored) {
      std::reverse(result.exterior.begin(), result.exterior.end());
    }
    for (linestr &interior : result.interiors) {
      for (Coord &point : interior) {
        point = grid.grid_to_world * point;
      }
      if (mirrored) {
        std::reverse(interior.begin(), interior.end());
      }
    }
  }
  return results;
}

} // namespace operations
} // namespace snail
