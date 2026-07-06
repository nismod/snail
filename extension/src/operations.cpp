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

// The machinery below splits a polygon along the lines of a raster grid,
// assembling the resulting pieces cell by cell. The approach:
// - split each ring of the polygon at every grid line crossing (using
//   findIntersectionsLineString above), giving ring pieces that each lie
//   within a single cell, with endpoints on the cell border
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

/// Tolerance (in cell units) within which a coordinate is considered to lie
/// on a grid line when assigning pieces to cells
const double ON_GRIDLINE_TOL = 1e-6;

/// Tolerance when comparing positions along a cell border (the border
/// parameter runs from 0 to 4, one unit per cell edge)
const double BORDER_PARAM_TOL = 1e-9;

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

/// Sorted x positions at which the rings cross the horizontal line
/// y == level. The half-open rule guarantees an even number of crossings
/// for closed rings, robust to floating-point noise, vertices on the line
/// and boundary sections that run along the line.
std::vector<double> ringCrossings(const std::vector<linestr> &rings,
                                  double level) {
  std::vector<double> xs;
  for (const linestr &ring : rings) {
    for (std::size_t i = 0; i + 1 < ring.size(); i++) {
      const Coord a = ring[i];
      const Coord b = ring[i + 1];
      if ((a.y < level) != (b.y < level)) {
        double t = (level - a.y) / (b.y - a.y);
        xs.push_back(a.x + t * (b.x - a.x));
      }
    }
  }
  std::sort(xs.begin(), xs.end());
  return xs;
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

/// Border geometry of a single grid cell: corner coordinates and a
/// counter-clockwise (in world orientation) border parameterisation
struct CellBorder {
  long ci;
  long cj;
  /// grid-space corner offsets in counter-clockwise (world) walk order
  int du[4];
  int dv[4];
  const grid::Grid &grid;

  CellBorder(const CellIndex index, const grid::Grid &grid)
      : ci(index.first), cj(index.second), grid(grid) {
    double det = grid.grid_to_world.a * grid.grid_to_world.e -
                 grid.grid_to_world.b * grid.grid_to_world.d;
    if (det >= 0) {
      // grid axes agree with world orientation
      int u[4] = {0, 1, 1, 0};
      int v[4] = {0, 0, 1, 1};
      std::copy(u, u + 4, du);
      std::copy(v, v + 4, dv);
    } else {
      // grid axes are mirrored (e.g. north-up raster with row index
      // increasing southwards) - reverse the walk to stay counter-clockwise
      // in world orientation
      int u[4] = {0, 0, 1, 1};
      int v[4] = {0, 1, 1, 0};
      std::copy(u, u + 4, du);
      std::copy(v, v + 4, dv);
    }
  }

  /// World coordinates of corner k (0 <= k < 4) along the border walk
  Coord corner(int k) const {
    return grid.grid_to_world * Coord(ci + du[k], cj + dv[k]);
  }

  /// Position of a (near-border) world point along the border walk, in
  /// [0, 4): the nearest point of the border, one unit per cell edge
  double parameter(const Coord p) const {
    Coord g = grid.world_to_grid * p;
    double u = std::min(std::max(g.x - ci, 0.0), 1.0);
    double v = std::min(std::max(g.y - cj, 0.0), 1.0);
    double best_distance = std::numeric_limits<double>::max();
    double best_t = 0;
    for (int k = 0; k < 4; k++) {
      int u0 = du[k];
      int v0 = dv[k];
      int u1 = du[(k + 1) % 4];
      int v1 = dv[(k + 1) % 4];
      double distance, s;
      if (u0 == u1) {
        // edge at constant u
        distance = std::fabs(u - u0);
        s = (v - v0) / (double)(v1 - v0);
      } else {
        // edge at constant v
        distance = std::fabs(v - v0);
        s = (u - u0) / (double)(u1 - u0);
      }
      s = std::min(std::max(s, 0.0), 1.0);
      if (distance < best_distance) {
        best_distance = distance;
        best_t = k + s;
      }
    }
    if (best_t >= 4.0) {
      best_t -= 4.0;
    }
    return best_t;
  }
};

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
/// so in general this is the cell containing the midpoint of the longest
/// segment. When that midpoint lies on a grid line (the piece runs along the
/// line), take the cell on the requested side of the direction of travel:
/// side_sign +1 for the left (in world orientation), -1 for the right.
CellIndex pieceCell(const linestr &points, double side_sign,
                    const grid::Grid &grid) {
  std::size_t seg = longestSegment(points);
  Coord a = grid.world_to_grid * points[seg];
  Coord b = grid.world_to_grid * points[seg + 1];
  Coord mid((a.x + b.x) / 2, (a.y + b.y) / 2);

  // Normal towards the requested side, in grid space. If the transform
  // mirrors orientation, the world-space left appears on the right in grid
  // space (holds component-wise for axis-aligned transforms).
  double det = grid.grid_to_world.a * grid.grid_to_world.e -
               grid.grid_to_world.b * grid.grid_to_world.d;
  double mirror = (det < 0) ? -1.0 : 1.0;
  double nx = -(b.y - a.y) * side_sign * mirror;
  double ny = (b.x - a.x) * side_sign * mirror;

  long ci, cj;
  double rx = std::round(mid.x);
  if (std::fabs(mid.x - rx) < ON_GRIDLINE_TOL) {
    ci = (nx > 0) ? (long)rx : (long)rx - 1;
  } else {
    ci = (long)std::floor(mid.x);
  }
  double ry = std::round(mid.y);
  if (std::fabs(mid.y - ry) < ON_GRIDLINE_TOL) {
    cj = (ny > 0) ? (long)ry : (long)ry - 1;
  } else {
    cj = (long)std::floor(mid.y);
  }
  return CellIndex(ci, cj);
}

/// A full cell as a polygon (counter-clockwise, closed)
geometry::Polygon cellBoxPolygon(const CellIndex index,
                                 const grid::Grid &grid) {
  CellBorder border(index, grid);
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
                  const grid::Grid &grid,
                  std::vector<geometry::Polygon> &results) {
  CellBorder border(index, grid);

  std::vector<Piece *> chains;
  std::vector<Piece *> closed_holes;
  std::vector<geometry::Polygon> faces;
  for (Piece &piece : pieces) {
    if (!piece.closed) {
      chains.push_back(&piece);
    } else if (piece.area2 > 0) {
      // a ring that never leaves the cell is a piece in itself
      faces.push_back(geometry::Polygon{piece.points, {}});
    } else if (piece.area2 < 0) {
      closed_holes.push_back(&piece);
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
  double cellsize = std::max(std::fabs(grid.grid_to_world.a),
                             std::fabs(grid.grid_to_world.e));
  auto append_point = [cellsize](linestr &ring, const Coord &point) {
    if (!ring.empty() &&
        utils::almost_equal(ring.back().x, point.x, cellsize) &&
        utils::almost_equal(ring.back().y, point.y, cellsize)) {
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
      if (utils::almost_equal(ring.back().x, ring.front().x, cellsize) &&
          utils::almost_equal(ring.back().y, ring.front().y, cellsize)) {
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
        if (simple.size() >= 4 && ringSignedArea2(simple) > 0) {
          faces.push_back(geometry::Polygon{simple, {}});
        }
      }
    }
  }

  // Attach holes that never leave the cell to the face that contains them
  for (Piece *hole : closed_holes) {
    Coord inside = ringInsidePoint(hole->points, hole->area2);
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
        faces.push_back(cellBoxPolygon(index, grid));
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
    faces[target].interiors.push_back(hole->points);
  }

  results.insert(results.end(), faces.begin(), faces.end());
}

} // namespace

std::vector<geometry::Polygon>
splitPolygonGrid(const std::vector<linestr> &rings_in, const grid::Grid &grid) {
  double cellsize = std::max(std::fabs(grid.grid_to_world.a),
                             std::fabs(grid.grid_to_world.e));

  // Clean and orient rings: exterior counter-clockwise, holes clockwise, so
  // the polygon interior is always on the left of the direction of travel
  std::vector<linestr> rings;
  for (std::size_t r = 0; r < rings_in.size(); r++) {
    linestr ring = rings_in[r];
    // drop consecutive duplicate points
    ring.erase(std::unique(ring.begin(), ring.end()), ring.end());
    // close the ring
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
    rings.push_back(ring);
  }
  if (rings.empty()) {
    return {};
  }

  // Split each ring at every grid line crossing and bucket the resulting
  // pieces by the cell they lie in
  std::map<CellIndex, std::vector<Piece>> cell_pieces;
  for (const linestr &ring : rings) {
    std::vector<linestr> splits =
        findIntersectionsLineString(geometry::LineString(ring), grid);
    std::vector<linestr> pieces;
    for (linestr &split : splits) {
      split.erase(std::unique(split.begin(), split.end()), split.end());
      if (split.size() < 2) {
        continue;
      }
      // Crossings that pass exactly through a vertex are not always
      // detected (the crossing at the vertex is lost to rounding), so break
      // pieces at every interior vertex which lies on a grid line. Breaking
      // at a vertex which only touches a grid line is harmless: the two
      // chains reconnect seamlessly when their cell is assembled.
      linestr piece = {split.front()};
      for (std::size_t k = 1; k < split.size(); k++) {
        piece.push_back(split[k]);
        if (k + 1 < split.size()) {
          Coord g = grid.world_to_grid * split[k];
          if (std::fabs(g.x - std::round(g.x)) < ON_GRIDLINE_TOL ||
              std::fabs(g.y - std::round(g.y)) < ON_GRIDLINE_TOL) {
            pieces.push_back(std::move(piece));
            piece = {split[k]};
          }
        }
      }
      pieces.push_back(std::move(piece));
    }
    if (pieces.empty()) {
      continue;
    }

    // The ring start is generally partway through a cell, leaving the last
    // and first pieces as two halves of the same chain: merge them when
    // they belong to the same cell. When the ring happens to start at a
    // crossing point they belong to different cells and stay separate; when
    // it starts at a point which merely touches a grid line, keep the break
    // (chains must only meet the cell border at their endpoints).
    if (pieces.size() >= 2 && pieces.back().back() == pieces.front().front()) {
      Coord g = grid.world_to_grid * pieces.front().front();
      bool junction_on_gridline =
          std::fabs(g.x - std::round(g.x)) < ON_GRIDLINE_TOL ||
          std::fabs(g.y - std::round(g.y)) < ON_GRIDLINE_TOL;
      if (!junction_on_gridline && pieceCell(pieces.back(), 1.0, grid) ==
                                       pieceCell(pieces.front(), 1.0, grid)) {
        linestr merged = std::move(pieces.back());
        merged.insert(merged.end(), pieces.front().begin() + 1,
                      pieces.front().end());
        pieces.front() = std::move(merged);
        pieces.pop_back();
      }
    }

    for (linestr &points : pieces) {
      Piece piece;
      piece.points = std::move(points);
      bool exactly_closed = piece.points.front() == piece.points.back();
      bool nearly_closed =
          utils::almost_equal(piece.points.front().x, piece.points.back().x,
                              cellsize) &&
          utils::almost_equal(piece.points.front().y, piece.points.back().y,
                              cellsize);
      piece.closed = exactly_closed || nearly_closed;
      if (piece.closed) {
        // snap closed
        piece.points.back() = piece.points.front();
        if (piece.points.size() < 4) {
          continue;
        }
        piece.area2 = ringSignedArea2(piece.points);
        // bucket by the side the piece's own interior is on
        double side = (piece.area2 < 0) ? -1.0 : 1.0;
        CellIndex cell = pieceCell(piece.points, side, grid);
        cell_pieces[cell].push_back(std::move(piece));
      } else {
        // an open chain: the polygon interior is on the left
        CellIndex cell = pieceCell(piece.points, 1.0, grid);
        CellBorder border(cell, grid);
        piece.t_in = border.parameter(piece.points.front());
        piece.t_out = border.parameter(piece.points.back());
        cell_pieces[cell].push_back(std::move(piece));
      }
    }
  }

  // Assemble the pieces covering each cell crossed by a ring
  std::vector<geometry::Polygon> results;
  for (auto &entry : cell_pieces) {
    assembleCell(entry.first, entry.second, grid, results);
  }

  // Cells wholly inside the polygon become full cell boxes. Scan along each
  // row of cell centres: the crossings of the boundary with the scanline
  // pair up (0,1), (2,3)... into intervals interior to the polygon (the
  // even-odd rule, correct in the presence of holes).
  double gx_min = std::numeric_limits<double>::max();
  double gx_max = std::numeric_limits<double>::lowest();
  double gy_min = std::numeric_limits<double>::max();
  double gy_max = std::numeric_limits<double>::lowest();
  for (const linestr &ring : rings) {
    for (const Coord &point : ring) {
      Coord g = grid.world_to_grid * point;
      gx_min = std::min(gx_min, g.x);
      gx_max = std::max(gx_max, g.x);
      gy_min = std::min(gy_min, g.y);
      gy_max = std::max(gy_max, g.y);
    }
  }
  long i_min = (long)std::floor(gx_min);
  long i_max = (long)std::floor(gx_max);
  long j_min = (long)std::floor(gy_min);
  long j_max = (long)std::floor(gy_max);

  for (long j = j_min; j <= j_max; j++) {
    double level = (grid.grid_to_world * Coord(0, j + 0.5)).y;
    std::vector<double> xs = ringCrossings(rings, level);
    for (std::size_t k = 0; k + 1 < xs.size(); k += 2) {
      // columns whose cell centre lies within the interior interval
      double g_a = (grid.world_to_grid * Coord(xs[k], level)).x;
      double g_b = (grid.world_to_grid * Coord(xs[k + 1], level)).x;
      if (g_a > g_b) {
        std::swap(g_a, g_b);
      }
      long i_lo = std::max((long)std::ceil(g_a - 0.5), i_min);
      long i_hi = std::min((long)std::floor(g_b - 0.5), i_max);
      for (long i = i_lo; i <= i_hi; i++) {
        if (cell_pieces.count(CellIndex(i, j))) {
          // cells crossed by the boundary are assembled above
          continue;
        }
        results.push_back(cellBoxPolygon(CellIndex(i, j), grid));
      }
    }
  }

  return results;
}

} // namespace operations
} // namespace snail
