#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <iterator>
#include <limits>
#include <map>
#include <utility>
#include <vector>

#include "geometry.hpp"
#include "grid.hpp"
#include "operations.hpp"

namespace snail {
namespace operations {

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

/// Piecewise decomposition of a linestring according to intersection points
std::vector<Ring> split_linestr(Ring linestring, Ring intersections) {
  // Add line start point
  linestring.push_back(intersections.at(0));
  // Loop over each intersection, and add a new feature for each
  std::vector<Ring> splits;
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
std::vector<Ring> findIntersectionsLineString(geometry::LineString linestring, grid::Grid raster,
                                                 bool bounded) {
  Ring coords = linestring.coordinates;

  std::vector<Ring> allsplits;
  Ring linestr_piece;
  for (std::size_t i = 0; i < coords.size() - 1; i++) {
    geometry::Line line(coords.at(i), coords.at(i + 1));

    bool single_cell = raster.cellIndices(line.start) == raster.cellIndices(line.end);

    // If the line starts and ends in the same cell,
    // or (bounded and (segment does not intersect overall grid bounds))
    if (single_cell || (bounded && !segmentIntersectsGridBounds(line, raster))) {
      // then don't split, just push back the coordinate
      linestr_piece.push_back(coords.at(i));
    } else {
      // otherwise do split this straight-line segment
      Ring intersections = raster.findIntersections(line);

      // if only splitting within grid bounds, filter the intersections
      if (bounded) {
        Ring filtered;
        filtered.reserve(intersections.size());
        for (std::size_t idx = 0; idx < intersections.size(); ++idx) {
          const auto &coordinate = intersections[idx];
          bool endpoint = (idx == 0) || (idx == intersections.size() - 1);
          // keep the endpoints as original coordinates in the linestring
          // and keep any split intersections from within the grid bounds
          if (endpoint || pointInBounds(coordinate, raster)) {
            filtered.push_back(coordinate);
          }
        }
        intersections = std::move(filtered);
      }

      if (intersections.size() == 1) {
        // The segment changes cell, but no crossing point was found within
        // it: the grid line it crosses passes exactly through the start or
        // the end of the segment.

        // Find which endpoint sits on the boundary between the two cells, and
        // break the running piece there.

        // u and v are line start and end
        // _idx in grid cell indices (int)
        std::tuple<int, int> u_idx = raster.cellIndices(line.start);
        std::tuple<int, int> v_idx = raster.cellIndices(line.end);
        // in grid cell coordinates (double)
        geometry::Coord u = raster.world_to_grid * line.start;
        geometry::Coord v = raster.world_to_grid * line.end;

        // Set up distance to nearest cell boundary (initialise here as max double)
        double du = std::numeric_limits<double>::max();
        double dv = std::numeric_limits<double>::max();

        // If i-index (at 0) differs
        if (std::get<0>(u_idx) != std::get<0>(v_idx)) {
          double boundary = std::max(std::get<0>(u_idx), std::get<0>(v_idx));
          du = std::min(du, std::fabs(u.x - boundary));
          dv = std::min(dv, std::fabs(v.x - boundary));
        }

        // If j-index (at 1) differs
        if (std::get<1>(u_idx) != std::get<1>(v_idx)) {
          double boundary = std::max(std::get<1>(u_idx), std::get<1>(v_idx));
          du = std::min(du, std::fabs(u.y - boundary));
          dv = std::min(dv, std::fabs(v.y - boundary));
        }

        if (du <= dv) {
          // break at the start point: close any running piece there and
          // carry on from it
          if (!linestr_piece.empty()) {
            snail::utils::append_new(linestr_piece, line.start);
            if (linestr_piece.size() >= 2) {
              allsplits.push_back(linestr_piece);
            }
          }
          linestr_piece = {line.start};
        } else {
          // break at the end point, treating it as the crossing
          if (linestr_piece.empty() || (linestr_piece.back() != line.start)) {
            linestr_piece.push_back(line.start);
          }
          linestr_piece.push_back(line.end);
          allsplits.push_back(linestr_piece);
          linestr_piece = {};
        }
      } else {
        std::vector<Ring> splits = split_linestr(linestr_piece, intersections);
        allsplits.insert(allsplits.end(), splits.begin(), splits.end());
        if (line.end == intersections.back()) {
          linestr_piece = {};
        } else {
          linestr_piece = {intersections.back()};
        }
      }
    }
  }

  if (linestr_piece.size() > 0) {
    linestr_piece.push_back(coords.back());
    allsplits.push_back(linestr_piece);
  }

  return allsplits;
}

bool pointInBounds(const geometry::Coord &pt, const grid::Grid &raster) {
  auto ll = raster.ll;
  auto ur = raster.ur;

  return pt.x >= ll.x && pt.x <= ur.x && pt.y >= ll.y && pt.y <= ur.y;
};

// The namespace below contains utilities to split a polygon along the lines of
// a raster grid, assembling the resulting pieces cell by cell. The approach:
// - split each ring of the polygon at every grid line crossing (using
//   findIntersectionsLineString above), giving ring pieces that each lie within
//   a single cell, with endpoints on the cell border
// - bucket the pieces by cell
// - within each cell, trace the boundary of each covered region: follow a ring
//   piece from where it enters the cell to where it leaves, then walk along the
//   cell border (passing corners) to the next piece, until the ring closes.
//   This basically follows the Weiler-Atherton clipping algorithm [1]
//   specialised to a grid cell.
// - cells wholly inside the polygon (no boundary pieces) become full cell
//   boxes, found with a parity-safe scanline through each row of cell centres
//
// [1] https://en.wikipedia.org/wiki/Weiler%E2%80%93Atherton_clipping_algorithm
namespace polygon_utils {


constexpr int border_corner_count = 4;
constexpr double border_period = 4.0;
constexpr double ring_inside_offset = 1e-6;

/// Tolerance (in cell units) within which a coordinate is considered to lie
/// on a grid line when assigning pieces to cells
constexpr double gridline_tolerance = 1e-6;

/// Tolerance when comparing positions along a cell border (the border
/// projection runs from 0 to 4, one unit per cell edge)
constexpr double border_tolerance = 1e-9;

/// Even-odd test for a point inside a closed ring. Points on the boundary
/// may be classified either way.
/// See https://en.wikipedia.org/wiki/Even%E2%80%93odd_rule
bool pointInRing(const Coord p, const Ring &ring) {
  bool inside = false;
  for (auto current = ring.begin(), next = std::next(current);
       next != ring.end(); ++current, ++next) {
    const Coord &a = *current;
    const Coord &b = *next;
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
std::vector<double> ringCrossings(const std::vector<Ring> &rings, double level) {
  std::vector<double> xs;
  for (const Ring &ring : rings) {
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
  Ring points;
  /// closed pieces are rings that never cross a grid line (or loops pinched
  /// at a single border point); open pieces are chains crossing the cell
  bool closed = false;
  /// twice the signed area (meaningful for closed pieces)
  double area2 = 0;
  /// border projection of the first and last point (open pieces only)
  double t_in = 0;
  double t_out = 0;
  bool used = false;
};

/// Border geometry of a single grid cell: corner coordinates and a
/// counter-clockwise (in world orientation) border projection
struct CellBorder {
  long ci;
  long cj;
  /// grid-space corner offsets in counter-clockwise (world) walk order
  std::array<int, border_corner_count> du{};
  std::array<int, border_corner_count> dv{};
  const grid::Grid *grid;

  CellBorder(const CellIndex index, const grid::Grid &grid) : ci(index.first), cj(index.second), grid(&grid) {
    double det = grid.grid_to_world.a * grid.grid_to_world.e - grid.grid_to_world.b * grid.grid_to_world.d;
    if (det >= 0) {
      // u,v coordinates describe a counter-clockwise walk, u as col index, v as
      // row index, assuming grid axes agree with world orientation:

      // 0,1 <- 1,1
      //         ^
      // 0,0 -> 1,0
      du = {0, 1, 1, 0};
      dv = {0, 0, 1, 1};
    } else {
      // grid axes are mirrored (e.g. north-up raster with row index
      // increasing southwards) - reverse the walk to stay counter-clockwise
      // in world orientation
      du = {0, 0, 1, 1};
      dv = {0, 1, 1, 0};
    }
  }

  /// World coordinates of corner k (0 <= k < 4) along the border walk
  Coord corner(int k) const {
    return grid->grid_to_world *
           Coord(static_cast<double>(ci + du.at(k)), static_cast<double>(cj + dv.at(k)));
  }

  /// Project a point (in world coordinates) onto cell border
  ///
  /// Returns a value in [0, 4) corresponding to the distance walked around the
  /// cell to reach the nearest point on the border, starting from the cell
  /// index corner and walking counter-clockwise.
  double project(const Coord p) const {
    Coord g = grid->world_to_grid * p;
    double u = std::clamp(g.x - static_cast<double>(ci), 0.0, 1.0);
    double v = std::clamp(g.y - static_cast<double>(cj), 0.0, 1.0);
    // d is distance of g to cell border
    double d_min = std::numeric_limits<double>::max();
    // t is projection of g onto cell border
    double t_min = 0;
    // for each edge, walking around the cell border
    for (int k = 0; k < border_corner_count; k++) {
      int u0 = du.at(k);
      int v0 = dv.at(k);
      int u1 = du.at((k + 1) % border_corner_count);
      int v1 = dv.at((k + 1) % border_corner_count);
      double d = 0.0;
      double s = 0.0;
      if (u0 == u1) {
        // edge at constant u
        d = std::fabs(u - u0);
        s = (v - v0) / (double)(v1 - v0);
      } else {
        // edge at constant v
        d = std::fabs(v - v0);
        s = (u - u0) / (double)(u1 - u0);
      }
      s = std::clamp(s, 0.0, 1.0);
      if (d < d_min) {
        d_min = d;
        t_min = k + s;
      }
    }
    if (t_min >= border_period) {
      t_min -= border_period;
    }
    return t_min;
  }
};

/// Coordinates of the longest segment of a piece
geometry::Line longestSegment(const Ring &points) {
  geometry::Line segment(*points.begin(), *std::next(points.begin()));
  double l_max = -1.0;
  for (auto u = points.begin(), v = std::next(u); v != points.end(); ++u, ++v) {
    double dx = v->x - u->x;
    double dy = v->y - u->y;
    // l is segment length squared
    double l = dx * dx + dy * dy;
    if (l > l_max) {
      l_max = l;
      segment = geometry::Line(*u, *v);
    }
  }
  return segment;
}

/// The cell that a ring piece belongs to. Pieces lie within a single cell,
/// so in general this is the cell containing the midpoint of the longest
/// segment. When that midpoint lies on a grid line (the piece runs along the
/// line), take the cell on the requested side of the direction of travel:
/// side_sign +1 for the left (in world orientation), -1 for the right.
CellIndex pieceCell(const Ring &points, double side_sign, const grid::Grid &grid) {
  geometry::Line segment = longestSegment(points);
  geometry::Line grid_segment = grid.world_to_grid * segment;
  Coord mid = grid_segment.midPoint();

  // Normal towards the requested side, in grid space. If the transform
  // mirrors orientation, the world-space left appears on the right in grid
  // space (holds component-wise for axis-aligned transforms).
  double det = grid.grid_to_world.a * grid.grid_to_world.e - grid.grid_to_world.b * grid.grid_to_world.d;
  double mirror = (det < 0) ? -1.0 : 1.0;
  double nx = -(grid_segment.end.y - grid_segment.start.y) * side_sign * mirror;
  double ny = (grid_segment.end.x - grid_segment.start.x) * side_sign * mirror;

  long ci = 0;
  long cj = 0;
  double rx = std::round(mid.x);
  if (std::fabs(mid.x - rx) < gridline_tolerance) {
    ci = (nx > 0) ? (long)rx : (long)rx - 1;
  } else {
    ci = (long)std::floor(mid.x);
  }
  double ry = std::round(mid.y);
  if (std::fabs(mid.y - ry) < gridline_tolerance) {
    cj = (ny > 0) ? (long)ry : (long)ry - 1;
  } else {
    cj = (long)std::floor(mid.y);
  }
  return CellIndex(ci, cj);
}

/// A full cell as a polygon (counter-clockwise, closed)
geometry::Polygon cellBoxPolygon(const CellIndex index, const grid::Grid &grid) {
  CellBorder border(index, grid);
  Ring ring = {border.corner(0), border.corner(1), border.corner(2), border.corner(3), border.corner(0)};
  return geometry::Polygon{ring, {}};
}

/// A point just inside a closed ring piece: the midpoint of its longest
/// segment, nudged towards the ring's own interior
Coord ringInsidePoint(const Ring &points, double area2) {
  geometry::Line segment = longestSegment(points);
  Coord mid = segment.midPoint();
  double length = segment.length();
  if (length == 0) {
    return mid;
  }
  // the ring interior is on the left if area is positive
  double sign = (area2 > 0) ? 1.0 : -1.0;
  double delta = sign * ring_inside_offset;
  return Coord(mid.x - (segment.end.y - segment.start.y) * delta,
               mid.y + (segment.end.x - segment.start.x) * delta);
}

/// Split a traced ring into simple rings at any repeated (pinch) vertices,
/// so regions that touch at a single point become separate rings
void splitAtRepeatedVertices(const Ring &ring, std::vector<Ring> &out) {
  std::size_t n = ring.size();
  for (std::size_t i = 0; i + 1 < n; i++) {
    for (std::size_t j = i + 2; j < n; j++) {
      if (i == 0 && j == n - 1) {
        // the closing vertex pair does not pinch the ring
        continue;
      }
      if (ring[i] == ring[j]) {
        auto i_offset = static_cast<std::ptrdiff_t>(i);
        auto j_offset = static_cast<std::ptrdiff_t>(j);
        Ring loop(ring.begin() + i_offset, ring.begin() + j_offset + 1);
        Ring rest(ring.begin(), ring.begin() + i_offset + 1);
        rest.insert(rest.end(), ring.begin() + j_offset + 1, ring.end());
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
void assembleCell(const CellIndex index, std::vector<Piece> &pieces, const grid::Grid &grid,
                  std::vector<geometry::Polygon> &results) {
  CellBorder border(index, grid);

  std::vector<Piece *> chains;
  std::vector<Piece *> closed_holes;
  std::vector<geometry::Polygon> faces;
  for (Piece &piece : pieces) {
    if (!piece.closed) {
      chains.push_back(&piece);
    } else if (piece.area2 > 0) {
      // a ring that never leaves the cell is a face in itself
      faces.push_back(geometry::Polygon{piece.points, {}});
    } else if (piece.area2 < 0) {
      // a closed negative-area ring is a hole
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
  double cellsize = std::max(std::fabs(grid.grid_to_world.a), std::fabs(grid.grid_to_world.e));
  auto append_point = [cellsize](Ring &ring, const Coord &point) {
    if (!ring.empty() && utils::almost_equal(ring.back().x, point.x, cellsize) &&
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
    Ring ring;
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
      double best_dt = std::numeric_limits<double>::max();
      double start_dt = std::numeric_limits<double>::max();
      for (Piece *candidate : chains) {
        if (candidate->used && candidate != start) {
          continue;
        }
        double dt = std::fmod(candidate->t_in - t, border_period);
        if (dt < 0) {
          dt += border_period;
        }
        // an entry point coinciding with the exit is at distance zero, not
        // a full lap
        if (dt > border_period - border_tolerance) {
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
      if (start_dt <= best_dt + border_tolerance) {
        best_dt = start_dt;
        next = start;
      }

      // Walk the border, adding any corners passed on the way, and noding
      // at any other chain endpoints passed - so that regions which pinch
      // together at a point on the border can be separated below
      std::vector<std::pair<double, Coord>> passed;
      double k = std::floor(t) + 1.0;
      for (int step = 0; step < border_corner_count; step++) {
        double d_corner = k - t;
        if (d_corner >= best_dt - border_tolerance) {
          break;
        }
        if (d_corner > border_tolerance) {
          passed.push_back(std::make_pair(
              d_corner, border.corner(static_cast<int>(std::fmod(k, border_period)))));
        }
        k += 1.0;
      }
      for (const std::pair<double, Coord> &endpoint : endpoints) {
        double dt = std::fmod(endpoint.first - t, border_period);
        if (dt < 0) {
          dt += border_period;
        }
        if (dt > border_tolerance && dt < best_dt - border_tolerance) {
          passed.push_back(std::make_pair(dt, endpoint.second));
        }
      }
      std::sort(passed.begin(), passed.end(),
                [](const std::pair<double, Coord> &a, const std::pair<double, Coord> &b) {
                  return a.first < b.first;
                });
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
      std::vector<Ring> simple_rings;
      splitAtRepeatedVertices(ring, simple_rings);
      for (const Ring &simple : simple_rings) {
        if (simple.size() >= 4 && geometry::ringTwiceSignedArea(simple) > 0) {
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
          double area2 = geometry::ringTwiceSignedArea(faces[i].exterior);
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

} // namespace polygon_utils

/// Split a valid polygon along the lines of a raster grid.
///
/// Rings are passed exterior first, then any interior rings (holes); each ring
/// may be closed or open (closure is applied).
///
/// In future, we may expect oriented rings: exterior counter-clockwise, holes
/// clockwise, so the polygon interior is always on the left of the direction of
/// travel. `shapely.orient_polygons` provides this.
///
/// Returns one polygon piece per covered region of a grid cell (could be more
/// than one piece per cell if concave); every piece lies within a single cell.
std::vector<geometry::Polygon> splitPolygonGrid(const std::vector<Ring> &rings_in,
                                                const grid::Grid &grid) {
  double cellsize = std::max(std::fabs(grid.grid_to_world.a), std::fabs(grid.grid_to_world.e));

  // Clean and orient rings: exterior counter-clockwise, holes clockwise, so
  // the polygon interior is always on the left of the direction of travel
  std::vector<Ring> rings;
  for (std::size_t r = 0; r < rings_in.size(); r++) {
    Ring ring = rings_in[r];
    // drop consecutive duplicate points
    ring.erase(std::unique(ring.begin(), ring.end()), ring.end());
    // close the ring
    if (!ring.empty() && !(ring.front() == ring.back())) {
      ring.push_back(ring.front());
    }
    bool is_exterior = (r == 0);
    double area2 = ring.size() >= 4 ? geometry::ringTwiceSignedArea(ring) : 0.0;
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
  // End of clean/orient section.
  if (rings.empty()) {
    return {};
  }

  // pieces grouped per cell
  std::map<polygon_utils::CellIndex, std::vector<polygon_utils::Piece>> cell_pieces;

  // Split each ring at every grid line crossing and bucket the resulting
  // pieces by the cell they lie in
  for (const Ring &ring : rings) {
    std::vector<Ring> pieces = findIntersectionsLineString(geometry::LineString(ring), grid);
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
      bool junction_on_gridline = std::fabs(g.x - std::round(g.x)) < polygon_utils::gridline_tolerance ||
                                  std::fabs(g.y - std::round(g.y)) < polygon_utils::gridline_tolerance;
      if (!junction_on_gridline &&
          polygon_utils::pieceCell(pieces.back(), 1.0, grid) == polygon_utils::pieceCell(pieces.front(), 1.0, grid)) {
        Ring merged = std::move(pieces.back());
        merged.insert(merged.end(), pieces.front().begin() + 1, pieces.front().end());
        pieces.front() = std::move(merged);
        pieces.pop_back();
      }
    }

    double cellsize_max = std::max(std::fabs(grid.grid_to_world.a), std::fabs(grid.grid_to_world.e));
    for (Ring &points : pieces) {
      polygon_utils::Piece piece;
      piece.points = std::move(points);
      bool exactly_closed = piece.points.front() == piece.points.back();
      bool nearly_closed = utils::almost_equal(piece.points.front().x, piece.points.back().x, cellsize_max) &&
                           utils::almost_equal(piece.points.front().y, piece.points.back().y, cellsize_max);
      piece.closed = exactly_closed || nearly_closed;
      if (piece.closed) {
        // snap closed
        piece.points.back() = piece.points.front();
        if (piece.points.size() < 4) {
          // four points is minimum valid closed ring
          // e.g. {A, B, C, A} for a triangle
          continue;
        }
        piece.area2 = geometry::ringTwiceSignedArea(piece.points);
        // bucket by the side the piece's own interior is on
        double side = (piece.area2 < 0) ? -1.0 : 1.0;
        polygon_utils::CellIndex cell = polygon_utils::pieceCell(piece.points, side, grid);
        cell_pieces[cell].push_back(std::move(piece));
      } else {
        // an open chain: the polygon interior is on the left
        polygon_utils::CellIndex cell = polygon_utils::pieceCell(piece.points, 1.0, grid);
        polygon_utils::CellBorder border(cell, grid);
        piece.t_in = border.project(piece.points.front());
        piece.t_out = border.project(piece.points.back());
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
  for (const Ring &ring : rings) {
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
    // NOLINTNEXTLINE(cppcoreguidelines-avoid-magic-numbers)
    double level = (grid.grid_to_world * Coord(0.0, static_cast<double>(j) + 0.5)).y;
    std::vector<double> xs = polygon_utils::ringCrossings(rings, level);
    for (std::size_t k = 0; k + 1 < xs.size(); k += 2) {
      // columns whose cell centre lies within the interior interval
      double g_a = (grid.world_to_grid * Coord(xs[k], level)).x;
      double g_b = (grid.world_to_grid * Coord(xs[k + 1], level)).x;
      if (g_a > g_b) {
        std::swap(g_a, g_b);
      }
      // NOLINTNEXTLINE(cppcoreguidelines-avoid-magic-numbers)
      long i_lo = std::max(static_cast<long>(std::ceil(g_a - 0.5)), i_min);
      // NOLINTNEXTLINE(cppcoreguidelines-avoid-magic-numbers)
      long i_hi = std::min(static_cast<long>(std::floor(g_b - 0.5)), i_max);
      for (long i = i_lo; i <= i_hi; i++) {
        if (cell_pieces.count(polygon_utils::CellIndex(i, j))) {
          // cells crossed by the boundary are assembled above
          continue;
        }
        results.push_back(polygon_utils::cellBoxPolygon(polygon_utils::CellIndex(i, j), grid));
      }
    }
  }

  return results;
}

} // namespace operations
} // namespace snail
