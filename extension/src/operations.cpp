#include <algorithm> /// copy_if
#include <string>
#include <vector>
#include "geometry.hpp"
#include "grid.hpp"
#include "operations.hpp"

namespace snail {
namespace operations {

using linestr = std::vector<geometry::Coord>;

/// Does any part of a straight line segment intersect the grid extent, either
/// partially (i.e. crossing one or more of bounding box lines) or entirely.
bool segmentIntersectsGridBounds(const geometry::Line &line,
                                 const grid::Grid &raster) {
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

  return crosses_x(xmin) || crosses_x(xmax) || crosses_y(ymin) ||
         crosses_y(ymax);
}

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
findIntersectionsLineString(geometry::LineString linestring, grid::Grid raster,
                            bool bounded) {
  linestr coords = linestring.coordinates;

  std::vector<linestr> allsplits;
  linestr linestr_piece;
  for (std::size_t i = 0; i < coords.size() - 1; i++) {
    geometry::Line line(coords.at(i), coords.at(i + 1));

    bool single_cell =
        raster.cellIndices(line.start) == raster.cellIndices(line.end);

    // If the line starts and ends in the same cell,
    // or (bounded and (segment does not intersect overall grid bounds))
    if (single_cell ||
        (bounded && !segmentIntersectsGridBounds(line, raster))) {
      // then don't split, just push back the coordinate
      linestr_piece.push_back(coords.at(i));
    } else {
      // otherwise do split this straight-line segment
      linestr intersections = raster.findIntersections(line);
      // if only splitting within grid bounds, filter the intersections
      if (bounded) {
        linestr filtered;
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
      std::vector<linestr> splits = split_linestr(linestr_piece, intersections);
      allsplits.insert(allsplits.end(), splits.begin(), splits.end());
      if (line.end == intersections.back()) {
        linestr_piece = {};
      } else {
        linestr_piece = {intersections.back()};
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

bool isOnGridLine(geometry::Coord point, Direction direction, double level,
                  double cellSize) {
  switch (direction) {
  case Direction::horizontal:
    return snail::utils::almost_equal(point.y, level, cellSize);
    return (point.y == level);
  case Direction::vertical:
    return snail::utils::almost_equal(point.x, level, cellSize);
    return (point.x == level);
  default:
    return false;
  }
}

// This aims to filter out vertices exactly on the line which should not be
// considered as actual crossing points.
//
//              |....../
//  >>-----x----o-----o-----  (don't include x)
//        /.\   |..../
//
// figure out what to do when some portion of the boundary is already
// along the grid line. This is a legitimate case for odd number of crossings:
//
//              |......|
//  >>-----o====o------o---
//        /............|
//
// Try something like "if the previous crossing (in sorted order) was also the
// immediately previous point on the exterior, discard it in favour of the
// current exterior/crossing point".
bool crossesGridLine(geometry::Coord prev, geometry::Coord next,
                     Direction direction, double level) {
  switch (direction) {
  case Direction::horizontal:
    return (prev.y <= level && next.y >= level) ||
           (prev.y >= level && next.y <= level);
  case Direction::vertical:
    return (prev.x <= level && next.x >= level) ||
           (prev.x >= level && next.x <= level);
  default:
    return false;
  }
}

/// Find the relevant grid coordinate (x or y) from cell index (row or col)
double gridCoordinate(int level, Direction direction, const grid::Grid &grid) {
  switch (direction) {
  case Direction::horizontal:
    return (grid.grid_to_world * geometry::Coord(0, level)).y;
  case Direction::vertical:
    return (grid.grid_to_world * geometry::Coord(level, 0)).x;
  default:
    return false;
  }
}

std::vector<linestr> splitAlongGridlines(linestr exterior_crossings,
                                         int min_level, int max_level,
                                         Direction direction, grid::Grid grid) {

  std::vector<geometry::Coord> crossings_on_gridline;
  std::vector<linestr> gridline_splits;
  double cell_size = std::max(grid.grid_to_world.a, grid.grid_to_world.e);

  for (int level = min_level; level <= max_level; level++) {
    // find level value in coordinates
    double level_value = gridCoordinate(level, direction, grid);

    // remove consecutive (adjacent) duplicates
    auto last =
        std::unique(exterior_crossings.begin(), exterior_crossings.end());
    exterior_crossings.erase(last, exterior_crossings.end());

    // find crossings at this level
    for (auto curr = exterior_crossings.begin();
         curr != exterior_crossings.end(); curr++) {

      // pick previous point on ring (wrap around)
      auto prev = (curr == exterior_crossings.begin())
                      ? (exterior_crossings.end() - 1)
                      : (curr - 1);

      // pick next point on ring (wrap around)
      auto next = ((curr + 1) == exterior_crossings.end())
                      ? exterior_crossings.begin()
                      : (curr + 1);

      // include if on the current line and prev/next are on opposite sides
      if (isOnGridLine(*curr, direction, level_value, cell_size) &&
          crossesGridLine(*prev, *next, direction, level_value)) {
        crossings_on_gridline.push_back(*curr);
      }
    }

    // sort crossings by x or y coordinate
    std::sort(crossings_on_gridline.begin(), crossings_on_gridline.end(),
              [direction](const geometry::Coord &a, const geometry::Coord &b) {
                switch (direction) {
                case Direction::horizontal:
                  return a.x < b.x;
                case Direction::vertical:
                  return a.y < b.y;
                default:
                  return false;
                }
              });

    if (crossings_on_gridline.size() % 2 != 0) {
      std::ostringstream msg;
      msg << "Expected even number of crossings on gridline.\n";
      msg << "  Found crossings on gridline:\n";
      for (auto c : crossings_on_gridline) {
        msg << "    c(" << c.x << "," << c.y << ")\n";
      }
      utils::Exception(msg.str());
      break;
    }

    // step through each pair of crossings (0,1) (2,3) ...
    auto itr = crossings_on_gridline.begin();
    while (itr != crossings_on_gridline.end()) {
      // Bail before trying to access beyond the end
      // Could remove this if we check earlier for even-length vector?
      if (std::next(itr) == crossings_on_gridline.end()) {
        utils::Exception("Out of range error.");
        break;
      }
      // construct a LineString along the gridline between these two crossings
      geometry::LineString segment({(*itr), (*(std::next(itr)))});

      std::vector<linestr> splits = findIntersectionsLineString(segment, grid);
      gridline_splits.insert(gridline_splits.end(), splits.begin(),
                             splits.end());

      // step forward two
      std::advance(itr, 2);
    }
    crossings_on_gridline.clear();
  }

  return (gridline_splits);
}

} // namespace operations
} // namespace snail
