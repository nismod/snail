#include <algorithm> /// copy_if
#include <iterator>  /// advance
#include <string>
#include <vector>
#include "geometry.hpp"
#include "grid.hpp"
#include "operations.hpp"

namespace snail {
namespace operations {

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
      std::vector<linestr> splits = split_linestr(linestr_piece, intersections);
      allsplits.insert(allsplits.end(), splits.begin(), splits.end());
      if (line.end == intersections.back()) {
        linestr_piece = {};
      } else {
        linestr_piece = {intersections.back()};
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

bool isOnGridLine(geometry::Coord point, Direction direction, double level) {
  switch (direction) {
  case Direction::horizontal:
    return snail::utils::almost_equal(point.y, level, 2);
    return (point.y == level);
  case Direction::vertical:
    return snail::utils::almost_equal(point.x, level, 2);
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
// TODO figure out what to do when some portion of the boundary is already
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
      if (isOnGridLine(*curr, direction, level_value) &&
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
      utils::Exception("Expected even number of crossings on gridline.");
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
