#include <algorithm> /// copy_if
#include <iterator>  /// advance
#include <string>
#include <vector>
#include "geometry.hpp"
#include "grid.hpp"
#include "operations.hpp"

namespace snail {
namespace operations {

using linestr = std::vector<geometry::Vec2<double>>;

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

bool isOnGridLine(geometry::Vec2<double> point, Direction direction,
                  int level) {
  switch (direction) {
    case Direction::horizontal: return (point.y == level);
    case Direction::vertical:   return (point.x == level);
    default: return false;
  }
}

std::vector<linestr> splitAlongGridlines(linestr exterior_crossings,
                                         int min_level, int max_level,
                                         Direction direction,
                                         grid::Grid grid) {
  std::vector<geometry::Vec2<double>> crossings_on_gridline;
  std::vector<linestr> gridline_splits;
  for (int level = min_level; level <= max_level; level++) {
    // filter to find crossings at this level
    std::copy_if(
      exterior_crossings.begin(),
      exterior_crossings.end(),
      std::back_inserter(crossings_on_gridline),
      [direction, level](geometry::Vec2<double> p) {
        return isOnGridLine(p, direction, level);
      }
    );
    linestr segment_coords(2);

    // step through each pair of crossings (0,1) (2,3) ...
    auto itr = crossings_on_gridline.begin();
    while (itr != crossings_on_gridline.end()) {
      segment_coords[0] = (*itr);
      segment_coords[1] = (*(std::next(itr)));

      geometry::LineString segment(segment_coords);
      std::vector<linestr> splits = findIntersectionsLineString(segment, grid);
      gridline_splits.insert(
        gridline_splits.end(), splits.begin(), splits.end());
      std::advance(itr, 2);
    }
    crossings_on_gridline.clear();
  }

  return (gridline_splits);
}

} // namespace operations
} // namespace snail
