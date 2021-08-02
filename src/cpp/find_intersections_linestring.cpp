#include <algorithm> /// copy_if
#include <iterator> /// advance
#include "geofeatures.hpp"
#include "geom.hpp"
#include "grid.hpp"

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
std::vector<linestr> findIntersectionsLineString(Feature feature,
                                                 Grid raster) {
  linestr linestring = feature.geometry;

  std::vector<linestr> allsplits;
  linestr linestr_piece;
  for (std::size_t i = 0; i < linestring.size() - 1; i++) {
    geometry::Line2<double> line(linestring.at(i), linestring.at(i + 1));

    // If the line starts and ends in different cells, it needs to be cleaned.
    if (raster.cellIndex(line.start) != raster.cellIndex(line.end)) {
      linestr intersections = raster.findIntersections(line);
      std::vector<linestr> splits = split_linestr(linestr_piece, intersections);
      allsplits.insert(allsplits.end(), splits.begin(), splits.end());
      if(line.end == intersections.back()) {
	linestr_piece = {};
      } else {
	linestr_piece = {intersections.back()};
      }
    } else {
      linestr_piece.push_back(linestring.at(i));
    }
  }

  if(linestr_piece.size() > 0) {
    linestr_piece.push_back(linestring.back());
    allsplits.push_back(linestr_piece);
  }

  return (allsplits);
}

std::vector<linestr> splitAlongGridlines(linestr exterior_crossings,
                                         int min_level, int max_level,
                                         Grid grid) {
  std::vector<geometry::Vec2<double>> crossings_on_gridline;
  std::vector<linestr> gridline_splits;
  for (int level = min_level; level <= max_level; level++) {
    auto it = std::copy_if(
        exterior_crossings.begin(), exterior_crossings.end(),
        std::back_inserter(crossings_on_gridline),
        [level](geometry::Vec2<double> p) { return p.y == level; });
    linestr segment(2);

    auto itr = crossings_on_gridline.begin();
    while (itr != crossings_on_gridline.end()) {
      segment[0] = (*itr);
      segment[1] = (*(std::next(itr)));

      Feature f;
      f.geometry.insert(f.geometry.begin(), segment.begin(), segment.end());
      std::vector<linestr> splits = findIntersectionsLineString(f, grid);
      gridline_splits.insert(gridline_splits.end(), splits.begin(),
                             splits.end());
      std::advance(itr, 2);
    }
    crossings_on_gridline.clear();
  }

  return (gridline_splits);
}
