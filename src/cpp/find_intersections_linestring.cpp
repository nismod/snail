#include "features.h"
#include "geom.h"
#include "raster.h"

using linestr = std::vector<geometry::Vec2<double>>;

// Splits a Line2 LINE into pieces contained within cells of raster data RASTER.
// LineString being splitted SPLIT_LINESTRING ends at intersection point.
std::vector<linestr> split(Ascii raster, geometry::Line2<double> line, linestr &split_linestring) {
  // Recover the points of intersection of the line with the grid / graticule lines...
  linestr intersections = raster.findIntersections(line);

  // Add the start of the line to the cleaned feature...
  split_linestring.push_back(line.start);

  // Loop over each intersection, and add a new feature for each...
  std::vector<linestr> splits;
  for(std::size_t j=1; j<intersections.size(); j++){
    // Add the crossing point to the cleaned features geometry...
    split_linestring.push_back(intersections.at(j));
    splits.push_back(split_linestring);
    split_linestring.clear();
    split_linestring.push_back(intersections.at(j));
  }
  return(splits);
}

std::vector<linestr> findIntersectionsLineString(Feature feature, Ascii raster) {
  linestr linestring = feature.geometry;

  std::vector<linestr> split_linestrings;
  linestr split_linestring;
  for(std::size_t i=0; i<linestring.size()-1; i++){
    geometry::Line2<double> line(linestring.at(i), linestring.at(i+1));
    
    // If the line starts and ends in different cells, it needs to be cleaned...
    if(raster.cellIndex(line.start) != raster.cellIndex(line.end)){
      std::vector<linestr> splits = split(raster, line, split_linestring);
      split_linestrings.insert(split_linestrings.end(), splits.begin(), splits.end());
    } else {
      split_linestring.push_back(linestring.at(i));
    }
  }
  split_linestring.push_back(linestring.back());
  split_linestrings.push_back(split_linestring);

  return(split_linestrings);      
}
