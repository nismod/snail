#include "features.h"
#include "geom.h"
#include "raster.h"

using linestr = std::vector<geometry::Vec2<double>>;

// Piecewiese decomposition of a linestring accoring to intersections points
std::vector<linestr> split_linestr(linestr linestring, linestr intersections) {
  // Add line start point
  linestring.push_back(intersections.at(0));
  // Loop over each intersection, and add a new feature for each...
  std::vector<linestr> splits;
  for(std::size_t j=1; j<intersections.size(); j++){
    // Add the crossing point to the cleaned features geometry...
    linestring.push_back(intersections.at(j));
    splits.push_back(linestring);
    linestring.clear();
    linestring.push_back(intersections.at(j));
  }
  return(splits);
}

std::vector<linestr> findIntersectionsLineString(Feature feature, Ascii raster) {
  linestr linestring = feature.geometry;

  std::vector<linestr> allsplits;
  linestr linestr_piece;
  for(std::size_t i=0; i<linestring.size()-1; i++){
    geometry::Line2<double> line(linestring.at(i), linestring.at(i+1));
    
    // If the line starts and ends in different cells, it needs to be cleaned...
    if(raster.cellIndex(line.start) != raster.cellIndex(line.end)){
      linestr intersections = raster.findIntersections(line);
      std::vector<linestr> splits = split_linestr(linestr_piece, intersections);
      allsplits.insert(allsplits.end(), splits.begin(), splits.end());
      linestr_piece = {intersections.back()};
    } else {
      linestr_piece.push_back(linestring.at(i));
    }
  }
  linestr_piece.push_back(linestring.back());
  allsplits.push_back(linestr_piece);

  return(allsplits);      
}
