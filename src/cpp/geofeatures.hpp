#ifndef FEATURES_H
#define FEATURES_H

#include <string>
#include <vector>

#include "geom.hpp"

// Basic feature structure...
struct Feature{
  std::vector<geometry::Vec2<double>>  geometry;    // Vector of points defining the geometry of the feature...
  std::vector<std::string>             attributes;  // Vector of attributes associated with the feature...
  geometry::Vec2<double>               ll;          // Lower-left of the feature's bounding-box (calculated)
  geometry::Vec2<double>               ur;          // Upper-right of the feature's bounding-box (calculated)
  // Helper method to get the geometric mid-point of a feature from its bounding box...
  geometry::Vec2<double> mid_point(void) const{
    return geometry::Vec2<double>((ll.x + ur.x)/2,(ll.y + ur.y)/2);
  }
  // Helper function to add a valid BB to a line feature...
  void addBB(void){
    // Give the ll and ur points a starting point...
    ll = geometry.at(0);
    ur = geometry.at(0);

    // Add each point to the internal representation...
    for(auto p : geometry){
      // Identify the min and max x, y values of the geometry...
      ll.x = std::min(p.x, ll.x);
      ur.x = std::max(p.x, ur.x);
      ll.y = std::min(p.y, ll.y);
      ur.y = std::max(p.y, ur.y);
    }
  }
};

// Structure to represent a polygon in 2D space (inherits from Feature)
struct Poly2 : Feature{
  geometry::Vec2<double> pOut;  // A point known to be outside of the polygon
  // Broad-phase (thence quick) test of whether a point in the bounding box of the Poly2...
  bool AABB(const geometry::Vec2<double> p) const {
    bool xOverlap = p.x >= ll.x && p.x <= ur.x;
    bool yOverlap = p.y >= ll.y && p.y <= ur.y;
    return xOverlap && yOverlap;
  }
  // Narrow-phase (thence slow) test of whether a point is inside the perrimeter of a Poly2...
  bool inPoly(const geometry::Vec2<double> p) const {
    // Form a point from the incoming point and the stored point known to be outside the feature...
    geometry::Line2<double> l(pOut, p);

    // Store the number of crossings (we need to check if the answer is even or odd, see below)...
    int numCrossings=0;

    // Loop over each line in the line-loop (by definition, this must be closed - see constructor)...
    for(std::size_t i=0; i<geometry.size()-1; i++){
      geometry::Line2<double> m(geometry.at(i), geometry.at(i+1));
      if(l.linesCross(m))
	numCrossings++;
    }

    // An even number of crossings means the incoming point MUST be on the same side of the feature's perimeter as the
    // point KNOWN to be outside the geometry, while an odd number means it must be on the opposite side (i.e. inside):
    return !(numCrossings%2 == 0);
  }
  // Initialise a Poly2 from a vector of points...
  Poly2(const std::vector<geometry::Vec2<double>> pts){
    // We want to have a point known to be outside of the polygon - will calculate that from the ll point in the geom...
    pOut = pts.at(1);

    // Give the ll and ur points a starting point...
    ll = pts.at(1);
    ur = pts.at(1);

    // Add each point to the internal representation...
    for(auto p : pts){
      geometry.push_back(p);
      // Identify the min and max x, y values of the geometry...
      ll.x = std::min(p.x, ll.x);
      ur.x = std::max(p.x, ur.x);
      ll.y = std::min(p.y, ll.y);
      ur.y = std::max(p.y, ur.y);
    }

    // Check if the point is a closed loop...
    geometry::Line2<double> l(geometry.at(0), geometry.at(geometry.size()-1));
    // If not, close the loop with a new point...
    if(l.length() > 0)
      geometry.push_back(geometry.at(0));

    // Finally, identify a point that is a little bit outside of the ll corner of the BB of the Poly2.
    pOut = ll - geometry::Vec2<double>(0.0001,0.0001);
  }
};

#endif //FEATURES_H
