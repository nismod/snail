#ifndef GEOMETRY_H
#define GEOMETRY_H

#include <cmath>
#include <tuple>
#include <vector>

#include "utils.hpp"

namespace snail {
namespace geometry {

/// A 2D point representation
struct Coord {
  double x;
  double y;
  // Add two Coords together
  Coord operator+(const Coord &a) const { return Coord(x + a.x, y + a.y); }
  // Subtract one Coord from another
  Coord operator-(const Coord &a) const { return Coord(x - a.x, y - a.y); }
  // Compare Coords for equality
  bool operator==(const Coord &a) const { return x == a.x && y == a.y; }
  // Compare Coords for inequality
  bool operator!=(const Coord &a) const { return x != a.x || y != a.y; }
  // Divide a Coord by a constant
  Coord operator/(const double a) const { return Coord(x / a, y / a); }

  // Default constructor: zero Coord
  Coord() : x(0.0), y(0.0) {}
  // Construct a Coord from doubles
  Coord(const double x, const double y) : x(x), y(y) {}
  // Construct a Coord from tuple of two ints
  Coord(const std::tuple<int, int> xy) : x(std::get<0>(xy)), y(std::get<1>(xy)) {}
  // Construct a Coord from tuple of two doubles
  Coord(const std::tuple<double, double> xy) : x(std::get<0>(xy)), y(std::get<1>(xy)) {}

  // Helper function to calculate the length of a Coord
  inline double length(void) const { return std::hypot(x, y); }
};

/// A 2D line segment with start and end points
struct Line {
  // Start point of the line, in 2D space
  Coord start;
  // End point of the line, in 2D space
  Coord end;
  // Calculate the midpoint of a line
  inline Coord midPoint(void) const { return Coord((end.x + start.x) / 2, (end.y + start.y) / 2); }
  // Calculate the GEOMETRIC length of a line, assuming planar
  inline double length(void) const {
    double dx = end.x - start.x;
    double dy = end.y - start.y;
    return std::hypot(dx, dy);
  }
  /// Calculate the bearing of a line.
  inline double bearing(void) const { return atan2(end.x - start.x, end.y - start.y); }

  /// Construct a line from two points
  Line(const Coord start, const Coord end) : start(start), end(end) {}
};

// A sequence of 2D points
using Ring = std::vector<Coord>;

/// Twice the signed area of a closed ring (positive if counter-clockwise).
/// This follows the shoelace formula [1]
/// Coordinates are taken relative to the first point: without this, rings
/// that are small relative to their distance from the origin (e.g. buildings
/// in geographic coordinates) lose all precision to cancellation.
/// [1] https://en.wikipedia.org/wiki/Shoelace_formula
double ringTwiceSignedArea(const Ring &ring) {
  if (ring.empty()) {
    return 0;
  }
  const Coord origin = ring.front();
  double area2 = 0;
  for (auto current = ring.begin(), next = std::next(current);
       next != ring.end(); ++current, ++next) {
    area2 += (current->x - origin.x) * (next->y - origin.y) -
             (next->x - origin.x) * (current->y - origin.y);
  }
  return area2;
}

/// A LineString. The list of points in coordinates
/// defines a series of connected straight-line segments.
struct LineString {
  std::vector<Coord> coordinates;

  LineString() : coordinates{} {}
  LineString(const std::vector<Coord> coordinates) : coordinates(coordinates) {}
};

/// A polygon: one exterior ring and zero or more interior rings (holes).
/// Rings are stored closed (first point equal to last point).
struct Polygon {
  Ring exterior;
  std::vector<Ring> interiors;

  double area() const {
    // holes are wound clockwise, so their signed area is negative
    double area2 = ringTwiceSignedArea(exterior);
    for (const Ring &interior : interiors) {
      area2 += ringTwiceSignedArea(interior);
    }
    return area2 * 0.5;
  }
};

} // namespace geometry
} // namespace snail

#endif // GEOMETRY_H
