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
  // Divide a Coord by a constant
  Coord operator/(const double a) const { return Coord(x / a, y / a); }

  // Construct a Coord from doubles
  Coord(const double x, const double y) : x(x), y(y) {}
  // Construct a Coord from tuple of two ints
  Coord(const std::tuple<int, int> xy)
      : x(std::get<0>(xy)), y(std::get<1>(xy)) {}
  // Construct a Coord from tuple of two doubles
  Coord(const std::tuple<double, double> xy)
      : x(std::get<0>(xy)), y(std::get<1>(xy)) {}

  // Helper function to calculate the length of a Coord
  inline double length(void) const { return sqrt(x * x + y * y); }
};

/// A templated 2D line representation
struct Line {
  // Start point of the line, in 2D space
  Coord start;
  // End point of the line, in 2D space
  Coord end;
  // Calculate the midpoint of a line
  inline Coord midPoint(void) const {
    return Coord((end.x + start.x) / 2, (end.y + start.y) / 2);
  }
  // Calculate the GEOMETRIC length of a line (NOTE: for lines in
  // spherical projection, use haversine formula to find great-circle length)
  inline double length(void) const {
    double dx = end.x - start.x;
    double dy = end.y - start.y;
    return sqrt(dx * dx + dy * dy);
  }
  /// Calculate the bearing of a line.
  inline double bearing(void) const {
    return atan2(end.x - start.x, end.y - start.y);
  }

  /// Construct a line from two points
  Line(const Coord start, const Coord end) : start(start), end(end) {}

  /// Calculate whether two Line objects cross in space (assumed to
  /// be in the sample plane).
  bool linesCross(const Line l) const {
    double dx = end.x - start.x;
    double dy = end.y - start.y;
    double _dx = l.end.x - l.start.x;
    double _dy = l.end.y - l.start.y;
    double denom = -_dx * dy + dx * _dy;
    double s =
        (-dy * (start.x - l.start.x) + dx * (start.y - l.start.y)) / denom;
    double t =
        (_dx * (start.y - l.start.y) - _dy * (start.x - l.start.x)) / denom;

    // NOTE: If needed, lines cross as this point Coord
    // intersection(p0_x + (t * s1_x), p0_y + (t * s1_y));
    if (s >= 0 && s <= 1 && t >= 0 && t <= 1)
      return true;

    return false;
  }
  /// Calculate whether a Line crosses a line comprising a pair of
  /// Coord points.
  bool linesCross(const Coord start, Coord end) {
    return linesCross(Line(start, end));
  }
};

/// A templated LineString representation. The list of points in coordinates
/// defines a series of connected straight-line segments.
struct LineString {
  std::vector<geometry::Coord> coordinates;

  LineString(const std::vector<geometry::Coord> coordinates)
      : coordinates(coordinates) {}
};

/// Haversine formula - calculate the distance between two points on the surface
/// of the earth, using great-circles (NOTE: Only use with data in
/// latitude/longitude coordinates).
inline double haversine(Line line) {
  double dLat_2 = ((line.end.y - line.start.y) * utils::toRad) / 2.0;
  double dLon_2 = ((line.end.x - line.start.x) * utils::toRad) / 2.0;

  double a = sin(dLat_2) * sin(dLat_2) + cos(line.start.y * utils::toRad) *
                                             cos(line.end.y * utils::toRad) *
                                             sin(dLon_2) * sin(dLon_2);
  double c = 2.0 * atan2(sqrt(a), sqrt(1 - a));

  return utils::R * c;
}
} // namespace geometry
} // namespace snail

#endif // GEOMETRY_H
