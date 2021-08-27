#ifndef GEOM_H
#define GEOM_H

#include <cmath>
#include <tuple>
#include <vector>

#include "utils.hpp"

namespace snail {
namespace geometry {

/// A templated 2D vector representation
template <typename T> struct Vec2 {
  T x;
  T y;
  // Add two Vec2<T> together
  Vec2<T> operator+(const Vec2<T> &a) const {
    return Vec2<T>(x + a.x, y + a.y);
  }
  // Subtract one Vec2<T> from another
  Vec2<T> operator-(const Vec2<T> &a) const {
    return Vec2<T>(x - a.x, y - a.y);
  }
  // Compare Vec2<T> for equality
  bool operator==(const Vec2<T> &a) const { return x == a.x && y == a.y; }
  // Divide a Vec2 by a constant
  Vec2<T> operator/(const double a) const { return Vec2<T>(x / a, y / a); }
  // Default constructor
  Vec2(void) {}
  // Construct a Vec2<T> from a pair of T
  Vec2(const T x, const T y) : x(x), y(y) {}
  // Construct a Vec2<T> from tuple of two T
  Vec2(const std::tuple<T, T> xy) : x(std::get<0>(xy)), y(std::get<1>(xy)) {}
  // Helper function to calculate the length of a Vec2<T>, cast to a double
  inline double length(void) const { return (double)sqrt(x * x + y * y); }
};

/// A templated 2D line representation
template <typename T> struct Line2 {
  // Start point of the line, in 2D space
  Vec2<T> start;
  // End point of the line, in 2D space
  Vec2<T> end;
  // Calculate the midpoint of a line
  inline Vec2<T> midPoint(void) const {
    return Vec2<T>((end.x + start.x) / 2, (end.y + start.y) / 2);
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
  /// Default constructor
  Line2(void) {}
  /// Construct a line from two points
  Line2(const Vec2<T> start, const Vec2<T> end) : start(start), end(end) {}
  /// Calculate whether two Line2<T> objects cross in space (assumed to
  /// be in the sample plane).
  bool linesCross(const Line2<T> l) const {
    T dx = end.x - start.x;
    T dy = end.y - start.y;
    T _dx = l.end.x - l.start.x;
    T _dy = l.end.y - l.start.y;
    T denom = -_dx * dy + dx * _dy;
    T s = (-dy * (start.x - l.start.x) + dx * (start.y - l.start.y)) / denom;
    T t = (_dx * (start.y - l.start.y) - _dy * (start.x - l.start.x)) / denom;

    // NOTE: If needed, lines cross as this point Vec2<double>
    // intersection(p0_x + (t * s1_x), p0_y + (t * s1_y));
    if (s >= 0 && s <= 1 && t >= 0 && t <= 1)
      return true;

    return false;
  }
  /// Calculate whether a Line2<T> crosses a line comprising a pair of
  /// Vec2<T> points.
  bool linesCross(const Vec2<T> start, Vec2<T> end) {
    return linesCross(Line2<T>(start, end));
  }
};

/// A templated LineString representation. The list of points in coordinates
/// defines a series of connected straight-line segments.
template <typename T> struct LineString {
  std::vector<geometry::Vec2<T>> coordinates;

  LineString(const std::vector<geometry::Vec2<T>> coordinates)
      : coordinates(coordinates) {}
};

/// Haversine formula - calculate the distance between two points on the surface
/// of the earth, using great-circles (NOTE: Only use with data in
/// latitude/longitude coordinates).
inline double haversine(Line2<double> line) {
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

#endif // GEOM_H
