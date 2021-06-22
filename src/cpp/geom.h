#ifndef GEOM_H
#define GEOM_H

#include <cmath>

#include "utils.h"

namespace geometry{

  // A templated 2D vector representation
  template <typename T>
    struct Vec2{
      T x;  // X data
      T y;  // Y data
      Vec2<T> operator+(const Vec2<T> &a)  const { return Vec2<T>(x + a.x, y + a.y); } // Add two Vec2<T> together
      Vec2<T> operator-(const Vec2<T> &a)  const { return Vec2<T>(x - a.x, y - a.y); } // Subtract one Vec2<T> from another
      bool    operator==(const Vec2<T> &a) const { return x == a.x && y == a.y;}      // Compare Vec2<T> for equality
      Vec2<T> operator/( const double  a)  const { return Vec2<T>(x / a, y / a); }     // Divide a Vec2 by another
      // Deafult constructor
      Vec2(void){}
      // Construct a Vec2<T> from a pair of T
    Vec2(const T x, const T y) : x(x), y(y) {}
      // Helper function to calculate the length of a Vec2<T>, cast to a double
      inline double length(void) const { return (double) sqrt(x*x + y*y); }
    };

  // A templated 2D line representation
  template <typename T>
    struct Line2{
      Vec2<T> start;  // Start point of the line, in 2D space
      Vec2<T> end;    // End point of the line, in 2D space
      inline Vec2<T> midPoint(void) const { return Vec2<T>((end.x + start.x)/2, (end.y + start.y)/2); }                          // Method to calculate the midpoint of a line
      inline double length(void) const { double dx = end.x - start.x; double dy = end.y - start.y; return sqrt(dx*dx + dy*dy); } // Method to calculate the GEOMETRIC length of a line (NOTE: for lines in spherical projection, use haversine formula to find great-circle length)
      inline double bearing(void) const { return atan2(end.x - start.x, end.y - start.y); }                                      // Method to calculate the bearing of a line...
      // Default constructor
      Line2(void){}
      // Construct a line from two points
    Line2(const Vec2<T> start, const Vec2<T> end) : start(start), end(end){}
      // Method to calculate whether two Line2<T> objects cross in space (assumed to be in the sampe plane)...
      bool linesCross(const Line2<T> l) const {
        T  dx =   end.x -   start.x; T  dy =   end.y -   start.y;
        T _dx = l.end.x - l.start.x; T _dy = l.end.y - l.start.y;
        T denom = -_dx * dy + dx * _dy;
        T s = (-dy * (start.x - l.start.x) +  dx * (start.y - l.start.y)) / denom;
        T t = (_dx * (start.y - l.start.y) - _dy * (start.x - l.start.x)) / denom;

        if (s >= 0 && s <= 1 && t >= 0 && t <= 1)
          return true; // NOTE: If needed, lines cross as this point Vec2<double> intersection(p0_x + (t * s1_x), p0_y + (t * s1_y));

        return false;
      }
      // Method to calculate whether a Line2<T> crosses a line comprising a pair of Vec2<T> points...
      bool linesCross(const Vec2<T> start, Vec2<T> end){
        return linesCross(Line2<T>(start, end));
      }
    };

  // Haversine formula - calculate the distance between two points on the surface of the earth, using great-circles (NOTE: Only use with projected data)...
  inline double haversine(Line2<double> line){
    double dLat_2 = ((line.end.y - line.start.y) * utils::toRad) / 2.0;
    double dLon_2 = ((line.end.x - line.start.x) * utils::toRad) / 2.0;

    double a = sin(dLat_2)*sin(dLat_2) + cos(line.start.y * utils::toRad)*cos(line.end.y * utils::toRad) * sin(dLon_2) * sin(dLon_2);
    double c = 2.0 * atan2(sqrt(a), sqrt(1-a));

    return utils::R*c;
  }
} // geometry

#endif //GEOM_H
