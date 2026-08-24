#include "geometry.hpp"

namespace snail {
namespace geometry {

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

} // namespace geometry
} // namespace snail
