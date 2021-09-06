#ifndef TRANSFORM_H
#define TRANSFORM_H

#include <vector>

#include "utils.hpp"
#include "geometry.hpp"

namespace snail {
namespace transform {

/**
 * Affine transform
 *
 * Represents a 2D transform, can be a linear transformation plus translation,
 * including scaling, rotation, translation or shear.
 *
 * This holds all the metadata needed to define world-to-grid or grid-to-world
 * coordinate transformations for a raster::Grid.
 */
struct Affine {
  double a;
  double b;
  double c;
  double d;
  double e;
  double f;

  /// Default construct identity transform
  Affine() : a{1}, b{0}, c{0}, d{0}, e{1}, f{0} {};

  /// Construct with six parameters, first two rows of 3x3 matrix
  Affine(double a, double b, double c, double d, double e, double f)
      : a{a}, b{b}, c{c}, d{d}, e{e}, f{f} {};

  /// Same as above but construct from vector
  Affine(std::vector<double> c)
      : a{c[0]}, b{c[1]}, c{c[2]}, d{c[3]}, e{c[4]}, f{c[5]} {};

  /// Construct from GDALGeoTransform ordering of parameters
  /// see
  /// https://gdal.org/api/gdaldataset_cpp.html#_CPPv4N11GDALDataset15GetGeoTransformEPd
  static Affine from_gdal(double c, double a, double b, double f, double d,
                          double e) {
    return Affine(a, b, c, d, e, f);
  }

  /// Invert transform
  /// see
  /// https://en.wikipedia.org/wiki/Invertible_matrix#Inversion_of_3_%C3%97_3_matrices
  /// and simplify since bottom row (g, h, i) is always (0, 0, 1)
  Affine operator~() {
    double determinant = a * e - b * d;
    if (determinant == 0) {
      utils::Exception("The transform is not invertible");
    }
    double ideterminant = 1 / determinant;
    double inverse_a = e * ideterminant;
    double inverse_b = -b * ideterminant;
    double inverse_d = -d * ideterminant;
    double inverse_e = a * ideterminant;
    double inverse_c = -c * inverse_a - f * inverse_b;
    double inverse_f = -c * inverse_d - f * inverse_e;
    return Affine(inverse_a, inverse_b, inverse_c, inverse_d, inverse_e,
                  inverse_f);
  }

  /// Apply transform to a 2-dimensional point
  geometry::Coord operator*(const geometry::Coord &p) const {
    return geometry::Coord(p.x * a + p.y * b + c, p.x * d + p.y * e + f);
  }
};

} // namespace transform
} // namespace snail
#endif // TRANSFORM_H
