#ifndef TRANSFORM_H
#define TRANSFORM_H

#include "exceptions.hpp"
#include "geom.hpp"

struct Affine {
    double a;
    double b;
    double c;
    double d;
    double e;
    double f;

    /// Default construct identity transform
    Affine()
        :a{1}, b{0}, c{0}, d{0}, e{1}, f{0} {};

    /// Construct with six parameters, first two rows of 3x3 matrix
    Affine(double a, double b, double c, double d, double e, double f)
        :a{a}, b{b}, c{c}, d{d}, e{e}, f{f} {};

    /// Construct from GDALGeoTransform ordering of parameters
    static Affine from_gdal(double c, double a, double b, double f, double d, double e) {
        return Affine(a, b, c, d, e, f);
    }

    /// Invert transform
    /// see https://en.wikipedia.org/wiki/Invertible_matrix#Inversion_of_3_%C3%97_3_matrices
    /// and simplify since bottom row (g, h, i) is always (0, 0, 1)
    Affine operator~() {
        double determinant = a * e - b * d;
        if (determinant == 0) {
            utils::Exception("The transform is not invertible");
        }
        double inverse_a = e / determinant;
        double inverse_b = -b / determinant;
        double inverse_d = -d / determinant;
        double inverse_e = a / determinant;
        double inverse_c = -c * inverse_a - f * inverse_b;
        double inverse_f = -c * inverse_d - f * inverse_e;
        return Affine(inverse_a, inverse_b, inverse_c, inverse_d, inverse_e, inverse_f);
    }

    /// Apply transform to a 2-dimensional point
    geometry::Vec2<double> operator*(const geometry::Vec2<double> &p) const {
        return geometry::Vec2<double>(
            p.x * a + p.y * b + c,
            p.x * d + p.y * e + f
        );
    }

    geometry::Vec2<double> operator*(const geometry::Vec2<int> &p) const {
        return geometry::Vec2<double>(
            p.x * a + p.y * b + c,
            p.x * d + p.y * e + f
        );
    }

};

#endif // TRANSFORM_H
