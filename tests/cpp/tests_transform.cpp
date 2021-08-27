#include <catch2/catch.hpp>

#include "geometry.hpp"
#include "transform.hpp"

using snail::transform::Affine;
using snail::geometry::Coord;

TEST_CASE("Default is identity", "[construction]") {
    Affine(a);
    REQUIRE( a.a == 1 );
    REQUIRE( a.b == 0 );
    REQUIRE( a.c == 0 );
    REQUIRE( a.d == 0 );
    REQUIRE( a.e == 1 );
    REQUIRE( a.f == 0 );
}

TEST_CASE("Construct with values", "[construction]") {
    Affine a(1, 2, 3, 4, 5, 6);
    REQUIRE( a.a == 1 );
    REQUIRE( a.b == 2 );
    REQUIRE( a.c == 3 );
    REQUIRE( a.d == 4 );
    REQUIRE( a.e == 5 );
    REQUIRE( a.f == 6 );
}

TEST_CASE("Construct from GDAL order", "[construction]") {
    Affine a = Affine::from_gdal(3, 1, 2, 6, 4, 5);
    REQUIRE( a.a == 1 );
    REQUIRE( a.b == 2 );
    REQUIRE( a.c == 3 );
    REQUIRE( a.d == 4 );
    REQUIRE( a.e == 5 );
    REQUIRE( a.f == 6 );
}

TEST_CASE("Invert identity is identity", "[invert]") {
    Affine(a);
    Affine inverse = ~a;
    REQUIRE( inverse.a == 1 );
    REQUIRE( inverse.b == 0 );
    REQUIRE( inverse.c == 0 );
    REQUIRE( inverse.d == 0 );
    REQUIRE( inverse.e == 1 );
    REQUIRE( inverse.f == 0 );
}

TEST_CASE("Invert okay", "[invert]") {
    Affine a(1, 2, 4, 1, 4, 2);
    Affine inverse = ~a;
    REQUIRE( inverse.a == Approx(2));
    REQUIRE( inverse.b == Approx(-1));
    REQUIRE( inverse.c == Approx(-6));
    REQUIRE( inverse.d == Approx(-0.5));
    REQUIRE( inverse.e == Approx(0.5));
    REQUIRE( inverse.f == Approx(1));

    Affine a_again = ~inverse;
    REQUIRE( a_again.a == Approx(1));
    REQUIRE( a_again.b == Approx(2));
    REQUIRE( a_again.c == Approx(4));
    REQUIRE( a_again.d == Approx(1));
    REQUIRE( a_again.e == Approx(4));
    REQUIRE( a_again.f == Approx(2));
}

TEST_CASE("Invert fails with zero determinant", "[invert]") {
    Affine a(0, 0, 0, 0, 0, 0);
    REQUIRE_THROWS_WITH( ~a, "The transform is not invertible" );
    Affine b(2, 1, 0, 2, 1, 0);
    REQUIRE_THROWS_WITH( ~b, "The transform is not invertible" );
}

TEST_CASE("Invert translation", "[invert]") {
    double x = 2;
    double y = 4;
    Affine a(1, 0, x, 0, 1, y);
    Affine inverse = ~a;
    REQUIRE( inverse.a == Approx(1));
    REQUIRE( inverse.b == Approx(0));
    REQUIRE( inverse.c == Approx(-2));
    REQUIRE( inverse.d == Approx(0));
    REQUIRE( inverse.e == Approx(1));
    REQUIRE( inverse.f == Approx(-4));
}

TEST_CASE("Invert scaling", "[invert]") {
    double scale = 3;
    Affine a(scale, 0, 0, 0, scale, 0);
    Affine inverse = ~a;
    REQUIRE( inverse.a == Approx(1/scale));
    REQUIRE( inverse.b == Approx(0));
    REQUIRE( inverse.c == Approx(0));
    REQUIRE( inverse.d == Approx(0));
    REQUIRE( inverse.e == Approx(1/scale));
    REQUIRE( inverse.f == Approx(0));
}

TEST_CASE("Scale transform", "[transform]") {
    double scale = 3;
    Affine a(scale, 0, 0, 0, scale, 0);
    Coord p(0.5, 0.5);
    Coord expected(1.5, 1.5);
    auto actual = a * p;
    REQUIRE( actual.x == Approx(expected.x));
    REQUIRE( actual.y == Approx(expected.y));
    REQUIRE( actual == expected);
}

TEST_CASE("Translate transform", "[transform]") {
    double x = 2;
    double y = 4;
    Affine a(1, 0, x, 0, 1, y);
    Coord p(0.5, 0.5);
    Coord expected(2.5, 4.5);
    auto actual = a * p;
    REQUIRE( actual.x == Approx(expected.x));
    REQUIRE( actual.y == Approx(expected.y));
    REQUIRE( actual == expected);
}
