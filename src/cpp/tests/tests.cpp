// This tells Catch to provide a main()
// only do this in one cpp file
#define CATCH_CONFIG_MAIN
#define TOL 0.001

#include <ostream>
#include <string>
#include <sstream>
#include <iostream>
#include <fstream>
#include <catch2/catch.hpp>

#include "../features.h"
#include "../raster.h"
#include "../geom.h"

using linestr = std::vector<geometry::Vec2<double>>;

std::vector<linestr> findIntersectionsLineString(Feature, Ascii);

TEST_CASE( "LineStrings are decomposed", "[decomposition]") {
  // Linestring points are marked by o:
  // Intersection points are marked by (o):
  // +---------------+--------------+
  // |               |              |
  // |               |              |
  // |               |              |
  // |               |       o      |
  // |               |       |      |
  // |               |       |      |
  // |               |       |      |
  // +---------------+------(o)-----+
  // |               |       |      |
  // |               |       |      |
  // |               |       |      |
  // |       o---o--(o)------o      |
  // |               |              |
  // |               |              |
  // |               |              |
  // +---------------+--------------+
  // (0,0)         (1,0)          (1,0)
  
  geometry::Vec2<double> point1(0.5, 0.5);
  geometry::Vec2<double> point2(0.75, 0.5);
  geometry::Vec2<double> point3(1.5, 0.5);
  geometry::Vec2<double> point4(1.5, 1.5);
  
  geometry::Vec2<double> inter1(1., 0.5);
  geometry::Vec2<double> inter2(1.5, 1.);

  std::vector<linestr> expected_splits = {
    {point1, point2, inter1},
    {inter1, point3, inter2},
    {inter2, point4}
  };
  
  Feature f;
  linestr geom = {point1, point2, point3, point4};
  f.geometry.insert(f.geometry.begin(), geom.begin(), geom.end());

  Ascii test_raster("./test_data/fake_raster.asc");
  std::vector<linestr> splits = findIntersectionsLineString(f, test_raster);

  // Test that we're getting the expected number of splits
  REQUIRE(splits.size() == expected_splits.size());
  // Test that each one of the splits have the expected size
  for (int i=0; i<splits.size();i++) {
    REQUIRE(splits[i].size() == expected_splits[i].size());
  }
  // Test that each one of the splits are made of the expected points
  for (int i=0; i<splits.size();i++) {
    for (int j=0;j<splits[i].size();j++) {
      geometry::Vec2<double> point = splits[i][j];
      geometry::Vec2<double> expected_point = expected_splits[i][j];

      REQUIRE(std::abs(point.x - expected_point.x) < TOL);
      REQUIRE(std::abs(point.y - expected_point.y) < TOL);
    }
  }
}

TEST_CASE( "LineStrings are decomposed II", "[decomposition]") {
  // Linestring points are marked by o:
  // Intersection points are marked by (o):
  // +---------------+--------------+
  // |               |              |
  // |               |              |
  // |               |              |
  // |               |       o      |
  // |               |      /       |
  // |               |    /-        |
  // |               |  /-          |
  // +---------------+(o)-----------+
  // |              (o)             |
  // |             /-|              |
  // |            /  |              |
  // |       o---o   |              |
  // |               |              |
  // |               |              |
  // |               |              |
  // +---------------+--------------+
  // (0,0)         (1,0)          (1,0)
  
  geometry::Vec2<double> point1(0.5, 0.5);
  geometry::Vec2<double> point2(0.75, 0.5);
  geometry::Vec2<double> point3(1.5, 1.5);
  
  geometry::Vec2<double> inter1(1., 0.8333);
  geometry::Vec2<double> inter2(1.125, 1.);

  std::vector<linestr> expected_splits = {
    {point1, point2, inter1},
    {inter1, inter2},
    {inter2, point3}
  };
  
  Feature f;
  linestr geom = {point1, point2, point3};
  f.geometry.insert(f.geometry.begin(), geom.begin(), geom.end());

  Ascii test_raster("./test_data/fake_raster.asc");
  std::vector<linestr> splits = findIntersectionsLineString(f, test_raster);

  // Test that we're getting the expected number of splits
  REQUIRE(splits.size() == expected_splits.size());
  // Test that each one of the splits have the expected size
  for (int i=0; i<splits.size();i++) {
    REQUIRE(splits[i].size() == expected_splits[i].size());
  }
  // Test that each one of the splits are made of the expected points
  for (int i=0; i<splits.size();i++) {
    for (int j=0;j<splits[i].size();j++) {
      geometry::Vec2<double> point = splits[i][j];
      geometry::Vec2<double> expected_point = expected_splits[i][j];

      REQUIRE(std::abs(point.x - expected_point.x) < TOL);
      REQUIRE(std::abs(point.y - expected_point.y) < TOL);
    }
  }
}

