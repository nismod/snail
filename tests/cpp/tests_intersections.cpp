#include <catch2/catch.hpp>
#include <fstream>
#include <iostream>
#include <ostream>
#include <sstream>
#include <string>
#include <tuple>
#include <vector>

#include "geometry.hpp"
#include "grid.hpp"
#include "transform.hpp"
#include "operations.hpp"

#define TOL 0.001

using linestr = std::vector<snail::geometry::Coord>;

struct Config {
  linestr linestring;
  std::vector<linestr> expected_splits;
};

TEST_CASE("LineStrings are decomposed", "[decomposition]") {

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
  // (0,0)         (1,0)          (2,0)
  Config case1;
  case1.linestring = {{0.5, 0.5}, {0.75, 0.5}, {1.5, 0.5}, {1.5, 1.5}};
  case1.expected_splits = {{{0.5, 0.5}, {0.75, 0.5}, {1., 0.5}},
			   {{1., 0.5}, {1.5, 0.5}, {1.5, 1.}},
			   {{1.5, 1.}, {1.5, 1.5}}};

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
  // (0,0)         (1,0)          (2,0)
  Config case2;
  case2.linestring = {{0.5, 0.5}, {0.75, 0.5}, {1.5, 1.5}};
  case2.expected_splits = {{{0.5, 0.5}, {0.75, 0.5}, {1., 0.8333}},
			   {{1., 0.8333}, {1.125, 1.}},
			   {{1.125, 1.}, {1.5, 1.5}}};

  // Linestring points are marked by o:
  // Intersection points are marked by (o):
  // +---------------+------(o)-----+
  // |               |       |      |
  // |               |       |      |
  // |               |       |      |
  // |               |       |      |
  // |               |       |      |
  // |               |       |      |
  // |               |       |      |
  // +---------------+------(o)-----+
  // |               |     /        |
  // |               |    /         |
  // |               |   /          |
  // |              (o)-/           |
  // |               |              |
  // |               |              |
  // |               |              |
  // +---------------+--------------+
  // (0,0)         (1,0)          (2,0)
  Config case3;
  case3.linestring = {{1.0, 0.5}, {1.5, 1.0}, {1.5, 2.0}};
  case3.expected_splits = {{{1.0, 0.5}, {1.5, 1.0}},
			   {{1.5, 1.0}, {1.5, 2.0}}};

  // Linestring points are marked by o:
  // Intersection points are marked by (o):
  // +---------------+--------------+
  // |               |              |
  // |               |              |
  // |               |              |
  // |               |       o      |
  // |               |              |
  // |               |              |
  // |               |              |
  // +--------------(o)--------------+
  // |               |              |
  // |               |              |
  // |               |              |
  // |       o       |              |
  // |               |              |
  // |               |              |
  // |               |              |
  // +---------------+--------------+
  // (0,0)         (1,0)          (2,0)
  Config case4;
  case4.linestring = {{0.5, 0.5}, {1.5, 1.5}};
  case4.expected_splits = {{{0.5, 0.5}, {1.0, 1.0}},
			   {{1.0, 1.0}, {1.5, 1.5}}};

  auto test_data = GENERATE_COPY(case1, case2, case3, case4);

  std::vector<linestr> expected_splits = test_data.expected_splits;

  linestr geom = test_data.linestring;
  snail::geometry::LineString line(geom);

  // Using default Affine transform(1, 0, 0, 0, 1, 0)
  snail::grid::Grid test_raster(2, 2, snail::transform::Affine());
  std::vector<linestr> splits = snail::operations::findIntersectionsLineString(line, test_raster);

  // Test that we're getting the expected number of splits
  REQUIRE(splits.size() == expected_splits.size());
  // Test that each one of the splits have the expected size
  for (int i = 0; i < splits.size(); i++) {
    REQUIRE(splits[i].size() == expected_splits[i].size());
  }
  // Test that each one of the splits are made of the expected points
  for (int i = 0; i < splits.size(); i++) {
    for (int j = 0; j < splits[i].size(); j++) {
      snail::geometry::Coord point = splits[i][j];
      snail::geometry::Coord expected_point = expected_splits[i][j];

      REQUIRE(std::abs(point.x - expected_point.x) < TOL);
      REQUIRE(std::abs(point.y - expected_point.y) < TOL);
    }
  }
}


TEST_CASE("Split with different grid", "[decomposition]") {

  std::vector<linestr> expected_splits = {
    {{191483.13281982497, 2044523.5152593486}, {191487.679611496, 2044520.0}},
    {{191487.679611496, 2044520.0}, {191565.0, 2044460.22131688}},
    {{191565.0, 2044460.22131688}, {191604.089585794, 2044430.0}},
    {{191604.089585794, 2044430.0}, {191618.28009818937, 2044419.0288944514}}
  };

  linestr geom = {
    {191483.13281982497, 2044523.5152593486},
    {191618.28009818937, 2044419.0288944514}
  };
  snail::geometry::LineString line(geom);

  snail::grid::Grid test_raster(2, 2, snail::transform::Affine(
    90.0, 0.0, 132165.0, 0.0, -90.0, 2055230.0
  ));
  std::vector<linestr> splits = snail::operations::findIntersectionsLineString(line, test_raster);

  // Test that we're getting the expected number of splits
  REQUIRE(splits.size() == expected_splits.size());
  // Test that each one of the splits have the expected size
  for (int i = 0; i < splits.size(); i++) {
    REQUIRE(splits[i].size() == expected_splits[i].size());
  }
  // Test that each one of the splits are made of the expected points
  for (int i = 0; i < splits.size(); i++) {
    for (int j = 0; j < splits[i].size(); j++) {
      snail::geometry::Coord point = splits[i][j];
      snail::geometry::Coord expected_point = expected_splits[i][j];

      REQUIRE(std::abs(point.x - expected_point.x) < TOL);
      REQUIRE(std::abs(point.y - expected_point.y) < TOL);
    }
  }
}

struct SplitGridConfig {
  linestr exterior_crossings;
  std::vector<linestr> expected_splits;
  int min_level = 0;
  int max_level = 2;
  snail::operations::Direction direction = snail::operations::Direction::horizontal;
};


TEST_CASE("Exterior ring splits to gridlines", "[decomposition]") {
  // Linestring points are marked by o:
  // Intersection points are marked by (o):
  // Gridline segments are marked by = and ‖
  // +---------------+--------------+
  // |               |              |
  // |               |              |
  // |               |              |
  // |               |       o      |
  // |               |     / |      |
  // |               |   /   |      |
  // |               | /     |      |
  // +--------------(o)=====(o)-----+
  // |             / ‖       |      |
  // |           /   ‖       |      |
  // |         /     ‖       |      |
  // |       o------(o)------o      |
  // |               |              |
  // |               |              |
  // |               |              |
  // +---------------+--------------+
  // (0,0)         (1,0)          (2,0)
  SplitGridConfig case1;
  case1.exterior_crossings = {
    {0.5, 0.5}, {1., 0.5}, {1.5, 1.}, {1., 1.}
  };
  case1.expected_splits = {
    {{1.5, 1.}, {1., 1.}}, // TODO check, would actually expect left-to-right
  };
  SplitGridConfig case2;
  case2.exterior_crossings = {
    {0.5, 0.5}, {1., 0.5}, {1.5, 1.}, {1., 1.}
  };
  case2.expected_splits = {
    {{1., 0.5}, {1., 1.}},
  };
  case2.direction = snail::operations::Direction::vertical;

  auto test_data = GENERATE_COPY(case1, case2);

  std::vector<linestr> expected_splits = test_data.expected_splits;

  // Using default Affine transform(1, 0, 0, 0, 1, 0)
  snail::grid::Grid grid(2, 2, snail::transform::Affine());

  std::vector<linestr> splits = snail::operations::splitAlongGridlines(
    test_data.exterior_crossings,
    test_data.min_level,
    test_data.max_level,
    test_data.direction,
    grid
  );
  // Test that we're getting the expected number of splits
  REQUIRE(splits.size() == expected_splits.size());
  // Test that each one of the splits have the expected size
  for (int i = 0; i < splits.size(); i++) {
    REQUIRE(splits[i].size() == expected_splits[i].size());
  }
  // Test that each one of the splits are made of the expected points
  for (int i = 0; i < splits.size(); i++) {
    for (int j = 0; j < splits[i].size(); j++) {
      snail::geometry::Coord point = splits[i][j];
      snail::geometry::Coord expected_point = expected_splits[i][j];

      REQUIRE(std::abs(point.x - expected_point.x) < TOL);
      REQUIRE(std::abs(point.y - expected_point.y) < TOL);
    }
  }
}
