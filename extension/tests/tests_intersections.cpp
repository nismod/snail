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

static double linestring_length(const linestr &segment) {
  double total = 0.0;
  for (std::size_t i = 1; i < segment.size(); ++i) {
    snail::geometry::Line piece(segment[i - 1], segment[i]);
    total += piece.length();
  }
  return total;
}

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
  case3.expected_splits = {{{1.0, 0.5}, {1.5, 1.0}}, {{1.5, 1.0}, {1.5, 2.0}}};

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
  case4.expected_splits = {{{0.5, 0.5}, {1.0, 1.0}}, {{1.0, 1.0}, {1.5, 1.5}}};

  // vertical along gridline
  Config case5;
  case5.linestring = {{0.5, 1.0}, {2.5, 1.0}};
  case5.expected_splits = {{{0.5, 1.0}, {1.0, 1.0}},
                           {{1.0, 1.0}, {2.0, 1.0}},
                           {{2.0, 1.0}, {2.5, 1.0}}};

  // horizontal along gridline
  Config case6;
  case6.linestring = {{0.0, 1.1}, {0.0, 4.7}};
  case6.expected_splits = {{{0.0, 1.1}, {0.0, 2.0}},
                           {{0.0, 2.0}, {0.0, 3.0}},
                           {{0.0, 3.0}, {0.0, 4.0}},
                           {{0.0, 4.0}, {0.0, 4.7}}};

  // Includes single point on gridline (was duplicating)
  Config case7;
  case7.linestring = {{2.9, 2.2}, {2., 1.5}, {0.9, 1.5}};
  case7.expected_splits = {{{2.9, 2.2}, {2.64286, 2}},
                           {{2.64286, 2.}, {2., 1.5}},
                           {{2., 1.5}, {1., 1.5}},
                           {{1., 1.5}, {0.9, 1.5}}};

  // V shape with floating point error
  Config case8;
  case8.linestring = {{0.5, 1.1}, {1.5, 0.9}, {2.5, 1.1}};
  case8.expected_splits = {{{0.5, 1.1}, {1., 1.}},
                           {{1., 1.}, {1.5, 0.9}, {2., 1.}},
                           {{2., 1.}, {2.5, 1.1}}};

  // Linestring points are marked by o:
  // Intersection points are marked by (o):
  // +---------------+--------------+
  // |               |              |
  // |               |              |
  // |               |              |
  // |               |              |
  // |               |              |
  // |               |              |
  // |               |              |
  // +---------------+--------------+
  // |               |              |
  // |               |              |
  // |               |              |
  // |      (o)      |              |
  // |     /         |              |
  // |  ---          |              |
  // | /             |              |
  //(o)--------------+--------------+
  // (0,0)         (1,0)          (2,0)
  Config case9;
  case9.linestring = {{0, 0}, {0.5, 0.5}};
  case9.expected_splits = {{{0, 0}, {0.5, 0.5}}};

  // TODO case7, case8
  auto test_data = GENERATE_COPY(case1, case2, case3, case4, case5, case6,
                                 // case7,
                                 // case8,
                                 case9);

  std::vector<linestr> expected_splits = test_data.expected_splits;

  linestr geom = test_data.linestring;
  snail::geometry::LineString line(geom);

  // Using default Affine transform(1, 0, 0, 0, 1, 0)
  snail::grid::Grid test_raster(2, 2, snail::transform::Affine());
  std::vector<linestr> splits =
      snail::operations::findIntersectionsLineString(line, test_raster);

  // DEBUG
  /*
  std::cout.precision(18);
  for (int i = 0; i < splits.size(); i++) {
    std::cout << "Split" << i << "\n";
    for (int j = 0; j < splits[i].size(); j++) {
      snail::geometry::Coord point = splits[i][j];
      std::cout << "  " << point.x << "," << point.y << "\n";
    }
  }
  */

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

TEST_CASE("LineString corner produces zero-length split", "[decomposition]") {
  snail::grid::Grid test_raster(2, 2, snail::transform::Affine());
  linestr coordinates = {{0.0, 0.0}, {1.0, 1.0}, {2.0, 0.0}};
  snail::geometry::LineString line(coordinates);

  auto splits =
      snail::operations::findIntersectionsLineString(line, test_raster);

  bool zero_length_found = false;
  for (const auto &segment : splits) {
    double length = linestring_length(segment);
    if (length < 1e-9) {
      zero_length_found = true;
      break;
    }
  }

  REQUIRE(!zero_length_found);
}

TEST_CASE("LineString outside grid remains unchanged", "[decomposition]") {
  snail::grid::Grid test_raster(2, 2, snail::transform::Affine());
  linestr coordinates = {{3.0, 0.5}, {6.0, 0.7}};
  snail::geometry::LineString line(coordinates);

  auto splits =
      snail::operations::findIntersectionsLineString(line, test_raster, true);

  REQUIRE(splits.size() == 1);
  REQUIRE(splits[0].size() == coordinates.size());
  for (std::size_t i = 0; i < coordinates.size(); ++i) {
    REQUIRE(std::abs(splits[0][i].x - coordinates[i].x) < TOL);
    REQUIRE(std::abs(splits[0][i].y - coordinates[i].y) < TOL);
  }
}

TEST_CASE("LineString partially overlapping grid splits correctly",
          "[decomposition]") {
  snail::grid::Grid test_raster(2, 2, snail::transform::Affine());
  linestr coordinates = {{-5.0, 0.5}, {1.5, 0.5}};
  snail::geometry::LineString line(coordinates);

  std::vector<linestr> expected = {
      {{-5.0, 0.5}, {0.0, 0.5}},
      {{0.0, 0.5}, {1.0, 0.5}},
      {{1.0, 0.5}, {1.5, 0.5}},
  };

  auto splits =
      snail::operations::findIntersectionsLineString(line, test_raster, true);

  REQUIRE(splits.size() == expected.size());
  for (std::size_t i = 0; i < expected.size(); ++i) {
    REQUIRE(splits[i].size() == expected[i].size());
    for (std::size_t j = 0; j < expected[i].size(); ++j) {
      REQUIRE(std::abs(splits[i][j].x - expected[i][j].x) < TOL);
      REQUIRE(std::abs(splits[i][j].y - expected[i][j].y) < TOL);
    }
  }
}

TEST_CASE("Bounded intersection leaves non-intersecting lines unsplit",
          "[bounds]") {
  snail::grid::Grid test_raster(2, 2, snail::transform::Affine());

  SECTION("Overlaps x and y range with near-touch at {2.,0.}") {
    snail::geometry::LineString line({{1., -1.}, {3.0000000001, 1.}});
    auto splits = snail::operations::findIntersectionsLineString(
        line, test_raster, true);

    REQUIRE(splits.size() == 1);
    REQUIRE(splits[0][0] == snail::geometry::Coord(1., -1.));
    REQUIRE(splits[0][1] == snail::geometry::Coord(3.0000000001, 1.));
  }

  SECTION("Overlaps x and y range without intersecting") {
    snail::geometry::LineString line({{1., -2.}, {2.5, -0.5}, {4., 1.}});
    auto splits = snail::operations::findIntersectionsLineString(
        line, test_raster, true);

    REQUIRE(splits.size() == 1);
    REQUIRE(splits[0][0] == snail::geometry::Coord(1., -2.));
    REQUIRE(splits[0][2] == snail::geometry::Coord(4., 1.));
  }
}

TEST_CASE("Bounded intersection handles reversed segment endpoints",
          "[bounds]") {
  snail::grid::Grid test_raster(2, 2, snail::transform::Affine());

  SECTION("reversed x direction") {
    snail::geometry::LineString line({{1.5, 0.5}, {-5.0, 0.5}});
    auto splits = snail::operations::findIntersectionsLineString(
        line, test_raster, true);

    REQUIRE(splits.size() == 3);
    REQUIRE(splits[0][1] == snail::geometry::Coord(1.0, 0.5));
    REQUIRE(splits[1][1] == snail::geometry::Coord(0.0, 0.5));
  }

  SECTION("reversed y direction") {
    snail::geometry::LineString line({{0.5, 1.5}, {0.5, -5.0}});
    auto splits = snail::operations::findIntersectionsLineString(
        line, test_raster, true);

    REQUIRE(splits.size() == 3);
    REQUIRE(splits[0][1] == snail::geometry::Coord(0.5, 1.0));
    REQUIRE(splits[1][1] == snail::geometry::Coord(0.5, 0.0));
  }
}

TEST_CASE("Bounded intersection handles segments parallel to grid bounds",
          "[bounds]") {
  snail::grid::Grid test_raster(2, 2, snail::transform::Affine());

  SECTION("vertical segment on a grid boundary") {
    snail::geometry::LineString line({{0.0, -1.0}, {0.0, 5.0}});
    auto splits = snail::operations::findIntersectionsLineString(
        line, test_raster, true);

    REQUIRE(splits == std::vector<linestr>{
      {{0.0, -1.0}, {0.0, 0.0}},
      {{0.0, 0.0}, {0.0, 1.0}},
      {{0.0, 1.0}, {0.0, 2.0}},
      {{0.0, 2.0}, {0.0, 5.0}},
    });
  }

  SECTION("horizontal segment on a grid boundary") {
    snail::geometry::LineString line({{-1.0, 0.0}, {5.0, 0.0}});
    auto splits = snail::operations::findIntersectionsLineString(
        line, test_raster, true);

    REQUIRE(splits == std::vector<linestr>{
      {{-1.0, 0.0}, {0.0, 0.0}},
      {{0.0, 0.0}, {1.0, 0.0}},
      {{1.0, 0.0}, {2.0, 0.0}},
      {{2.0, 0.0}, {5.0, 0.0}},
    });
  }
}

TEST_CASE("Bounded intersection leaves disjoint parallel segments unchanged",
          "[bounds]") {
  snail::grid::Grid test_raster(2, 2, snail::transform::Affine());

  SECTION("vertical segment outside") {
    snail::geometry::LineString line({{3.0, -1.0}, {3.0, 3.0}});
    auto splits = snail::operations::findIntersectionsLineString(
        line, test_raster, true);
    REQUIRE(splits == std::vector<linestr>{{{3.0, -1.0}, {3.0, 3.0}}});
  }

  SECTION("horizontal segment outside") {
    snail::geometry::LineString line({{-1.0, 3.0}, {3.0, 3.0}});
    auto splits = snail::operations::findIntersectionsLineString(
        line, test_raster, true);
    REQUIRE(splits == std::vector<linestr>{{{-1.0, 3.0}, {3.0, 3.0}}});
  }
}

TEST_CASE("LineString crossing whole grid splits when bounded, any direction",
          "[decomposition]") {
  // Regression test: segments crossing the grid with both endpoints outside
  // must be detected (and split) whatever their orientation, including
  // right-to-left and top-to-bottom.
  Config right_to_left;
  right_to_left.linestring = {{5.0, 0.5}, {-5.0, 0.5}};
  right_to_left.expected_splits = {
      {{5.0, 0.5}, {2.0, 0.5}},
      {{2.0, 0.5}, {1.0, 0.5}},
      {{1.0, 0.5}, {0.0, 0.5}},
      {{0.0, 0.5}, {-5.0, 0.5}},
  };

  Config top_to_bottom;
  top_to_bottom.linestring = {{0.5, 5.0}, {0.5, -5.0}};
  top_to_bottom.expected_splits = {
      {{0.5, 5.0}, {0.5, 2.0}},
      {{0.5, 2.0}, {0.5, 1.0}},
      {{0.5, 1.0}, {0.5, 0.0}},
      {{0.5, 0.0}, {0.5, -5.0}},
  };

  auto test_data = GENERATE_COPY(right_to_left, top_to_bottom);
  std::vector<linestr> expected = test_data.expected_splits;

  snail::grid::Grid test_raster(2, 2, snail::transform::Affine());
  snail::geometry::LineString line(test_data.linestring);

  auto splits =
      snail::operations::findIntersectionsLineString(line, test_raster, true);

  REQUIRE(splits.size() == expected.size());
  for (std::size_t i = 0; i < expected.size(); ++i) {
    REQUIRE(splits[i].size() == expected[i].size());
    for (std::size_t j = 0; j < expected[i].size(); ++j) {
      REQUIRE(std::abs(splits[i][j].x - expected[i][j].x) < TOL);
      REQUIRE(std::abs(splits[i][j].y - expected[i][j].y) < TOL);
    }
  }
}

TEST_CASE("Segment collinear with grid edge splits within bounds when bounded",
          "[decomposition]") {
  // Regression test: a vertical segment collinear with a grid edge used to
  // trigger a 0/0 division in the bounded-intersection test. A segment on
  // the boundary counts as inside (consistent with pointInBounds), so it is
  // split at the gridlines within the grid extent and left in one piece
  // beyond it.
  Config left_edge;
  left_edge.linestring = {{0.0, -1.0}, {0.0, 3.0}};
  left_edge.expected_splits = {
      {{0.0, -1.0}, {0.0, 0.0}},
      {{0.0, 0.0}, {0.0, 1.0}},
      {{0.0, 1.0}, {0.0, 2.0}},
      {{0.0, 2.0}, {0.0, 3.0}},
  };

  Config right_edge;
  right_edge.linestring = {{2.0, -1.0}, {2.0, 3.0}};
  right_edge.expected_splits = {
      {{2.0, -1.0}, {2.0, 0.0}},
      {{2.0, 0.0}, {2.0, 1.0}},
      {{2.0, 1.0}, {2.0, 2.0}},
      {{2.0, 2.0}, {2.0, 3.0}},
  };

  auto test_data = GENERATE_COPY(left_edge, right_edge);
  std::vector<linestr> expected = test_data.expected_splits;

  snail::grid::Grid test_raster(2, 2, snail::transform::Affine());
  snail::geometry::LineString line(test_data.linestring);

  auto splits =
      snail::operations::findIntersectionsLineString(line, test_raster, true);

  REQUIRE(splits.size() == expected.size());
  for (std::size_t i = 0; i < expected.size(); ++i) {
    REQUIRE(splits[i].size() == expected[i].size());
    for (std::size_t j = 0; j < expected[i].size(); ++j) {
      REQUIRE(std::abs(splits[i][j].x - expected[i][j].x) < TOL);
      REQUIRE(std::abs(splits[i][j].y - expected[i][j].y) < TOL);
    }
  }
}

TEST_CASE("Segment collinear with grid edge outside extent remains unchanged",
          "[decomposition]") {
  // collinear with the left grid edge (x = 0) but wholly beyond the top of
  // the grid: no part of the segment is within the grid extent, so with
  // bounded=true it is returned unchanged.
  snail::grid::Grid test_raster(2, 2, snail::transform::Affine());
  linestr coordinates = {{0.0, 3.0}, {0.0, 5.0}};
  snail::geometry::LineString line(coordinates);

  auto splits =
      snail::operations::findIntersectionsLineString(line, test_raster, true);

  REQUIRE(splits.size() == 1);
  REQUIRE(splits[0].size() == coordinates.size());
  for (std::size_t i = 0; i < coordinates.size(); ++i) {
    REQUIRE(std::abs(splits[0][i].x - coordinates[i].x) < TOL);
    REQUIRE(std::abs(splits[0][i].y - coordinates[i].y) < TOL);
  }
}

TEST_CASE("Segment collinear with interior gridline splits at crossings",
          "[decomposition]") {
  // A segment lying along an interior gridline (y = 1) is split at the
  // vertical gridline it crosses (x = 1); bounded and unbounded behaviour
  // agree because the segment is entirely inside the grid.
  snail::grid::Grid test_raster(2, 2, snail::transform::Affine());
  linestr coordinates = {{0.5, 1.0}, {1.5, 1.0}};
  snail::geometry::LineString line(coordinates);

  std::vector<linestr> expected = {
      {{0.5, 1.0}, {1.0, 1.0}},
      {{1.0, 1.0}, {1.5, 1.0}},
  };

  bool bounded = GENERATE(true, false);
  auto splits = snail::operations::findIntersectionsLineString(
      line, test_raster, bounded);

  REQUIRE(splits.size() == expected.size());
  for (std::size_t i = 0; i < expected.size(); ++i) {
    REQUIRE(splits[i].size() == expected[i].size());
    for (std::size_t j = 0; j < expected[i].size(); ++j) {
      REQUIRE(std::abs(splits[i][j].x - expected[i][j].x) < TOL);
      REQUIRE(std::abs(splits[i][j].y - expected[i][j].y) < TOL);
    }
  }
}

TEST_CASE("Diagonal segment inside grid is split when bounded",
          "[decomposition]") {
  // A segment entirely inside the grid must still be split when bounded=true
  // (both endpoints inside means the segment intersects the grid extent).
  snail::grid::Grid test_raster(2, 2, snail::transform::Affine());
  linestr coordinates = {{0.3, 0.4}, {1.8, 1.6}};
  snail::geometry::LineString line(coordinates);

  std::vector<linestr> expected = {
      {{0.3, 0.4}, {1.0, 0.96}},
      {{1.0, 0.96}, {1.05, 1.0}},
      {{1.05, 1.0}, {1.8, 1.6}},
  };

  auto splits =
      snail::operations::findIntersectionsLineString(line, test_raster, true);

  REQUIRE(splits.size() == expected.size());
  for (std::size_t i = 0; i < expected.size(); ++i) {
    REQUIRE(splits[i].size() == expected[i].size());
    for (std::size_t j = 0; j < expected[i].size(); ++j) {
      REQUIRE(std::abs(splits[i][j].x - expected[i][j].x) < TOL);
      REQUIRE(std::abs(splits[i][j].y - expected[i][j].y) < TOL);
    }
  }
}

TEST_CASE("Bounded splits keep interior vertices within grid extents",
          "[decomposition]") {
  snail::grid::Grid test_raster(2, 2, snail::transform::Affine());
  linestr coordinates = {{-2.0, 0.5}, {1.5, 0.5}};
  snail::geometry::LineString line(coordinates);

  auto splits =
      snail::operations::findIntersectionsLineString(line, test_raster, true);

  REQUIRE(splits.size() == 3);
  for (std::size_t segment_idx = 0; segment_idx < splits.size();
       ++segment_idx) {
    const auto &segment = splits[segment_idx];
    REQUIRE(segment.size() >= 2);
    for (std::size_t point_idx = 0; point_idx < segment.size(); ++point_idx) {
      bool is_start = (segment_idx == 0 && point_idx == 0);
      bool is_end =
          (segment_idx == splits.size() - 1 && point_idx == segment.size() - 1);
      if (is_start || is_end) {
        continue;
      }
      REQUIRE(
          snail::operations::pointInBounds(segment[point_idx], test_raster));
    }
  }
}

TEST_CASE("pointInBounds treats boundary as inside", "[bounds]") {
  snail::grid::Grid test_raster(2, 2, snail::transform::Affine());
  snail::geometry::Coord on_edge_x(2.0, 1.0);
  REQUIRE(snail::operations::pointInBounds(on_edge_x, test_raster));
  snail::geometry::Coord on_edge_y(1.0, 2.0);
  REQUIRE(snail::operations::pointInBounds(on_edge_y, test_raster));
  snail::geometry::Coord inner(0.9, 1.1);
  REQUIRE(snail::operations::pointInBounds(inner, test_raster));
  snail::geometry::Coord outer(0.9, 11.1);
  REQUIRE(!snail::operations::pointInBounds(outer, test_raster));
}

TEST_CASE("Split with different grid", "[decomposition]") {
  Config case1;
  case1.expected_splits = {
      {{191483.13281982497, 2044523.5152593486}, {191487.679611496, 2044520.0}},
      {{191487.679611496, 2044520.0}, {191565.0, 2044460.22131688}},
      {{191565.0, 2044460.22131688}, {191604.089585794, 2044430.0}},
      {{191604.089585794, 2044430.0},
       {191618.28009818937, 2044419.0288944514}}};
  case1.linestring = {{191483.13281982497, 2044523.5152593486},
                      {191618.28009818937, 2044419.0288944514}};

  Config case2;
  case2.expected_splits = {
      {{190040.9085973615, 2043440.0}, {190035.0, 2043440.0}},
      {{190035.0, 2043440.0}, {189945.0, 2043440.0}},
      {{189945.0, 2043440.0}, {189855.0, 2043440.0}},
      {{189855.0, 2043440.0}, {189819.85637632824, 2043440.0}}};
  case2.linestring = {{190040.9085973615, 2043440.0},
                      {189819.85637632824, 2043440.0}};

  auto test_data = GENERATE_COPY(case1, case2);

  std::vector<linestr> expected_splits = test_data.expected_splits;
  linestr geom = test_data.linestring;
  snail::geometry::LineString line(geom);

  snail::grid::Grid test_raster(
      2, 2,
      snail::transform::Affine(90.0, 0.0, 132165.0, 0.0, -90.0, 2055230.0));
  std::vector<linestr> splits =
      snail::operations::findIntersectionsLineString(line, test_raster);

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
  snail::operations::Direction direction =
      snail::operations::Direction::horizontal;
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
  case1.exterior_crossings = {{0.5, 0.5}, {1., 0.5}, {1.5, 1.}, {1., 1.}};
  case1.expected_splits = {
      {{1., 1.}, {1.5, 1.}},
  };
  case1.direction = snail::operations::Direction::horizontal;

  SplitGridConfig case2;
  case2.exterior_crossings = {{0.5, 0.5}, {1., 0.5}, {1.5, 1.}, {1., 1.}};
  case2.expected_splits = {
      {{1., 0.5}, {1., 1.}},
  };
  case2.direction = snail::operations::Direction::vertical;

  // Concave shape
  // +------+------+------+
  // |      |      |      |
  // | o--o |      | o--o |
  // | |..| |      | |..| |
  // +(o==o)+------+(o==o)+
  // | |..| |      | |..| |
  // | |..o(o)----(o)o..| |
  // | |....‖......‖....| |
  // +(o)===+======+===(o)+
  // | |....‖......‖....| |
  // | |....‖......‖....| |
  // | o---(o)----(o)---o |
  // +------+------+------+
  // 0      1      2      3
  SplitGridConfig case3;
  case3.exterior_crossings = {
      {0.1, 0.1}, {1., 0.1},  {2., 0.1},  {2.9, 0.1}, // bottom edge
      {2.9, 1.},  {2.9, 2.},  {2.9, 2.2},             // right edge
      {2.1, 2.2}, {2.1, 2.},  {2.1, 1.5},             // inside right
      {2., 1.5},  {1., 1.5},  {0.9, 1.5},             // inside top
      {0.9, 2.},  {0.9, 2.2},                         // inside left
      {0.1, 2.2}, {0.1, 2.},  {0.1, 1.},  {0.1, 0.1}  // left edge
  };
  case3.expected_splits = {// full width interior
                           {{0.1, 1.}, {1., 1.}},
                           {{1., 1.}, {2., 1.}},
                           {{2., 1.}, {2.9, 1.}},
                           // left tower
                           {{0.1, 2.}, {0.9, 2.}},
                           // right tower
                           {{2.1, 2.}, {2.9, 2.}}};
  case3.direction = snail::operations::Direction::horizontal;

  // Kite shape
  SplitGridConfig case4;
  case4.exterior_crossings = {{0.5, 1.25}, {1., 1.},   {1.5, 0.75}, {2., 1.},
                              {2.5, 1.25}, {2.25, 1.}, {2., 0.75},  {1.5, 0.25},
                              {1., 0.75},  {0.75, 1.}};
  case4.expected_splits = {{{0.75, 1.0}, {1.0, 1.0}},
                           {{2.0, 1.0}, {2.25, 1.0}}};
  case4.direction = snail::operations::Direction::horizontal;

  auto test_data = GENERATE_COPY(case1, case2, case3, case4);

  std::vector<linestr> expected_splits = test_data.expected_splits;

  // Using default Affine transform(1, 0, 0, 0, 1, 0)
  snail::grid::Grid grid(2, 2, snail::transform::Affine());

  std::vector<linestr> splits = snail::operations::splitAlongGridlines(
      test_data.exterior_crossings, test_data.min_level, test_data.max_level,
      test_data.direction, grid);
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

TEST_CASE("Exterior ring to gridlines with fractional grid",
          "[decomposition]") {
  // Using Affine transform with fractional cell size
  snail::grid::Grid grid(
      2, 2, snail::transform::Affine(0.5, 0.0, 0.0, 0.0, 0.5, 0.0));

  std::vector<linestr> splits = snail::operations::splitAlongGridlines(
      {{.3, .3},
       {.3, .5},
       {.3, .8},
       {.5, .8},
       {.8, .8},
       {.8, .5},
       {.8, .3},
       {.5, .3}},
      0, 2, snail::operations::Direction::horizontal, grid);
  std::vector<linestr> expected_splits = {{
      {{.3, .5}, {.5, .5}},
      {{.5, .5}, {.8, .5}},
  }};
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
