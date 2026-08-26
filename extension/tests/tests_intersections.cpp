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
using snail::geometry::LineString;

/// Split a linestring, as a list of pieces, so that the expectations below
/// can be written as plain lists of points
static std::vector<linestr> split_line(const linestr &coordinates,
                                       const snail::grid::Grid &grid, const bool bounded = false) {
  snail::operations::LinePieces pieces =
      snail::operations::splitLineStringGrid(coordinates, grid, bounded);
  std::vector<linestr> out;
  for (std::size_t p = 0; p + 1 < pieces.offsets.size(); p++) {
    out.push_back(linestr(pieces.coordinates.begin() + pieces.offsets[p],
                          pieces.coordinates.begin() + pieces.offsets[p + 1]));
  }
  return out;
}

static std::vector<linestr> split_line(const LineString &line,
                                       const snail::grid::Grid &grid,
                                       const bool bounded = false) {
  return split_line(line.coordinates, grid, bounded);
}

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
  std::vector<linestr> splits = split_line(geom, test_raster);

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
  for (size_t i = 0; i < splits.size(); i++) {
    REQUIRE(splits[i].size() == expected_splits[i].size());
  }
  // Test that each one of the splits are made of the expected points
  for (size_t i = 0; i < splits.size(); i++) {
    for (size_t j = 0; j < splits[i].size(); j++) {
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

  auto splits = split_line(coordinates, test_raster);

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

  auto splits = split_line(line, test_raster, true);

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

  auto splits = split_line(line, test_raster, true);

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
    auto splits = split_line(
        line, test_raster, true);

    REQUIRE(splits.size() == 1);
    REQUIRE(splits[0][0] == snail::geometry::Coord(1., -1.));
    REQUIRE(splits[0][1] == snail::geometry::Coord(3.0000000001, 1.));
  }

  SECTION("Overlaps x and y range without intersecting") {
    snail::geometry::LineString line({{1., -2.}, {2.5, -0.5}, {4., 1.}});
    auto splits = split_line(
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
    auto splits = split_line(
        line, test_raster, true);

    REQUIRE(splits.size() == 3);
    REQUIRE(splits[0][1] == snail::geometry::Coord(1.0, 0.5));
    REQUIRE(splits[1][1] == snail::geometry::Coord(0.0, 0.5));
  }

  SECTION("reversed y direction") {
    snail::geometry::LineString line({{0.5, 1.5}, {0.5, -5.0}});
    auto splits = split_line(
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
    auto splits = split_line(
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
    auto splits = split_line(
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
    auto splits = split_line(
        line, test_raster, true);
    REQUIRE(splits == std::vector<linestr>{{{3.0, -1.0}, {3.0, 3.0}}});
  }

  SECTION("horizontal segment outside") {
    snail::geometry::LineString line({{-1.0, 3.0}, {3.0, 3.0}});
    auto splits = split_line(
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

  auto splits = split_line(line, test_raster, true);

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

  auto splits = split_line(line, test_raster, true);

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

  auto splits = split_line(line, test_raster, true);

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
  auto splits = split_line(line, test_raster, bounded);

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

  auto splits = split_line(line, test_raster, true);

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

  auto splits = split_line(line, test_raster, true);

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
  std::vector<linestr> splits = split_line(geom, test_raster);

  // Test that we're getting the expected number of splits
  REQUIRE(splits.size() == expected_splits.size());
  // Test that each one of the splits have the expected size
  for (size_t i = 0; i < splits.size(); i++) {
    REQUIRE(splits[i].size() == expected_splits[i].size());
  }
  // Test that each one of the splits are made of the expected points
  for (size_t i = 0; i < splits.size(); i++) {
    for (size_t j = 0; j < splits[i].size(); j++) {
      snail::geometry::Coord point = splits[i][j];
      snail::geometry::Coord expected_point = expected_splits[i][j];

      REQUIRE(std::abs(point.x - expected_point.x) < TOL);
      REQUIRE(std::abs(point.y - expected_point.y) < TOL);
    }
  }
}


static double total_area(const std::vector<snail::geometry::Polygon> &pieces) {
  double total = 0.0;
  for (const snail::geometry::Polygon &piece : pieces) {
    total += piece.area();
  }
  return total;
}

static double ring_area2(const linestr &ring) {
  // relative to the first point, to avoid cancellation for small rings far
  // from the origin
  double area2 = 0.0;
  double x0 = ring.front().x;
  double y0 = ring.front().y;
  for (std::size_t i = 0; i + 1 < ring.size(); ++i) {
    area2 += (ring[i].x - x0) * (ring[i + 1].y - y0) -
             (ring[i + 1].x - x0) * (ring[i].y - y0);
  }
  return area2;
}

TEST_CASE("Square is split into cell pieces", "[polygon]") {
  // +---------------+--------------+
  // |               |              |
  // |      o--------+--------o     |
  // |      |........|........|     |
  // |      |........|........|     |
  // +------+--------+--------+-----+
  // |      |........|........|     |
  // |      |........|........|     |
  // |      o--------+--------o     |
  // |               |              |
  // +---------------+--------------+
  // (0,0)         (1,0)          (2,0)
  snail::grid::Grid grid(2, 2, snail::transform::Affine());
  std::vector<linestr> rings = {
      {{0.5, 0.5}, {1.5, 0.5}, {1.5, 1.5}, {0.5, 1.5}, {0.5, 0.5}}};

  auto pieces = snail::operations::splitPolygonGrid(rings, grid);

  REQUIRE(pieces.size() == 4);
  for (const auto &piece : pieces) {
    REQUIRE(std::abs(piece.area() - 0.25) < TOL);
  }
  REQUIRE(std::abs(total_area(pieces) - 1.0) < TOL);
}

TEST_CASE("Concave polygon may split to several pieces per cell", "[polygon]") {
  // U-shape: both arms cross into the cell above, so cell (0,1) holds two
  // disjoint pieces
  snail::grid::Grid grid(2, 2, snail::transform::Affine());
  std::vector<linestr> rings = {{{0.2, 0.2},
                                 {0.8, 0.2},
                                 {0.8, 1.5},
                                 {0.6, 1.5},
                                 {0.6, 0.5},
                                 {0.4, 0.5},
                                 {0.4, 1.5},
                                 {0.2, 1.5},
                                 {0.2, 0.2}}};

  auto pieces = snail::operations::splitPolygonGrid(rings, grid);

  REQUIRE(pieces.size() == 3);
  REQUIRE(std::abs(total_area(pieces) - 0.58) < TOL);
}

TEST_CASE("Concave polygon splits across grid", "[polygon]") {
  // U-shape spanning 3x3 cells, open towards the top
  snail::grid::Grid grid(2, 2, snail::transform::Affine());
  std::vector<linestr> rings = {{{0.1, 0.1},
                                 {2.9, 0.1},
                                 {2.9, 2.2},
                                 {2.1, 2.2},
                                 {2.1, 1.5},
                                 {0.9, 1.5},
                                 {0.9, 2.2},
                                 {0.1, 2.2},
                                 {0.1, 0.1}}};

  auto pieces = snail::operations::splitPolygonGrid(rings, grid);

  REQUIRE(pieces.size() == 8);
  REQUIRE(std::abs(total_area(pieces) - 5.04) < TOL);
}

TEST_CASE("Polygon touching a gridline splits cleanly", "[polygon]") {
  // Triangle with its apex exactly on a horizontal gridline and its base
  // corners exactly on vertical gridlines - all tangent, no crossings
  snail::grid::Grid grid(2, 2, snail::transform::Affine());
  std::vector<linestr> rings = {
      {{1.0, 0.2}, {2.0, 0.2}, {1.5, 1.0}, {1.0, 0.2}}};

  auto pieces = snail::operations::splitPolygonGrid(rings, grid);

  REQUIRE(pieces.size() == 1);
  REQUIRE(std::abs(total_area(pieces) - 0.4) < TOL);
}

TEST_CASE("Polygon boundary along a gridline splits cleanly", "[polygon]") {
  // Hexagon with one boundary section running exactly along the gridline
  // y=1, interior below it - the case that could not be handled by counting
  // boundary vertices on the gridline
  snail::grid::Grid grid(2, 2, snail::transform::Affine());
  std::vector<linestr> rings = {{{0.2, 0.2},
                                 {1.8, 0.2},
                                 {1.8, 1.0},
                                 {1.2, 1.0},
                                 {1.2, 1.8},
                                 {0.2, 1.8},
                                 {0.2, 0.2}}};

  auto pieces = snail::operations::splitPolygonGrid(rings, grid);

  REQUIRE(pieces.size() == 4);
  REQUIRE(std::abs(total_area(pieces) - 2.08) < TOL);
}

TEST_CASE("Polygon with hole in a single cell", "[polygon]") {
  snail::grid::Grid grid(2, 2, snail::transform::Affine());
  std::vector<linestr> rings = {
      // exterior spanning 3x3 cells
      {{0.5, 0.5}, {2.5, 0.5}, {2.5, 2.5}, {0.5, 2.5}, {0.5, 0.5}},
      // hole wholly inside cell (1,1), which is otherwise fully covered
      {{1.25, 1.25}, {1.75, 1.25}, {1.75, 1.75}, {1.25, 1.75}, {1.25, 1.25}}};

  auto pieces = snail::operations::splitPolygonGrid(rings, grid);

  REQUIRE(pieces.size() == 9);
  REQUIRE(std::abs(total_area(pieces) - 3.75) < TOL);
  // exactly one piece has the hole
  int pieces_with_hole = 0;
  for (const auto &piece : pieces) {
    if (piece.interiors.size() == 1) {
      pieces_with_hole++;
      REQUIRE(std::abs(piece.area() - 0.75) < TOL);
    }
  }
  REQUIRE(pieces_with_hole == 1);
}

TEST_CASE("Polygon with hole crossing gridlines", "[polygon]") {
  snail::grid::Grid grid(2, 2, snail::transform::Affine());
  std::vector<linestr> rings = {
      // exterior spanning 3x3 cells
      {{0.5, 0.5}, {2.5, 0.5}, {2.5, 2.5}, {0.5, 2.5}, {0.5, 0.5}},
      // hole crossing cell borders
      {{0.9, 0.9}, {2.1, 0.9}, {2.1, 1.1}, {0.9, 1.1}, {0.9, 0.9}}};

  auto pieces = snail::operations::splitPolygonGrid(rings, grid);

  REQUIRE(pieces.size() == 9);
  REQUIRE(std::abs(total_area(pieces) - 3.76) < TOL);
}

TEST_CASE("Issue 45 polygons with vertices near gridlines", "[polygon]") {
  // Building polygons that used to fail with "Expected even number of
  // crossings on gridline": vertices sit within ~1e-14 of a gridline
  snail::grid::Grid grid(
      698, 252,
      snail::transform::Affine(0.0031, 0.0, -78.34655, 0.0, -0.0031, 18.52365));

  linestr building1 = {{-77.280182457, 17.97282044099908},
                       {-77.28015690000001, 17.97282209999907},
                       {-77.28015000000001, 17.9727272999991},
                       {-77.28013888, 17.97272804099908},
                       {-77.28013888, 17.97270079499907},
                       {-77.28014880000001, 17.97270009999908},
                       {-77.2801487, 17.9726981999991},
                       {-77.28018086100001, 17.97269609699906},
                       {-77.280182457, 17.97282044099908}};

  linestr building2 = {{-78.02110589999999, 18.45206629999907},
                       {-78.02105, 18.45206859999908},
                       {-78.0210484, 18.45203429999912},
                       {-78.0211044, 18.45203199999911},
                       {-78.02110589999999, 18.45206629999907}};

  linestr building3 = {{-77.34835, 18.45626119999909},
                       {-77.348350719, 18.45625000599908},
                       {-77.34840013100001, 18.45625000599913},
                       {-77.3483981, 18.4562824999991},
                       {-77.3483772, 18.45628129999909},
                       {-77.3483755, 18.45630829999907},
                       {-77.34833759999999, 18.45630619999902},
                       {-77.34834050000002, 18.45626069999907},
                       {-77.34835, 18.45626119999909}};

  auto test_ring = GENERATE_COPY(building1, building2, building3);
  auto pieces = snail::operations::splitPolygonGrid({test_ring}, grid);

  double expected_area = std::abs(ring_area2(test_ring)) / 2.0;
  REQUIRE(pieces.size() >= 1);
  REQUIRE(std::abs(total_area(pieces) - expected_area) < 1e-6 * expected_area);
}

/// A north-up raster transform: scaled, offset, and with y running downwards
/// so that the split's mirror correction is exercised too. Deliberately not
/// the identity - under the identity a coordinate transformed twice looks
/// exactly like one transformed once, which would hide the bug these tests
/// are here to catch.
static snail::transform::Affine appendTestTransform() {
  return snail::transform::Affine(2, 0, 100, 0, -3, 50);
}

/// A grid-space position in the world coordinates that transform implies
static snail::geometry::Coord atCell(double column, double row) {
  return snail::geometry::Coord(2 * column + 100, 50 - 3 * row);
}

TEST_CASE("Splitting appends to a shared accumulator", "[polygon][append]") {
  // Splitting a column of geometries fills one accumulator rather than one
  // result per geometry, so that the whole batch can be handed to Arrow as
  // it stands. Doing so must give exactly what splitting each on its own
  // and concatenating would, with offsets already absolute.
  snail::grid::Grid grid(4, 4, appendTestTransform());
  std::vector<std::vector<linestr>> polygons = {
      {{atCell(0.5, 0.5), atCell(1.5, 0.5), atCell(1.5, 1.5),
        atCell(0.5, 1.5), atCell(0.5, 0.5)}},
      // one with a hole, and one lying wholly inside a single cell
      {{atCell(0.2, 0.2), atCell(3.8, 0.2), atCell(3.8, 3.8),
        atCell(0.2, 3.8), atCell(0.2, 0.2)},
       {atCell(1.2, 1.2), atCell(1.2, 2.8), atCell(2.8, 2.8),
        atCell(2.8, 1.2), atCell(1.2, 1.2)}},
      {{atCell(2.1, 2.1), atCell(2.9, 2.1), atCell(2.9, 2.9),
        atCell(2.1, 2.9), atCell(2.1, 2.1)}},
  };

  snail::operations::PolygonPieces together;
  std::size_t expected_pieces = 0;
  std::vector<snail::geometry::Coord> expected_coordinates;
  for (const auto &rings : polygons) {
    std::vector<snail::operations::CoordSpan> spans(rings.begin(), rings.end());
    auto alone = snail::operations::splitPolygonGridPieces(spans, grid);
    expected_pieces += alone.size();
    expected_coordinates.insert(expected_coordinates.end(),
                                alone.coordinates.begin(),
                                alone.coordinates.end());
    snail::operations::splitPolygonGridPieces(spans, grid, together);
  }

  REQUIRE(expected_pieces > 0);
  REQUIRE(together.size() == expected_pieces);
  REQUIRE(together.coordinates.size() == expected_coordinates.size());
  for (std::size_t i = 0; i < expected_coordinates.size(); i++) {
    REQUIRE(together.coordinates[i].x == expected_coordinates[i].x);
    REQUIRE(together.coordinates[i].y == expected_coordinates[i].y);
  }

  // offsets index the accumulator as a whole: ascending, starting at zero,
  // and ending at what it holds
  REQUIRE(together.ring_offsets.front() == 0);
  REQUIRE(together.ring_offsets.back() ==
          (int32_t)together.coordinates.size());
  for (std::size_t r = 1; r < together.ring_offsets.size(); r++) {
    REQUIRE(together.ring_offsets[r] >= together.ring_offsets[r - 1]);
  }
  REQUIRE(together.polygon_offsets.front() == 0);
  REQUIRE(together.polygon_offsets.back() ==
          (int32_t)together.ring_offsets.size() - 1);
}

TEST_CASE("Linestring splitting appends to a shared accumulator",
          "[linestring][append]") {
  snail::grid::Grid grid(4, 4, appendTestTransform());
  std::vector<linestr> lines = {
      {atCell(0.5, 0.5), atCell(3.5, 0.5)},
      {atCell(0.5, 0.5), atCell(0.75, 0.5), atCell(1.5, 1.5)},
      // degenerate: a single repeated point contributes no piece at all
      {atCell(2.5, 2.5), atCell(2.5, 2.5)},
  };

  snail::operations::LinePieces together;
  std::size_t expected_pieces = 0;
  std::vector<snail::geometry::Coord> expected_coordinates;
  for (const auto &line : lines) {
    auto alone = snail::operations::splitLineStringGrid(line, grid);
    expected_pieces += alone.size();
    expected_coordinates.insert(expected_coordinates.end(),
                                alone.coordinates.begin(),
                                alone.coordinates.end());
    snail::operations::splitLineStringGrid(line, grid, false, together);
  }

  REQUIRE(expected_pieces > 0);
  REQUIRE(together.size() == expected_pieces);
  REQUIRE(together.coordinates.size() == expected_coordinates.size());
  for (std::size_t i = 0; i < expected_coordinates.size(); i++) {
    REQUIRE(together.coordinates[i].x == expected_coordinates[i].x);
    REQUIRE(together.coordinates[i].y == expected_coordinates[i].y);
  }
  REQUIRE(together.offsets.front() == 0);
  REQUIRE(together.offsets.back() == (int32_t)together.coordinates.size());
  for (std::size_t p = 1; p < together.offsets.size(); p++) {
    REQUIRE(together.offsets[p] >= together.offsets[p - 1]);
  }
}
