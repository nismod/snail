// Stress tests for the grid splitting routines.
//
// These generate many shapes across several grids and check the invariants
// that must hold whatever the input, rather than specific coordinates:
// splitting conserves area (or length), every piece lies within one cell,
// and the rings that come out are well formed. Shapes are drawn from a
// deterministic generator, so a failure here reproduces everywhere.

#include <catch2/catch.hpp>
#include <cmath>
#include <cstdint>
#include <limits>
#include <string>
#include <utility>
#include <vector>

#include "geometry.hpp"
#include "grid.hpp"
#include "operations.hpp"
#include "transform.hpp"

using snail::geometry::Coord;
using linestr = std::vector<Coord>;

namespace {

/// Deterministic generator (xorshift64*), so that these cases are identical
/// on every platform and every run
struct Rng {
  std::uint64_t state;
  explicit Rng(std::uint64_t seed) : state(seed ? seed : 1) {}
  std::uint64_t next() {
    state ^= state >> 12;
    state ^= state << 25;
    state ^= state >> 27;
    return state * 0x2545F4914F6CDD1DULL;
  }
  double uniform(double low, double high) {
    // 53 bits of mantissa from the top of the word
    double unit = (double)(next() >> 11) / (double)(1ULL << 53);
    return low + unit * (high - low);
  }
  int integer(int low, int high) {
    return low + (int)(next() % (std::uint64_t)(high - low + 1));
  }
};

double ringArea2(const linestr &ring) {
  if (ring.size() < 3) {
    return 0;
  }
  const Coord origin = ring.front();
  double area2 = 0;
  for (std::size_t i = 0; i + 1 < ring.size(); i++) {
    area2 += (ring[i].x - origin.x) * (ring[i + 1].y - origin.y) -
             (ring[i + 1].x - origin.x) * (ring[i].y - origin.y);
  }
  return area2;
}

/// Signed area of a polygon with holes; holes wind opposite to the exterior
/// so their contribution subtracts
double polygonArea(const snail::geometry::Polygon &polygon) {
  double area2 = ringArea2(polygon.exterior);
  for (const linestr &interior : polygon.interiors) {
    area2 += ringArea2(interior);
  }
  return area2 / 2.0;
}

double ringsArea(const std::vector<linestr> &rings) {
  double area2 = 0;
  for (std::size_t r = 0; r < rings.size(); r++) {
    double ring = ringArea2(rings[r]);
    // the first ring is the exterior; the rest are holes, whichever way
    // round they were given
    area2 += (r == 0) ? std::fabs(ring) : -std::fabs(ring);
  }
  return area2 / 2.0;
}

linestr closeRing(linestr ring) {
  if (!ring.empty() && !(ring.front() == ring.back())) {
    ring.push_back(ring.front());
  }
  return ring;
}

/// A convex-ish blob: radii varied around a circle, so edges cross grid
/// lines at many angles
linestr blob(Rng &rng, Coord centre, double radius, int segments) {
  linestr ring;
  for (int i = 0; i < segments; i++) {
    double angle = 2 * snail::utils::PI * i / segments;
    double r = radius * rng.uniform(0.75, 1.0);
    ring.push_back(
        Coord(centre.x + r * std::cos(angle), centre.y + r * std::sin(angle)));
  }
  return closeRing(ring);
}

/// A comb: thin teeth reaching up from a spine, so that single cells hold
/// several disjoint pieces and many edges run nearly parallel to grid lines
linestr comb(Rng &rng, Coord origin, int teeth, double tooth_width,
             double tooth_height, double spine_height) {
  linestr ring;
  double x = origin.x;
  ring.push_back(Coord(x, origin.y));
  for (int i = 0; i < teeth; i++) {
    double height = tooth_height * rng.uniform(0.5, 1.0);
    ring.push_back(Coord(x, origin.y + spine_height));
    ring.push_back(Coord(x, origin.y + spine_height + height));
    ring.push_back(Coord(x + tooth_width, origin.y + spine_height + height));
    ring.push_back(Coord(x + tooth_width, origin.y + spine_height));
    x += 2 * tooth_width;
    ring.push_back(Coord(x, origin.y + spine_height));
  }
  ring.push_back(Coord(x, origin.y));
  return closeRing(ring);
}

/// An axis-aligned rectangle placed exactly on grid lines, so that whole
/// edges are collinear with cell borders and corners land on cell corners
linestr alignedRectangle(Rng &rng, const snail::grid::Grid &grid) {
  double i = grid.grid_to_world.a;
  double j = grid.grid_to_world.e;
  double x0 = grid.grid_to_world.c + i * rng.integer(1, 5);
  double y0 = grid.grid_to_world.f + j * rng.integer(1, 5);
  double x1 = x0 + i * rng.integer(1, 4);
  double y1 = y0 + j * rng.integer(1, 4);
  linestr ring = {Coord(x0, y0), Coord(x1, y0), Coord(x1, y1), Coord(x0, y1)};
  return closeRing(ring);
}

/// A long thin sliver at an arbitrary angle
linestr sliver(Rng &rng, Coord centre, double length, double width) {
  double angle = rng.uniform(0, 2 * snail::utils::PI);
  double dx = std::cos(angle), dy = std::sin(angle);
  double nx = -dy * width / 2, ny = dx * width / 2;
  Coord a(centre.x - dx * length / 2, centre.y - dy * length / 2);
  Coord b(centre.x + dx * length / 2, centre.y + dy * length / 2);
  linestr ring = {Coord(a.x + nx, a.y + ny), Coord(b.x + nx, b.y + ny),
                  Coord(b.x - nx, b.y - ny), Coord(a.x - nx, a.y - ny)};
  return closeRing(ring);
}

/// A square with a row of square holes punched through it. The holes are
/// disjoint, as interior rings of one polygon must be: nesting them would
/// not be a valid polygon.
std::vector<linestr> punched(Coord centre, double radius, int holes) {
  std::vector<linestr> rings;
  rings.push_back(closeRing({Coord(centre.x - radius, centre.y - radius),
                             Coord(centre.x + radius, centre.y - radius),
                             Coord(centre.x + radius, centre.y + radius),
                             Coord(centre.x - radius, centre.y + radius)}));
  // lay the holes along the middle, each within its own share of the width
  double pitch = 2 * radius / holes;
  double r = pitch * 0.3;
  for (int h = 0; h < holes; h++) {
    double cx = centre.x - radius + pitch * (h + 0.5);
    rings.push_back(
        closeRing({Coord(cx - r, centre.y - r), Coord(cx + r, centre.y - r),
                   Coord(cx + r, centre.y + r), Coord(cx - r, centre.y + r)}));
  }
  return rings;
}

struct GridCase {
  std::string name;
  snail::grid::Grid grid;
};

std::vector<GridCase> gridCases() {
  return {
      {"unit", snail::grid::Grid(64, 64, snail::transform::Affine())},
      {"fractional",
       snail::grid::Grid(64, 64,
                         snail::transform::Affine(0.3, 0, -1.7, 0, 0.3, 2.1))},
      {"mirrored (y south)",
       snail::grid::Grid(64, 64, snail::transform::Affine(1, 0, 0, 0, -1, 40))},
      {"large offset",
       snail::grid::Grid(
           64, 64, snail::transform::Affine(2.5, 0, 400000, 0, -2.5, 6000000))},
      {"coarse (whole shape in one cell)",
       snail::grid::Grid(
           4, 4, snail::transform::Affine(1000, 0, -2000, 0, 1000, -2000))},
      {"fine (many cells)",
       snail::grid::Grid(4096, 4096,
                         snail::transform::Affine(0.05, 0, 0, 0, 0.05, 0))},
  };
}

/// Map a shape generated in cell units onto the grid's world coordinates,
/// so that shapes sit inside the raster whatever its transform - as real
/// features do - while spanning a comparable number of cells on each
linestr toWorld(const linestr &ring, const snail::grid::Grid &grid) {
  linestr out;
  out.reserve(ring.size());
  for (const Coord &point : ring) {
    out.push_back(grid.grid_to_world * point);
  }
  return out;
}

std::vector<linestr> toWorld(const std::vector<linestr> &rings,
                             const snail::grid::Grid &grid) {
  std::vector<linestr> out;
  out.reserve(rings.size());
  for (const linestr &ring : rings) {
    out.push_back(toWorld(ring, grid));
  }
  return out;
}

/// The cell a piece lies in. Its vertices may sit on cell borders, so take
/// the middle of its extent, which is inside the cell.
std::pair<long, long> pieceCellOf(const snail::geometry::Polygon &piece,
                                  const snail::grid::Grid &grid) {
  double gx_min = 1e300, gx_max = -1e300, gy_min = 1e300, gy_max = -1e300;
  for (const Coord &point : piece.exterior) {
    Coord g = grid.world_to_grid * point;
    gx_min = std::min(gx_min, g.x);
    gx_max = std::max(gx_max, g.x);
    gy_min = std::min(gy_min, g.y);
    gy_max = std::max(gy_max, g.y);
  }
  return std::make_pair((long)std::floor((gx_min + gx_max) / 2),
                        (long)std::floor((gy_min + gy_max) / 2));
}

/// How far the split area may drift from the input's. Splitting moves each
/// vertex into grid coordinates and back, which can shift it by an ulp of
/// the largest magnitude involved in that round trip; the shift acts along
/// the whole boundary, so the area it accounts for grows with that
/// magnitude and with the perimeter. The magnitude is not the shape's own
/// coordinates but the grid's offset, which for a projected CRS runs to
/// millions - a shape a long way from its grid's origin is held to an
/// absolute precision set by that distance, not by its own size.
double areaTolerance(const std::vector<linestr> &rings,
                     const snail::grid::Grid &grid, double expected) {
  double scale = std::max(std::fabs(grid.grid_to_world.c),
                          std::fabs(grid.grid_to_world.f));
  double perimeter = 0;
  for (const linestr &ring : rings) {
    for (std::size_t i = 0; i < ring.size(); i++) {
      scale = std::max(scale, std::fabs(ring[i].x));
      scale = std::max(scale, std::fabs(ring[i].y));
      if (i + 1 < ring.size()) {
        perimeter +=
            std::hypot(ring[i + 1].x - ring[i].x, ring[i + 1].y - ring[i].y);
      }
    }
  }
  const double eps = std::numeric_limits<double>::epsilon();
  return 8 * eps * scale * perimeter + 1e-9 * std::fabs(expected);
}

/// Check everything that must hold of a split, whatever the input
void checkPieces(const std::vector<snail::geometry::Polygon> &pieces,
                 const std::vector<linestr> &rings,
                 const snail::grid::Grid &grid, const std::string &what) {
  double expected = ringsArea(rings);
  double total = 0;
  for (const snail::geometry::Polygon &piece : pieces) {
    total += polygonArea(piece);

    INFO(what << ": ring shape");
    REQUIRE(piece.exterior.size() >= 4);
    REQUIRE(piece.exterior.front() == piece.exterior.back());

    // exteriors wind counter-clockwise, holes clockwise
    INFO(what << ": winding");
    REQUIRE(ringArea2(piece.exterior) > 0);
    for (const linestr &interior : piece.interiors) {
      REQUIRE(interior.size() >= 4);
      REQUIRE(interior.front() == interior.back());
      REQUIRE(ringArea2(interior) < 0);
    }

    // the piece lies within a single cell: in grid coordinates its extent
    // is at most one unit on each axis
    double gx_min = 1e300, gx_max = -1e300, gy_min = 1e300, gy_max = -1e300;
    for (const Coord &point : piece.exterior) {
      Coord g = grid.world_to_grid * point;
      gx_min = std::min(gx_min, g.x);
      gx_max = std::max(gx_max, g.x);
      gy_min = std::min(gy_min, g.y);
      gy_max = std::max(gy_max, g.y);
    }
    INFO(what << ": piece spans more than one cell");
    REQUIRE(gx_max - gx_min <= 1 + 1e-6);
    REQUIRE(gy_max - gy_min <= 1 + 1e-6);
  }

  INFO(what << ": area conservation");
  REQUIRE(std::fabs(total - expected) <= areaTolerance(rings, grid, expected));
}

} // namespace

TEST_CASE("Random shapes split conserving area", "[stress]") {
  for (const GridCase &grid_case : gridCases()) {
    Rng rng(0x5A17ULL);
    for (int trial = 0; trial < 40; trial++) {
      Coord centre(rng.uniform(3, 12), rng.uniform(3, 12));
      std::vector<linestr> rings;
      std::string what = grid_case.name + " trial " + std::to_string(trial);

      switch (trial % 5) {
      case 0:
        rings = {blob(rng, centre, rng.uniform(0.3, 5.0), rng.integer(3, 40))};
        what += " blob";
        break;
      case 1:
        rings = {comb(rng, centre, rng.integer(2, 8), rng.uniform(0.1, 0.9),
                      rng.uniform(0.5, 4.0), rng.uniform(0.2, 2.0))};
        what += " comb";
        break;
      case 2:
        rings = {alignedRectangle(rng, snail::grid::Grid())};
        what += " aligned rectangle";
        break;
      case 3:
        rings = {
            sliver(rng, centre, rng.uniform(1.0, 9.0), rng.uniform(0.01, 0.3))};
        what += " sliver";
        break;
      default:
        rings = punched(centre, rng.uniform(1.0, 5.0), rng.integer(1, 4));
        what += " punched";
        break;
      }
      rings = toWorld(rings, grid_case.grid);

      INFO(what);
      std::vector<snail::geometry::Polygon> pieces =
          snail::operations::splitPolygonGrid(rings, grid_case.grid);
      checkPieces(pieces, rings, grid_case.grid, what);
    }
  }
}

TEST_CASE("Shapes on grid lines split conserving area", "[stress]") {
  // Vertices and whole edges placed exactly on grid lines and cell corners,
  // which is where crossing detection is most delicate
  snail::grid::Grid grid(64, 64, snail::transform::Affine());
  Rng rng(0x6C1DULL);

  for (int trial = 0; trial < 60; trial++) {
    // a rectangle with some corners snapped onto the grid and others not
    double x0 = rng.integer(1, 6);
    double y0 = rng.integer(1, 6);
    double x1 = x0 + rng.integer(1, 3) + (trial % 2 ? 0.0 : 0.37);
    double y1 = y0 + rng.integer(1, 3) + (trial % 3 ? 0.0 : 0.42);
    linestr ring =
        closeRing({Coord(x0, y0), Coord(x1, y0), Coord(x1, y1), Coord(x0, y1)});

    // an L notched out of it, with the notch edges on grid lines
    if (trial % 4 == 0) {
      double mx = std::floor((x0 + x1) / 2);
      double my = std::floor((y0 + y1) / 2);
      if (mx > x0 && mx < x1 && my > y0 && my < y1) {
        ring = closeRing({Coord(x0, y0), Coord(x1, y0), Coord(x1, my),
                          Coord(mx, my), Coord(mx, y1), Coord(x0, y1)});
      }
    }

    std::vector<linestr> rings = {ring};
    std::string what = "gridline trial " + std::to_string(trial);
    INFO(what);
    std::vector<snail::geometry::Polygon> pieces =
        snail::operations::splitPolygonGrid(rings, grid);
    checkPieces(pieces, rings, grid, what);
  }
}

TEST_CASE("Hole touching a cell border splits into separate pieces",
          "[stress]") {
  // A hole tangent to a cell border pinches the covered region in two; it
  // must not come back as one piece with a hole touching its shell, which
  // is not a valid polygon
  snail::grid::Grid grid(64, 64, snail::transform::Affine());
  std::vector<linestr> rings = {closeRing({Coord(4.3, 8.2), Coord(4.8, 8.8),
                                           Coord(5.3, 8.2), Coord(4.8, 7.7)}),
                                closeRing({Coord(4.6, 8.2), Coord(4.8, 8.0),
                                           Coord(5.0, 8.2), Coord(4.8, 8.4)})};

  std::vector<snail::geometry::Polygon> pieces =
      snail::operations::splitPolygonGrid(rings, grid);
  checkPieces(pieces, rings, grid, "tangent hole");

  // the cell holding the hole contributes two pieces, neither with a hole
  int in_cell = 0;
  for (const snail::geometry::Polygon &piece : pieces) {
    if (pieceCellOf(piece, grid) == std::make_pair(4L, 8L)) {
      in_cell++;
      REQUIRE(piece.interiors.empty());
    }
  }
  REQUIRE(in_cell == 2);
}

namespace {

/// A wandering linestring, which may double back and touch grid lines
linestr wander(Rng &rng, Coord start, int steps, double step) {
  linestr line = {start};
  Coord at = start;
  for (int i = 0; i < steps; i++) {
    double angle = rng.uniform(0, 2 * snail::utils::PI);
    at = Coord(at.x + step * std::cos(angle) * rng.uniform(0.2, 1.0),
               at.y + step * std::sin(angle) * rng.uniform(0.2, 1.0));
    line.push_back(at);
  }
  return line;
}

double lineLength(const linestr &line) {
  double length = 0;
  for (std::size_t i = 0; i + 1 < line.size(); i++) {
    length += std::hypot(line[i + 1].x - line[i].x, line[i + 1].y - line[i].y);
  }
  return length;
}

} // namespace

TEST_CASE("Random linestrings split conserving length", "[stress]") {
  for (const GridCase &grid_case : gridCases()) {
    Rng rng(0xB0A7ULL);
    for (int trial = 0; trial < 40; trial++) {
      Coord start(rng.uniform(3, 12), rng.uniform(3, 12));
      linestr line;
      std::string what = grid_case.name + " trial " + std::to_string(trial);

      if (trial % 3 == 0) {
        // straight, at an arbitrary angle
        double angle = rng.uniform(0, 2 * snail::utils::PI);
        double length = rng.uniform(0.05, 9.0);
        line = {start, Coord(start.x + length * std::cos(angle),
                             start.y + length * std::sin(angle))};
        what += " straight";
      } else if (trial % 3 == 1) {
        line = wander(rng, start, rng.integer(2, 12), rng.uniform(0.1, 3.0));
        what += " wandering";
      } else {
        // axis-aligned staircase, running along and across grid lines
        line = {start};
        Coord at = start;
        for (int i = 0; i < rng.integer(2, 8); i++) {
          at = (i % 2) ? Coord(at.x, std::floor(at.y) + rng.integer(1, 3))
                       : Coord(std::floor(at.x) + rng.integer(1, 3), at.y);
          line.push_back(at);
        }
        what += " staircase";
      }
      line = toWorld(line, grid_case.grid);

      INFO(what);
      snail::operations::LinePieces split =
          snail::operations::splitLineStringGrid(line, grid_case.grid);
      std::vector<linestr> pieces;
      for (std::size_t p = 0; p + 1 < split.offsets.size(); p++) {
        pieces.push_back(
            linestr(split.coordinates.begin() + split.offsets[p],
                    split.coordinates.begin() + split.offsets[p + 1]));
      }

      // the pieces are the line, cut up: same total length, joined end to
      // end, from the line's start to its end
      double total = 0;
      for (const linestr &piece : pieces) {
        INFO(what << ": piece shape");
        REQUIRE(piece.size() >= 2);
        total += lineLength(piece);

        // each piece lies within one cell, which is what lets a raster
        // value be looked up per piece
        double gx_min = 1e300, gx_max = -1e300;
        double gy_min = 1e300, gy_max = -1e300;
        for (const Coord &point : piece) {
          Coord g = grid_case.grid.world_to_grid * point;
          gx_min = std::min(gx_min, g.x);
          gx_max = std::max(gx_max, g.x);
          gy_min = std::min(gy_min, g.y);
          gy_max = std::max(gy_max, g.y);
        }
        INFO(what << ": piece spans more than one cell");
        REQUIRE(gx_max - gx_min <= 1 + 1e-6);
        REQUIRE(gy_max - gy_min <= 1 + 1e-6);
      }
      INFO(what << ": length conservation");
      double expected = lineLength(line);
      REQUIRE(std::fabs(total - expected) <=
              1e-9 * expected +
                  8 * std::numeric_limits<double>::epsilon() *
                      std::max(std::fabs(grid_case.grid.grid_to_world.c),
                               std::fabs(grid_case.grid.grid_to_world.f)) *
                      (double)line.size());

      if (!pieces.empty()) {
        INFO(what << ": endpoints preserved");
        REQUIRE(pieces.front().front() == line.front());
        REQUIRE(pieces.back().back() == line.back());
        INFO(what << ": pieces join end to end");
        for (std::size_t i = 0; i + 1 < pieces.size(); i++) {
          REQUIRE(pieces[i].back() == pieces[i + 1].front());
        }
      }
    }
  }
}
