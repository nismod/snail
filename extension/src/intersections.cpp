#include <cstdint>
#include <fstream>
#include <iostream>
#include <ostream>
#include <sstream>
#include <string>
#include <tuple>
#include <vector>

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "geometry.hpp"
#include "grid.hpp"
#include "transform.hpp"
#include "operations.hpp"

namespace snail {

namespace py = pybind11;
namespace geo = geometry;

using linestr = std::vector<geometry::Coord>;

/// All coordinates of a geometry, as an (n, 2) array, in one bulk read
py::array_t<double, py::array::c_style | py::array::forcecast>
geometryCoordinates(py::object geometry) {
  return py::array_t<double, py::array::c_style | py::array::forcecast>::ensure(
      py::module_::import("shapely").attr("get_coordinates")(geometry));
}

linestr convert_py2cpp(py::object linestring_py) {
  auto coords = geometryCoordinates(linestring_py);
  linestr linestring;
  if (!coords || coords.ndim() != 2 || coords.shape(1) < 2) {
    return linestring;
  }
  auto xy = coords.unchecked<2>();
  linestring.reserve(xy.shape(0));
  for (py::ssize_t i = 0; i < xy.shape(0); i++) {
    linestring.push_back(geo::Coord(xy(i, 0), xy(i, 1)));
  }
  return linestring;
}

std::vector<py::object> convert_cpp2py(std::vector<linestr> splits) {
  const py::object shapely_linestr =
      py::module_::import("shapely.geometry").attr("LineString");

  std::vector<py::object> splits_py;
  std::vector<std::vector<double>> split_py;
  std::vector<double> point_py;
  for (auto split : splits) {
    for (auto point : split) {
      point_py.push_back(point.x);
      point_py.push_back(point.y);
      split_py.push_back(point_py);
      point_py.clear();
    }
    splits_py.push_back(shapely_linestr(split_py));
    split_py.clear();
  }
  return splits_py;
}

std::vector<py::object> splitLineString(py::object linestring_py, int nrows,
                                        int ncols,
                                        std::vector<double> transform) {
  linestr linestring = convert_py2cpp(linestring_py);
  transform::Affine affine(transform[0], transform[1], transform[2],
                           transform[3], transform[4], transform[5]);
  grid::Grid grid(ncols, nrows, affine);
  geometry::LineString line(linestring);
  std::vector<linestr> splits =
      operations::findIntersectionsLineString(line, grid);
  return convert_cpp2py(splits);
}

py::object splitPolygon(py::object polygon, int nrows, int ncols,
                        std::vector<double> transform) {
  // All rings of the polygon (exterior first, then any interior rings), in
  // one bulk coordinate read: each ring runs up to the point at which it
  // closes back on its start. Consecutive duplicate points are dropped as
  // we go, so that a repeated start point is not mistaken for the closure.
  auto coords = geometryCoordinates(polygon);
  std::vector<linestr> rings;
  if (coords && coords.ndim() == 2 && coords.shape(1) >= 2) {
    auto xy = coords.unchecked<2>();
    py::ssize_t n = xy.shape(0);
    py::ssize_t i = 0;
    while (i < n) {
      linestr ring;
      ring.push_back(geo::Coord(xy(i, 0), xy(i, 1)));
      i++;
      for (; i < n; i++) {
        geo::Coord point(xy(i, 0), xy(i, 1));
        if (point == ring.back()) {
          continue;
        }
        ring.push_back(point);
        if (ring.size() >= 3 && point == ring.front()) {
          i++;
          break;
        }
      }
      rings.push_back(std::move(ring));
    }
  }

  transform::Affine affine(transform[0], transform[1], transform[2],
                           transform[3], transform[4], transform[5]);
  grid::Grid grid(ncols, nrows, affine);

  std::vector<geo::Polygon> splits = operations::splitPolygonGrid(rings, grid);

  if (splits.empty()) {
    return py::list();
  }

  // Build all output geometries in one call with shapely's bulk constructor
  // (much faster than one Polygon() call per piece): a flat coordinate
  // array, ring extents and polygon extents as offset arrays
  std::size_t n_points = 0;
  std::size_t n_rings = 0;
  for (const geo::Polygon &split : splits) {
    n_points += split.exterior.size();
    n_rings += 1 + split.interiors.size();
    for (const linestr &interior : split.interiors) {
      n_points += interior.size();
    }
  }

  py::array_t<double> out_coords({(py::ssize_t)n_points, (py::ssize_t)2});
  py::array_t<std::int64_t> ring_offsets((py::ssize_t)n_rings + 1);
  py::array_t<std::int64_t> polygon_offsets((py::ssize_t)splits.size() + 1);
  auto coords_out = out_coords.mutable_unchecked<2>();
  auto ring_offsets_out = ring_offsets.mutable_unchecked<1>();
  auto polygon_offsets_out = polygon_offsets.mutable_unchecked<1>();

  py::ssize_t point = 0;
  py::ssize_t ring = 0;
  polygon_offsets_out(0) = 0;
  ring_offsets_out(0) = 0;
  for (std::size_t s = 0; s < splits.size(); s++) {
    for (const geo::Coord &p : splits[s].exterior) {
      coords_out(point, 0) = p.x;
      coords_out(point, 1) = p.y;
      point++;
    }
    ring_offsets_out(++ring) = point;
    for (const linestr &interior : splits[s].interiors) {
      for (const geo::Coord &p : interior) {
        coords_out(point, 0) = p.x;
        coords_out(point, 1) = p.y;
        point++;
      }
      ring_offsets_out(++ring) = point;
    }
    polygon_offsets_out(s + 1) = ring;
  }

  py::module_ shapely = py::module_::import("shapely");
  return shapely.attr("from_ragged_array")(
      shapely.attr("GeometryType").attr("POLYGON"), out_coords,
      py::make_tuple(ring_offsets, polygon_offsets));
}

std::tuple<int, int> get_cell_indices(py::object linestring, int nrows,
                                      int ncols,
                                      std::vector<double> transform) {
  py::tuple bounds = linestring.attr("bounds");
  double minx = (py::float_)bounds[0];
  double miny = (py::float_)bounds[1];
  double maxx = (py::float_)bounds[2];
  double maxy = (py::float_)bounds[3];
  geo::Coord midpoint = geo::Coord((maxx + minx) * 0.5, (maxy + miny) * 0.5);

  transform::Affine affine(transform[0], transform[1], transform[2],
                           transform[3], transform[4], transform[5]);
  grid::Grid grid(ncols, nrows, affine);
  return grid.cellIndices(midpoint);
}

} // namespace snail

PYBIND11_MODULE(intersections, m) {
  m.doc() = "Vector geometry to grid intersections";

  m.def("split_linestring", &snail::splitLineString,
        "Split LineString along a grid");
  m.def("get_cell_indices", &snail::get_cell_indices,
        "Get LineString cell indices in a grid");
  m.def("split_polygon", &snail::splitPolygon,
        "Split Polygon along a grid, returning Polygon pieces");
}
