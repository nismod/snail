#include <fstream>
#include <iostream>
#include <ostream>
#include <sstream>
#include <string>
#include <tuple>
#include <vector>

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

linestr convert_py2cpp(py::object linestring_py) {
  py::object coords = linestring_py.attr("coords");
  linestr linestring;
  for (py::size_t i = 0; i < py::len(coords); i++) {
    py::tuple xy = (py::tuple)coords[py::cast(i)];
    geo::Coord p((py::float_)xy[0], (py::float_)xy[1]);
    linestring.push_back(p);
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

std::vector<py::object> splitPolygon(py::object polygon, int nrows, int ncols,
                                     std::vector<double> transform) {
  // All rings of the polygon: exterior first, then any interiors (holes)
  std::vector<linestr> rings;
  rings.push_back(convert_py2cpp(polygon.attr("exterior")));
  py::object interiors = polygon.attr("interiors");
  for (py::handle interior : interiors) {
    rings.push_back(
        convert_py2cpp(py::reinterpret_borrow<py::object>(interior)));
  }

  transform::Affine affine(transform[0], transform[1], transform[2],
                           transform[3], transform[4], transform[5]);
  grid::Grid grid(ncols, nrows, affine);

  std::vector<geo::Polygon> splits = operations::splitPolygonGrid(rings, grid);

  const py::object shapely_polygon =
      py::module_::import("shapely.geometry").attr("Polygon");

  std::vector<py::object> splits_py;
  std::vector<std::vector<double>> ring_py;
  for (const geo::Polygon &split : splits) {
    for (const geo::Coord &point : split.exterior) {
      ring_py.push_back({point.x, point.y});
    }
    std::vector<std::vector<std::vector<double>>> interiors_py;
    for (const linestr &interior : split.interiors) {
      std::vector<std::vector<double>> interior_py;
      for (const geo::Coord &point : interior) {
        interior_py.push_back({point.x, point.y});
      }
      interiors_py.push_back(interior_py);
    }
    splits_py.push_back(shapely_polygon(ring_py, interiors_py));
    ring_py.clear();
  }
  return splits_py;
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
