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
  /// It is assumed that polygon is oriented (counter-clockwise)
  py::tuple bounds = polygon.attr("bounds");
  double minx = (py::float_)bounds[0];
  double miny = (py::float_)bounds[1];
  double maxx = (py::float_)bounds[2];
  double maxy = (py::float_)bounds[3];

  linestr exterior = convert_py2cpp(polygon.attr("exterior"));
  transform::Affine affine(transform[0], transform[1], transform[2],
                           transform[3], transform[4], transform[5]);
  grid::Grid grid(ncols, nrows, affine);

  // Corners of geometry bbox as cell indices
  geometry::Coord ll = grid.world_to_grid * geometry::Coord(minx, miny);
  geometry::Coord ur = grid.world_to_grid * geometry::Coord(maxx, maxy);

  geometry::LineString line(exterior);
  std::vector<linestr> exterior_splits =
      operations::findIntersectionsLineString(line, grid);
  std::vector<geometry::Coord> exterior_with_crossings;
  for (auto split : exterior_splits) {
    exterior_with_crossings.insert(exterior_with_crossings.end(), split.begin(),
                                   split.end());
  }

  std::vector<linestr> horiz_splits = operations::splitAlongGridlines(
      exterior_with_crossings, floor(std::min(ll.y, ur.y)),
      ceil(std::max(ll.y, ur.y)) + 1, operations::Direction::horizontal, grid);
  std::vector<linestr> vert_splits = operations::splitAlongGridlines(
      exterior_with_crossings, floor(std::min(ll.x, ur.x)),
      ceil(std::max(ll.x, ur.x)) + 1, operations::Direction::vertical, grid);

  std::vector<linestr> all_splits;
  all_splits.insert(all_splits.end(), exterior_splits.begin(),
                    exterior_splits.end());
  all_splits.insert(all_splits.end(), horiz_splits.begin(), horiz_splits.end());
  all_splits.insert(all_splits.end(), vert_splits.begin(), vert_splits.end());

  return convert_cpp2py(all_splits);
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
  m.def("split_polygon", &snail::splitPolygon, "Split Polygon along a grid");
}
