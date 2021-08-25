#include <fstream>
#include <iostream>
#include <ostream>
#include <sstream>
#include <string>
#include <tuple>
#include <vector>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "geom.hpp"
#include "grid.hpp"
#include "transform.hpp"
#include "find_intersections_linestring.hpp"

namespace py = pybind11;
namespace geo = geometry;

using linestr = std::vector<geometry::Vec2<double>>;

py::object SHPLY_LINESTR =
    py::module_::import("shapely.geometry").attr("LineString");

linestr convert_py2cpp(py::object linestring_py) {
  py::object coords = linestring_py.attr("coords");
  linestr linestring;
  for (int i = 0; i < py::len(coords); i++) {
    py::tuple xy = (py::tuple)coords[py::cast(i)];
    geo::Vec2<double> p((py::float_)xy[0], (py::float_)xy[1]);
    linestring.push_back(p);
  }
  return linestring;
}

std::vector<py::object> convert_cpp2py(std::vector<linestr> splits) {

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
    splits_py.push_back(SHPLY_LINESTR(split_py));
    split_py.clear();
  }
  return splits_py;
}

std::vector<py::object> split(py::object linestring_py, int nrows, int ncols,
                            std::vector<double> transform) {
  linestr linestring = convert_py2cpp(linestring_py);
  Affine affine(transform[0], transform[1], transform[2], transform[3],
                transform[4], transform[5]);
  Grid grid(ncols, nrows, affine);
  geometry::LineString<double> line(linestring);
  std::vector<linestr> splits = findIntersectionsLineString(line, grid);
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
  Affine affine(transform[0], transform[1], transform[2], transform[3],
                transform[4], transform[5]);
  Grid grid(ncols, nrows, affine);
  geometry::LineString<double> line(exterior);
  std::vector<linestr> exterior_splits =
      findIntersectionsLineString(line, grid);
  std::vector<geometry::Vec2<double>> exterior_crossings;
  for (auto split : exterior_splits) {
    exterior_crossings.push_back(split[0]);
  }
  std::vector<linestr> horiz_splits = splitAlongGridlines(
      exterior_crossings, floor(miny) + 1, floor(maxy), "horizontal", grid);
  std::vector<linestr> vert_splits = splitAlongGridlines(
      exterior_crossings, floor(minx) + 1, floor(maxx), "vertical", grid);
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
  geo::Vec2<double> midpoint =
      geo::Vec2<double>((maxx + minx) * 0.5, (maxy + miny) * 0.5);

  Affine affine(transform[0], transform[1], transform[2], transform[3],
                transform[4], transform[5]);
  Grid grid(ncols, nrows, affine);
  geo::Vec2<int> cell = grid.cellIndices(midpoint);
  return std::make_tuple(cell.x, cell.y);
}

PYBIND11_MODULE(intersections, m) {
  m.doc() = "pybind11 example plugin"; // optional module docstring

  m.def("split", &split, "A function");
  m.def("get_cell_indices", &get_cell_indices, "Getting cell indices");
  m.def("split_polygon", &splitPolygon, "Splitting a polygon");
}
