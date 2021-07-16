#include <fstream>
#include <iostream>
#include <ostream>
#include <sstream>
#include <string>
#include <tuple>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "geofeatures.hpp"
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
std::vector<py::object> fun(py::object linestring_py, int nrows, int ncols,
                            std::vector<double> transform) {
  linestr linestring = convert_py2cpp(linestring_py);
  Affine affine(transform[0], transform[1], transform[2], transform[3],
                transform[4], transform[5]);
  Grid grid(ncols, nrows, affine);
  Feature f;
  f.geometry.insert(f.geometry.begin(), linestring.begin(), linestring.end());
  std::vector<linestr> splits = findIntersectionsLineString(f, grid);
  return convert_cpp2py(splits);
}

PYBIND11_MODULE(intersections, m) {
  m.doc() = "pybind11 example plugin"; // optional module docstring

  m.def("fun", &fun, "A function");
}
