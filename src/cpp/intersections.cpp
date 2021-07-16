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

double fun(py::object linestring_frompy, int nrows, int ncols, std::vector<double> transform) {
  py::object coords = linestring_frompy.attr("coords");
  int size = py::len(coords);
  std::vector<geo::Vec2<double>> linestring;
  for (int i = 0; i<size;i++) {
    py::tuple xy = (py::tuple)coords[py::cast(i)];
    geo::Vec2<double> p((py::float_)xy[0], (py::float_)xy[1]);
    linestring.push_back(p);
  }
  Affine affine(transform[0],
		transform[1],
		transform[2],
		transform[3],
		transform[4],
		transform[5]);
  Grid grid(ncols, nrows, affine);
  Feature f;
  f.geometry.insert(f.geometry.begin(), linestring.begin(), linestring.end());
  std::vector<linestr> splits = findIntersectionsLineString(f, grid);
}

PYBIND11_MODULE(intersections, m) {
  m.doc() = "pybind11 example plugin"; // optional module docstring

  m.def("fun", &fun, "A function");
}
