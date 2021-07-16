#include <fstream>
#include <iostream>
#include <ostream>
#include <sstream>
#include <string>
#include <tuple>

#include <pybind11/pybind11.h>
#include "geofeatures.hpp"
#include "geom.hpp"
#include "grid.hpp"
#include "transform.hpp"
#include "find_intersections_linestring.hpp"

namespace py = pybind11;
namespace geo = geometry;

double fun(py::object linestring_frompy) {
  py::object coords = linestring_frompy.attr("coords");
  int size = py::len(coords);
  std::vector<geo::Vec2<double>> linestring;
  for (int i = 0; i<size;i++) {
    py::tuple xy = (py::tuple)coords[py::cast(i)];
    geo::Vec2<double> p((py::float_)xy[0], (py::float_)xy[1]);
    linestring.push_back(p);
  }
  // Now find intersections...
}

PYBIND11_MODULE(intersections, m) {
  m.doc() = "pybind11 example plugin"; // optional module docstring

  m.def("fun", &fun, "A function");
}
