#include <cstdint>
#include <cstring>
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
#include "geoarrow.hpp"
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

/// Copy an offset array out as the int64 numpy array shapely expects
py::array_t<std::int64_t> offsetArray(const std::vector<std::size_t> &offsets) {
  py::array_t<std::int64_t> array((py::ssize_t)offsets.size());
  std::int64_t *out = array.mutable_data();
  for (std::size_t i = 0; i < offsets.size(); i++) {
    out[i] = (std::int64_t)offsets[i];
  }
  return array;
}

/// Build shapely Polygons from split pieces, in a single bulk call. The
/// pieces are already in the flat layout shapely consumes, so the
/// coordinates are one contiguous copy.
py::object polygonsFromPieces(const operations::PolygonPieces &pieces) {
  static_assert(sizeof(geo::Coord) == 2 * sizeof(double),
                "Coord must be a bare pair of doubles to copy in bulk");

  py::array_t<double> coords(
      {(py::ssize_t)pieces.coordinates.size(), (py::ssize_t)2});
  std::memcpy(coords.mutable_data(), pieces.coordinates.data(),
              pieces.coordinates.size() * sizeof(geo::Coord));

  py::module_ shapely = py::module_::import("shapely");
  return shapely.attr("from_ragged_array")(
      shapely.attr("GeometryType").attr("POLYGON"), coords,
      py::make_tuple(offsetArray(pieces.ring_offsets),
                     offsetArray(pieces.polygon_offsets)));
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

/// Build shapely LineStrings from split pieces, in a single bulk call
py::object lineStringsFromPieces(const operations::LinePieces &pieces) {
  py::array_t<double> coords(
      {(py::ssize_t)pieces.coordinates.size(), (py::ssize_t)2});
  std::memcpy(coords.mutable_data(), pieces.coordinates.data(),
              pieces.coordinates.size() * sizeof(geo::Coord));

  py::module_ shapely = py::module_::import("shapely");
  return shapely.attr("from_ragged_array")(
      shapely.attr("GeometryType").attr("LINESTRING"), coords,
      py::make_tuple(offsetArray(pieces.offsets)));
}

py::object splitLineString(py::object linestring_py, int nrows, int ncols,
                           std::vector<double> transform, bool bounded = false) {
  linestr linestring = convert_py2cpp(linestring_py);
  transform::Affine affine(transform[0], transform[1], transform[2],
                           transform[3], transform[4], transform[5]);
  grid::Grid grid(ncols, nrows, affine);
  operations::LinePieces splits =
      operations::splitLineStringGrid(linestring, grid, bounded);
  if (splits.size() == 0) {
    return py::list();
  }
  return lineStringsFromPieces(splits);
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

  operations::PolygonPieces splits =
      operations::splitPolygonGridPieces(rings, grid);

  if (splits.size() == 0) {
    return py::list();
  }
  return polygonsFromPieces(splits);
}

std::tuple<int, int> get_cell_indices(py::object linestring, int nrows, int ncols,
                                      std::vector<double> transform) {
  py::tuple bounds = linestring.attr("bounds");
  double minx = (py::float_)bounds[0];
  double miny = (py::float_)bounds[1];
  double maxx = (py::float_)bounds[2];
  double maxy = (py::float_)bounds[3];
  // NOLINTNEXTLINE(cppcoreguidelines-avoid-magic-numbers)
  geo::Coord midpoint = geo::Coord((maxx + minx) * 0.5, (maxy + miny) * 0.5);

  transform::Affine affine(transform[0], transform[1], transform[2], transform[3], transform[4],
                           transform[5]); // NOLINT(cppcoreguidelines-avoid-magic-numbers)
  grid::Grid grid(ncols, nrows, affine);
  return grid.cellIndices(midpoint);
}

} // namespace snail

// NOLINTNEXTLINE
PYBIND11_MODULE(intersections, m) {
  m.doc() = "Vector geometry to grid intersections";

  m.def("split_linestring", &snail::splitLineString,
        pybind11::arg("linestring_py"), pybind11::arg("nrows"),
        pybind11::arg("ncols"), pybind11::arg("transform"),
        pybind11::arg("bounded") = false,
        "Split LineString along a grid, returning LineString pieces");
  m.def("get_cell_indices", &snail::get_cell_indices,
        pybind11::arg("linestring"), pybind11::arg("nrows"),
        pybind11::arg("ncols"), pybind11::arg("transform"),
        "Get LineString cell indices in a grid");
  m.def("split_polygon", &snail::splitPolygon, pybind11::arg("polygon"),
        pybind11::arg("nrows"), pybind11::arg("ncols"),
        pybind11::arg("transform"),
        "Split Polygon along a grid, returning Polygon pieces");

  // Whole geometry columns are exchanged as GeoArrow arrays: this
  // registers split_linestrings, split_polygons and GeoArrowArray
  snail::geoarrow::register_module(m);
}
