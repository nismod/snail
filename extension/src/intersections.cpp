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

  operations::PolygonPieces splits =
      operations::splitPolygonGridPieces(rings, grid);

  if (splits.size() == 0) {
    return py::list();
  }
  return polygonsFromPieces(splits);
}

py::object splitPolygons(py::object polygons, int nrows, int ncols,
                         std::vector<double> transform) {
  // Read every polygon in one call, as flat coordinate and offset arrays
  // (the same layout the results are returned in), so that splitting a
  // whole table of features crosses into Python twice rather than twice
  // per feature
  py::module_ shapely = py::module_::import("shapely");
  py::tuple ragged = shapely.attr("to_ragged_array")(polygons);
  int geometry_type = py::cast<int>(ragged[0].attr("value"));
  int polygon_type =
      py::cast<int>(shapely.attr("GeometryType").attr("POLYGON").attr("value"));
  if (geometry_type != polygon_type) {
    throw py::type_error("split_polygons expects Polygon geometries; explode "
                         "MultiPolygons before splitting");
  }

  using OffsetArray =
      py::array_t<std::int64_t, py::array::c_style | py::array::forcecast>;
  auto coords =
      py::array_t<double, py::array::c_style | py::array::forcecast>::ensure(
          ragged[1]);
  py::tuple offsets = ragged[2];
  auto ring_offsets = OffsetArray::ensure(offsets[0]);
  auto polygon_offsets = OffsetArray::ensure(offsets[1]);

  auto xy = coords.unchecked<2>();
  auto ring_at = ring_offsets.unchecked<1>();
  auto polygon_at = polygon_offsets.unchecked<1>();

  transform::Affine affine(transform[0], transform[1], transform[2],
                           transform[3], transform[4], transform[5]);
  grid::Grid grid(ncols, nrows, affine);

  operations::PolygonPieces all_pieces;
  std::vector<std::int64_t> parent;
  std::vector<linestr> rings;
  py::ssize_t n_polygons = polygon_at.shape(0) - 1;
  for (py::ssize_t p = 0; p < n_polygons; p++) {
    rings.clear();
    for (std::int64_t r = polygon_at(p); r < polygon_at(p + 1); r++) {
      linestr ring;
      ring.reserve(ring_at(r + 1) - ring_at(r));
      for (std::int64_t i = ring_at(r); i < ring_at(r + 1); i++) {
        ring.push_back(geo::Coord(xy(i, 0), xy(i, 1)));
      }
      rings.push_back(std::move(ring));
    }

    operations::PolygonPieces pieces =
        operations::splitPolygonGridPieces(rings, grid);

    // concatenate onto the combined result, shifting the offsets
    std::size_t coordinate_base = all_pieces.coordinates.size();
    std::size_t ring_base = all_pieces.ring_offsets.size() - 1;
    all_pieces.coordinates.insert(all_pieces.coordinates.end(),
                                  pieces.coordinates.begin(),
                                  pieces.coordinates.end());
    for (std::size_t r = 1; r < pieces.ring_offsets.size(); r++) {
      all_pieces.ring_offsets.push_back(coordinate_base +
                                        pieces.ring_offsets[r]);
    }
    for (std::size_t q = 1; q < pieces.polygon_offsets.size(); q++) {
      all_pieces.polygon_offsets.push_back(ring_base +
                                           pieces.polygon_offsets[q]);
      parent.push_back((std::int64_t)p);
    }
  }

  py::array_t<std::int64_t> parent_out((py::ssize_t)parent.size());
  if (!parent.empty()) {
    std::memcpy(parent_out.mutable_data(), parent.data(),
                parent.size() * sizeof(std::int64_t));
  }
  return py::make_tuple(polygonsFromPieces(all_pieces), parent_out);
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
  m.def("split_polygons", &snail::splitPolygons,
        "Split Polygons along a grid, returning Polygon pieces and the index "
        "of the polygon each piece came from");
}
