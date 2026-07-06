/// GeoArrow-based vectorised linestring splitting (see issue #14).
///
/// Accepts a GeoArrow "geoarrow.linestring" array - as produced by
/// geopandas.GeoSeries.to_arrow(geometry_encoding="geoarrow") from a
/// geometry column - via the Arrow PyCapsule C data interface, splits
/// every linestring along the gridlines of a raster grid, and returns a
/// new GeoArrow linestring array (plus an index array mapping each split
/// back to its parent linestring, so feature attributes can be joined).
///
/// The Arrow C data interface is a stable C ABI, so no Arrow library
/// dependency is needed: coordinates are read directly from the Arrow
/// buffers with no per-geometry Python object handling and no copies of
/// the input.
///
/// References:
/// - https://arrow.apache.org/docs/format/CDataInterface.html
/// - https://arrow.apache.org/docs/format/CDataInterface/PyCapsuleInterface.html
/// - https://geoarrow.org/format.html

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "geoarrow.hpp"
#include "geometry.hpp"
#include "grid.hpp"
#include "operations.hpp"
#include "transform.hpp"

// Arrow C data interface structs - a frozen ABI, defined once per project
// https://arrow.apache.org/docs/format/CDataInterface.html#structure-definitions
#ifndef ARROW_C_DATA_INTERFACE
#define ARROW_C_DATA_INTERFACE

#define ARROW_FLAG_DICTIONARY_ORDERED 1
#define ARROW_FLAG_NULLABLE 2
#define ARROW_FLAG_MAP_KEYS_SORTED 4

struct ArrowSchema {
  // Array type description
  const char *format;
  const char *name;
  const char *metadata;
  int64_t flags;
  int64_t n_children;
  struct ArrowSchema **children;
  struct ArrowSchema *dictionary;

  // Release callback
  void (*release)(struct ArrowSchema *);
  // Opaque producer-specific data
  void *private_data;
};

struct ArrowArray {
  // Array data description
  int64_t length;
  int64_t null_count;
  int64_t offset;
  int64_t n_buffers;
  int64_t n_children;
  const void **buffers;
  struct ArrowArray **children;
  struct ArrowArray *dictionary;

  // Release callback
  void (*release)(struct ArrowArray *);
  // Opaque producer-specific data
  void *private_data;
};

#endif // ARROW_C_DATA_INTERFACE

namespace snail {
namespace geoarrow {

namespace py = pybind11;

using linestr = std::vector<geometry::Coord>;

static const char *GEOARROW_LINESTRING = "geoarrow.linestring";

/// Read a value from an ArrowSchema metadata blob. The blob is a sequence
/// of length-prefixed key/value byte strings (int32 count, then per pair
/// int32 key length, key bytes, int32 value length, value bytes).
/// Returns an empty string if the key is not present.
static std::string metadata_value(const char *metadata, const std::string &key) {
  if (metadata == nullptr) {
    return "";
  }
  const char *pos = metadata;
  int32_t n_pairs;
  std::memcpy(&n_pairs, pos, sizeof(int32_t));
  pos += sizeof(int32_t);
  for (int32_t i = 0; i < n_pairs; i++) {
    int32_t key_length;
    std::memcpy(&key_length, pos, sizeof(int32_t));
    pos += sizeof(int32_t);
    std::string current_key(pos, key_length);
    pos += key_length;
    int32_t value_length;
    std::memcpy(&value_length, pos, sizeof(int32_t));
    pos += sizeof(int32_t);
    std::string value(pos, value_length);
    pos += value_length;
    if (current_key == key) {
      return value;
    }
  }
  return "";
}

/// Encode a single key/value pair as an ArrowSchema metadata blob.
static std::string encode_metadata(const std::string &key,
                                   const std::string &value) {
  std::string blob;
  auto append_int32 = [&blob](int32_t v) {
    blob.append(reinterpret_cast<const char *>(&v), sizeof(int32_t));
  };
  append_int32(1);
  append_int32(static_cast<int32_t>(key.size()));
  blob.append(key);
  append_int32(static_cast<int32_t>(value.size()));
  blob.append(value);
  return blob;
}

/// Check whether any slot in an ArrowArray is null.
static bool has_nulls(const ArrowArray *array) {
  if (array->null_count > 0) {
    return true;
  }
  if (array->null_count == 0 || array->n_buffers == 0) {
    return false;
  }
  // null_count is unknown (-1), inspect the validity bitmap if present
  const uint8_t *validity = static_cast<const uint8_t *>(array->buffers[0]);
  if (validity == nullptr) {
    return false;
  }
  for (int64_t i = 0; i < array->length; i++) {
    int64_t bit = array->offset + i;
    if (((validity[bit >> 3] >> (bit & 7)) & 1) == 0) {
      return true;
    }
  }
  return false;
}

/// Views onto the buffers of a GeoArrow interleaved linestring array
/// (list<vertices: fixed_size_list<xy: double>[2]>).
struct LineStringArrayView {
  int64_t num_geometries;
  // exactly one of these offset buffer pointers is set, depending on
  // whether the outer list has 32-bit or 64-bit offsets
  const int32_t *offsets32 = nullptr;
  const int64_t *offsets64 = nullptr;
  int64_t list_offset;    // slot offset of the outer list array
  int64_t vertex_offset;  // slot offset of the vertices fixed-size-list array
  int64_t double_offset;  // slot offset of the inner double array
  const double *coords;

  int64_t vertex_begin(int64_t geometry_index) const {
    int64_t slot = list_offset + geometry_index;
    return offsets32 ? offsets32[slot] : offsets64[slot];
  }

  int64_t vertex_end(int64_t geometry_index) const {
    return vertex_begin(geometry_index + 1);
  }

  geometry::Coord vertex(int64_t vertex_index) const {
    int64_t position = double_offset + 2 * (vertex_offset + vertex_index);
    return geometry::Coord(coords[position], coords[position + 1]);
  }
};

/// Validate an Arrow schema/array pair as a GeoArrow interleaved
/// linestring array and return a view onto its buffers.
static LineStringArrayView
parse_linestrings(const ArrowSchema *schema, const ArrowArray *array) {
  // check GeoArrow extension name if present (plain
  // list<fixed_size_list<double>[2]> arrays are also accepted)
  std::string extension_name =
      metadata_value(schema->metadata, "ARROW:extension:name");
  if (!extension_name.empty() && extension_name != GEOARROW_LINESTRING) {
    throw std::invalid_argument(
        "Expected a geoarrow.linestring array, got Arrow extension type '" +
        extension_name + "'");
  }

  LineStringArrayView view;

  // outer list array, with either 32-bit or 64-bit offsets
  bool large_list;
  if (std::strcmp(schema->format, "+l") == 0) {
    large_list = false;
  } else if (std::strcmp(schema->format, "+L") == 0) {
    large_list = true;
  } else {
    throw std::invalid_argument(
        std::string("Expected a GeoArrow linestring array (Arrow list type), "
                    "got Arrow format '") +
        schema->format + "'");
  }
  if (schema->n_children != 1 || array->n_children != 1) {
    throw std::invalid_argument(
        "Malformed Arrow list array: expected exactly one child");
  }

  // vertices child: interleaved coordinates as fixed_size_list<double>[2]
  const ArrowSchema *vertices_schema = schema->children[0];
  if (std::strcmp(vertices_schema->format, "+s") == 0) {
    throw std::invalid_argument(
        "GeoArrow linestring arrays with separated (struct) coordinates are "
        "not supported: pass interleaved coordinates, e.g. "
        "GeoSeries.to_arrow(geometry_encoding='geoarrow', interleaved=True)");
  }
  if (std::strcmp(vertices_schema->format, "+w:2") != 0) {
    throw std::invalid_argument(
        std::string("Expected linestring vertices as interleaved 2D "
                    "coordinates (Arrow format '+w:2'), got '") +
        vertices_schema->format + "'");
  }
  if (vertices_schema->n_children != 1 ||
      std::strcmp(vertices_schema->children[0]->format, "g") != 0) {
    throw std::invalid_argument(
        "Expected linestring coordinates of type double (Arrow format 'g')");
  }

  const ArrowArray *vertices = array->children[0];
  if (vertices->n_children != 1) {
    throw std::invalid_argument(
        "Malformed Arrow fixed-size-list array: expected exactly one child");
  }
  const ArrowArray *doubles = vertices->children[0];

  if (has_nulls(array) || has_nulls(vertices) || has_nulls(doubles)) {
    throw std::invalid_argument(
        "Cannot split missing (null) geometries: drop or fill null "
        "geometries first");
  }

  view.num_geometries = array->length;
  if (large_list) {
    view.offsets64 = static_cast<const int64_t *>(array->buffers[1]);
  } else {
    view.offsets32 = static_cast<const int32_t *>(array->buffers[1]);
  }
  view.list_offset = array->offset;
  view.vertex_offset = vertices->offset;
  view.double_offset = doubles->offset;
  view.coords = static_cast<const double *>(doubles->buffers[1]);
  if (view.num_geometries > 0 &&
      (array->buffers[1] == nullptr || view.coords == nullptr)) {
    throw std::invalid_argument(
        "Malformed Arrow array: missing offsets or coordinates buffer");
  }
  return view;
}

/// Owns the buffers of a GeoArrow linestring array built as output:
/// interleaved xy coordinates and int32 geometry offsets.
struct LineStringArrayData {
  std::vector<int32_t> offsets;
  std::vector<double> coords;

  int64_t num_geometries() const {
    return static_cast<int64_t>(offsets.size()) - 1;
  }
  int64_t num_vertices() const {
    return static_cast<int64_t>(coords.size()) / 2;
  }
};

// -- Export of LineStringArrayData through the Arrow C data interface -------

struct ExportedSchemaPrivate {
  std::string metadata;
  ArrowSchema vertices;
  ArrowSchema xy;
  ArrowSchema *top_children[1];
  ArrowSchema *vertices_children[1];
};

static void release_exported_child_schema(ArrowSchema *schema) {
  schema->release = nullptr;
}

static void release_exported_schema(ArrowSchema *schema) {
  auto *data = static_cast<ExportedSchemaPrivate *>(schema->private_data);
  for (int64_t i = 0; i < schema->n_children; i++) {
    ArrowSchema *child = schema->children[i];
    if (child->release != nullptr) {
      child->release(child);
    }
  }
  delete data;
  schema->release = nullptr;
}

/// Build an ArrowSchema describing a geoarrow.linestring array with
/// interleaved coordinates: list<vertices: fixed_size_list<xy: double>[2]>.
/// The caller owns the returned struct (release, then delete).
static ArrowSchema *export_linestring_schema() {
  auto *data = new ExportedSchemaPrivate();
  data->metadata =
      encode_metadata("ARROW:extension:name", GEOARROW_LINESTRING);

  data->xy.format = "g";
  data->xy.name = "xy";
  data->xy.metadata = nullptr;
  data->xy.flags = 0;
  data->xy.n_children = 0;
  data->xy.children = nullptr;
  data->xy.dictionary = nullptr;
  data->xy.release = release_exported_child_schema;
  data->xy.private_data = nullptr;

  data->vertices_children[0] = &data->xy;
  data->vertices.format = "+w:2";
  data->vertices.name = "vertices";
  data->vertices.metadata = nullptr;
  data->vertices.flags = 0;
  data->vertices.n_children = 1;
  data->vertices.children = data->vertices_children;
  data->vertices.dictionary = nullptr;
  data->vertices.release = release_exported_child_schema;
  data->vertices.private_data = nullptr;

  data->top_children[0] = &data->vertices;
  ArrowSchema *schema = new ArrowSchema();
  schema->format = "+l";
  schema->name = "";
  schema->metadata = data->metadata.data();
  schema->flags = ARROW_FLAG_NULLABLE;
  schema->n_children = 1;
  schema->children = data->top_children;
  schema->dictionary = nullptr;
  schema->release = release_exported_schema;
  schema->private_data = data;
  return schema;
}

struct ExportedArrayPrivate {
  std::shared_ptr<LineStringArrayData> data;
  ArrowArray vertices;
  ArrowArray xy;
  ArrowArray *top_children[1];
  ArrowArray *vertices_children[1];
  const void *top_buffers[2];
  const void *vertices_buffers[1];
  const void *xy_buffers[2];
};

static void release_exported_child_array(ArrowArray *array) {
  array->release = nullptr;
}

static void release_exported_array(ArrowArray *array) {
  auto *data = static_cast<ExportedArrayPrivate *>(array->private_data);
  for (int64_t i = 0; i < array->n_children; i++) {
    ArrowArray *child = array->children[i];
    if (child->release != nullptr) {
      child->release(child);
    }
  }
  delete data;
  array->release = nullptr;
}

/// Build an ArrowArray over shared linestring data, without copying the
/// coordinate or offset buffers. The caller owns the returned struct
/// (release, then delete); the underlying buffers are kept alive by the
/// shared_ptr held in private_data.
static ArrowArray *
export_linestring_array(std::shared_ptr<LineStringArrayData> shared) {
  auto *data = new ExportedArrayPrivate();
  data->data = shared;

  data->xy_buffers[0] = nullptr;
  data->xy_buffers[1] = shared->coords.data();
  data->xy.length = static_cast<int64_t>(shared->coords.size());
  data->xy.null_count = 0;
  data->xy.offset = 0;
  data->xy.n_buffers = 2;
  data->xy.n_children = 0;
  data->xy.buffers = data->xy_buffers;
  data->xy.children = nullptr;
  data->xy.dictionary = nullptr;
  data->xy.release = release_exported_child_array;
  data->xy.private_data = nullptr;

  data->vertices_buffers[0] = nullptr;
  data->vertices_children[0] = &data->xy;
  data->vertices.length = shared->num_vertices();
  data->vertices.null_count = 0;
  data->vertices.offset = 0;
  data->vertices.n_buffers = 1;
  data->vertices.n_children = 1;
  data->vertices.buffers = data->vertices_buffers;
  data->vertices.children = data->vertices_children;
  data->vertices.dictionary = nullptr;
  data->vertices.release = release_exported_child_array;
  data->vertices.private_data = nullptr;

  data->top_buffers[0] = nullptr;
  data->top_buffers[1] = shared->offsets.data();
  data->top_children[0] = &data->vertices;
  ArrowArray *array = new ArrowArray();
  array->length = shared->num_geometries();
  array->null_count = 0;
  array->offset = 0;
  array->n_buffers = 2;
  array->n_children = 1;
  array->buffers = data->top_buffers;
  array->children = data->top_children;
  array->dictionary = nullptr;
  array->release = release_exported_array;
  array->private_data = data;
  return array;
}

static void schema_capsule_destructor(PyObject *capsule) {
  auto *schema = static_cast<ArrowSchema *>(
      PyCapsule_GetPointer(capsule, "arrow_schema"));
  if (schema != nullptr) {
    if (schema->release != nullptr) {
      schema->release(schema);
    }
    delete schema;
  }
}

static void array_capsule_destructor(PyObject *capsule) {
  auto *array =
      static_cast<ArrowArray *>(PyCapsule_GetPointer(capsule, "arrow_array"));
  if (array != nullptr) {
    if (array->release != nullptr) {
      array->release(array);
    }
    delete array;
  }
}

/// A GeoArrow linestring array, exposing the Arrow PyCapsule interface so
/// it can be consumed zero-copy by pyarrow.array,
/// geopandas.GeoSeries.from_arrow, shapely, and other Arrow-aware
/// libraries.
class LineStringArray {
public:
  explicit LineStringArray(std::shared_ptr<LineStringArrayData> data)
      : data(data) {}

  int64_t size() const { return data->num_geometries(); }

  /// Arrow PyCapsule interface: return ("arrow_schema", "arrow_array")
  /// capsules. May be called any number of times; each export shares
  /// ownership of the underlying buffers, so nothing is copied.
  py::tuple arrow_c_array(py::object requested_schema) const {
    // requested_schema is accepted for protocol compatibility and ignored:
    // this array is always geoarrow.linestring with interleaved coordinates
    (void)requested_schema;
    ArrowSchema *schema = export_linestring_schema();
    ArrowArray *array = export_linestring_array(data);
    return py::make_tuple(
        py::capsule(schema, "arrow_schema", schema_capsule_destructor),
        py::capsule(array, "arrow_array", array_capsule_destructor));
  }

private:
  std::shared_ptr<LineStringArrayData> data;
};

// -- The split function ------------------------------------------------------

/// Split every linestring in a GeoArrow array along the gridlines of a
/// raster grid defined by (width, height, transform).
static py::tuple splitLineStrings(py::object linestrings, int width,
                                  int height, std::vector<double> transform) {
  // Import through the Arrow PyCapsule interface. The capsules stay alive
  // (and hold the buffers alive) for the duration of this function.
  py::tuple capsules = linestrings.attr("__arrow_c_array__")();
  py::capsule schema_capsule = capsules[0].cast<py::capsule>();
  py::capsule array_capsule = capsules[1].cast<py::capsule>();
  const ArrowSchema *schema = static_cast<const ArrowSchema *>(
      PyCapsule_GetPointer(schema_capsule.ptr(), "arrow_schema"));
  const ArrowArray *array = static_cast<const ArrowArray *>(
      PyCapsule_GetPointer(array_capsule.ptr(), "arrow_array"));
  if (schema == nullptr || array == nullptr) {
    throw py::error_already_set();
  }

  LineStringArrayView view = parse_linestrings(schema, array);

  transform::Affine affine(transform[0], transform[1], transform[2],
                           transform[3], transform[4], transform[5]);
  grid::Grid grid(width, height, affine);

  auto splits = std::make_shared<LineStringArrayData>();
  splits->offsets.push_back(0);
  std::vector<int64_t> source_index;

  {
    // pure buffer crunching from here on - release the GIL
    py::gil_scoped_release release;

    linestr coords;
    for (int64_t i = 0; i < view.num_geometries; i++) {
      int64_t begin = view.vertex_begin(i);
      int64_t end = view.vertex_end(i);

      coords.clear();
      for (int64_t v = begin; v < end; v++) {
        coords.push_back(view.vertex(v));
      }

      if (coords.size() < 2) {
        // empty linestring: pass through unsplit
        source_index.push_back(i);
        for (auto point : coords) {
          splits->coords.push_back(point.x);
          splits->coords.push_back(point.y);
        }
        splits->offsets.push_back(
            static_cast<int32_t>(splits->coords.size() / 2));
        continue;
      }

      geometry::LineString line(coords);
      std::vector<linestr> pieces =
          operations::findIntersectionsLineString(line, grid);
      for (const linestr &piece : pieces) {
        // drop zero-length pieces, sometimes returned on the edge of the
        // raster (see snail.intersection.split_linestrings)
        double length = 0.0;
        for (std::size_t p = 1; p < piece.size(); p++) {
          length += geometry::Line(piece[p - 1], piece[p]).length();
        }
        if (length == 0.0) {
          continue;
        }
        source_index.push_back(i);
        for (auto point : piece) {
          splits->coords.push_back(point.x);
          splits->coords.push_back(point.y);
        }
        splits->offsets.push_back(
            static_cast<int32_t>(splits->coords.size() / 2));
      }
    }
  }

  if (splits->coords.size() / 2 >
      static_cast<std::size_t>(std::numeric_limits<int32_t>::max())) {
    throw std::overflow_error(
        "Split geometries have too many vertices for a GeoArrow linestring "
        "array with 32-bit offsets: split the input into chunks");
  }

  py::array_t<int64_t> index(static_cast<py::ssize_t>(source_index.size()));
  std::copy(source_index.begin(), source_index.end(),
            index.mutable_data());

  return py::make_tuple(LineStringArray(splits), index);
}

void register_module(py::module_ &m) {
  py::class_<LineStringArray>(m, "LineStringArray",
                              "GeoArrow linestring array, consumable via "
                              "the Arrow PyCapsule interface (e.g. by "
                              "pyarrow.array or "
                              "geopandas.GeoSeries.from_arrow)")
      .def("__len__", &LineStringArray::size)
      .def("__arrow_c_array__", &LineStringArray::arrow_c_array,
           py::arg("requested_schema") = py::none());

  m.def("split_linestrings", &splitLineStrings, py::arg("linestrings"),
        py::arg("width"), py::arg("height"), py::arg("transform"),
        R"(Split an array of linestrings along a raster grid.

Parameters
----------
linestrings : object
    GeoArrow linestring array with interleaved coordinates, as any object
    implementing __arrow_c_array__ (for example the result of
    geopandas.GeoSeries.to_arrow(geometry_encoding="geoarrow")).
width, height : int
    Raster grid dimensions (number of columns, number of rows).
transform : list of float
    First six elements of the affine grid-to-world transform.

Returns
-------
(LineStringArray, numpy.ndarray)
    The split linestrings as a GeoArrow array, and an int64 array with
    the index of the parent linestring for each split.
)");
}

} // namespace geoarrow
} // namespace snail
