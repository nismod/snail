/// GeoArrow data exchange for the Python/C++ interface (see issue #14).
///
/// Geometries cross between Python and C++ as GeoArrow arrays: flat
/// coordinate and offset buffers behind the Arrow C data interface, as
/// produced by GeoSeries.to_arrow(geometry_encoding="geoarrow") from a
/// geopandas geometry column, and consumed by GeoSeries.from_arrow.
///
/// This is the same coordinates-and-offsets layout that operations::
/// LinePieces and PolygonPieces already use, so splitting reads its input
/// straight out of the Arrow buffers and hands its output back as Arrow
/// buffers, with no copy of the coordinates in either direction and no
/// geometry objects built per feature.
///
/// The Arrow C data interface is a stable C ABI, so nothing here links
/// against an Arrow library, and the extension does not call back into
/// shapely to convert geometries.
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
#include <utility>
#include <vector>

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "geoarrow.hpp"
#include "geometry.hpp"
#include "grid.hpp"
#include "operations.hpp"
#include "transform.hpp"

// Arrow C data interface structs - a frozen ABI, reproduced verbatim from
// the specification rather than depending on an Arrow library
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
namespace geo = geometry;

using linestr = std::vector<geo::Coord>;

static_assert(sizeof(geo::Coord) == 2 * sizeof(double),
              "Coord must be a bare pair of doubles to alias Arrow's "
              "interleaved coordinate buffer");

/// The GeoArrow geometry types handled here
enum class GeometryType { linestring, polygon };

static const char *extensionName(GeometryType type) {
  return type == GeometryType::polygon ? "geoarrow.polygon"
                                       : "geoarrow.linestring";
}

// -- Arrow schema metadata ---------------------------------------------------

/// Read a value out of an ArrowSchema metadata blob, which holds a count of
/// key/value pairs followed by each pair as a length-prefixed byte string.
/// Returns an empty string if the key is absent.
static std::string metadataValue(const char *metadata, const std::string &key) {
  if (metadata == nullptr) {
    return "";
  }
  const char *pos = metadata;
  int32_t n_pairs = 0;
  std::memcpy(&n_pairs, pos, sizeof(int32_t));
  pos += sizeof(int32_t);
  for (int32_t i = 0; i < n_pairs; i++) {
    int32_t key_length = 0;
    std::memcpy(&key_length, pos, sizeof(int32_t));
    pos += sizeof(int32_t);
    std::string current_key(pos, key_length);
    pos += key_length;
    int32_t value_length = 0;
    std::memcpy(&value_length, pos, sizeof(int32_t));
    pos += sizeof(int32_t);
    if (current_key == key) {
      return std::string(pos, value_length);
    }
    pos += value_length;
  }
  return "";
}

/// Encode a single key/value pair as an ArrowSchema metadata blob
static std::string encodeMetadata(const std::string &key,
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

// -- Reading GeoArrow arrays -------------------------------------------------

/// Does any slot of an array hold a null?
static bool hasNulls(const ArrowArray *array) {
  if (array->null_count > 0) {
    return true;
  }
  if (array->null_count == 0 || array->n_buffers == 0) {
    return false;
  }
  // null_count is unknown (-1): inspect the validity bitmap, if there is one
  const auto *validity = static_cast<const uint8_t *>(array->buffers[0]);
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

/// A list array's offsets, which may be 32- or 64-bit, read through the
/// array's own slot offset
class ListOffsets {
public:
  ListOffsets() = default;

  ListOffsets(const ArrowSchema *schema, const ArrowArray *array,
              const char *what) {
    if (std::strcmp(schema->format, "+l") == 0) {
      offsets32 = static_cast<const int32_t *>(array->buffers[1]);
    } else if (std::strcmp(schema->format, "+L") == 0) {
      offsets64 = static_cast<const int64_t *>(array->buffers[1]);
    } else {
      throw std::invalid_argument(std::string("Expected ") + what +
                                  " as an Arrow list array, got format '" +
                                  schema->format + "'");
    }
    if (array->length > 0 && offsets32 == nullptr && offsets64 == nullptr) {
      throw std::invalid_argument(std::string("Malformed Arrow array: ") +
                                  what + " has no offsets buffer");
    }
    slot = array->offset;
    length = array->length;
  }

  /// Start of element i, in the units of the child array
  int64_t begin(int64_t i) const {
    int64_t at = slot + i;
    return offsets32 != nullptr ? offsets32[at] : offsets64[at];
  }

  int64_t end(int64_t i) const { return begin(i + 1); }

  int64_t size() const { return length; }

private:
  const int32_t *offsets32 = nullptr;
  const int64_t *offsets64 = nullptr;
  int64_t slot = 0;
  int64_t length = 0;
};

/// Interleaved xy coordinates: a fixed_size_list<double>[2] over a double
/// array, read through both arrays' slot offsets
class Coordinates {
public:
  Coordinates() = default;

  Coordinates(const ArrowSchema *schema, const ArrowArray *array) {
    if (std::strcmp(schema->format, "+s") == 0) {
      throw std::invalid_argument(
          "GeoArrow arrays with separated (struct) coordinates are not "
          "supported: pass interleaved coordinates, as produced by "
          "GeoSeries.to_arrow(geometry_encoding='geoarrow', "
          "interleaved=True)");
    }
    if (std::strcmp(schema->format, "+w:2") != 0) {
      throw std::invalid_argument(
          std::string("Expected interleaved 2D coordinates (Arrow format "
                      "'+w:2'), got '") +
          schema->format + "'");
    }
    if (schema->n_children != 1 || array->n_children != 1) {
      throw std::invalid_argument("Malformed Arrow fixed-size-list array: "
                                  "expected exactly one child");
    }
    if (std::strcmp(schema->children[0]->format, "g") != 0) {
      throw std::invalid_argument(
          std::string("Expected coordinates of type double (Arrow format "
                      "'g'), got '") +
          schema->children[0]->format + "'");
    }
    const ArrowArray *doubles = array->children[0];
    if (hasNulls(array) || hasNulls(doubles)) {
      throw std::invalid_argument("Cannot split geometries with missing "
                                  "(null) coordinates");
    }
    xy = static_cast<const double *>(doubles->buffers[1]);
    // element v of the fixed-size list is the pair of doubles at
    // 2 * (list slot offset + v), shifted again by the child's own offset
    base = doubles->offset + 2 * array->offset;
    if (array->length > 0 && xy == nullptr) {
      throw std::invalid_argument(
          "Malformed Arrow array: no coordinates buffer");
    }
  }

  geo::Coord at(int64_t vertex) const {
    int64_t i = base + 2 * vertex;
    return {xy[i], xy[i + 1]};
  }

private:
  const double *xy = nullptr;
  int64_t base = 0;
};

/// Check the GeoArrow extension name, when the producer declares one. A
/// plain nested list array of the right shape is accepted too, so that
/// arrays built directly with pyarrow work.
static void checkExtensionName(const ArrowSchema *schema, GeometryType type) {
  std::string name = metadataValue(schema->metadata, "ARROW:extension:name");
  if (name.empty() || name == extensionName(type)) {
    return;
  }
  std::string message = std::string("Expected a ") + extensionName(type) +
                        " array, got Arrow extension type '" + name + "'";
  // multi-part geometries are the likely mistake, and the fix is the
  // caller's to make: point at it rather than at the Arrow type
  if (name.rfind("geoarrow.multi", 0) == 0) {
    message += "; merge or explode multi-part geometries before splitting";
  }
  throw std::invalid_argument(message);
}

/// A geoarrow.linestring array: list<vertices: fixed_size_list<xy: double>[2]>
struct LineStringReader {
  ListOffsets lines;
  Coordinates coordinates;

  LineStringReader(const ArrowSchema *schema, const ArrowArray *array) {
    checkExtensionName(schema, GeometryType::linestring);
    if (schema->n_children != 1 || array->n_children != 1) {
      throw std::invalid_argument(
          "Malformed Arrow list array: expected exactly one child");
    }
    if (hasNulls(array)) {
      throw std::invalid_argument(
          "Cannot split missing (null) geometries: drop or fill null "
          "geometries first");
    }
    lines = ListOffsets(schema, array, "linestrings");
    coordinates = Coordinates(schema->children[0], array->children[0]);
  }

  int64_t size() const { return lines.size(); }

  /// Read linestring i into the given buffer
  void read(int64_t i, linestr &out) const {
    out.clear();
    int64_t begin = lines.begin(i);
    int64_t end = lines.end(i);
    out.reserve(static_cast<std::size_t>(end - begin));
    for (int64_t v = begin; v < end; v++) {
      out.push_back(coordinates.at(v));
    }
  }
};

/// A geoarrow.polygon array:
/// list<rings: list<vertices: fixed_size_list<xy: double>[2]>>
struct PolygonReader {
  ListOffsets polygons;
  ListOffsets rings;
  Coordinates coordinates;
  int64_t ring_slot = 0;

  PolygonReader(const ArrowSchema *schema, const ArrowArray *array) {
    checkExtensionName(schema, GeometryType::polygon);
    if (schema->n_children != 1 || array->n_children != 1) {
      throw std::invalid_argument(
          "Malformed Arrow list array: expected exactly one child");
    }
    const ArrowSchema *rings_schema = schema->children[0];
    const ArrowArray *rings_array = array->children[0];
    if (rings_schema->n_children != 1 || rings_array->n_children != 1) {
      throw std::invalid_argument(
          "Expected a geoarrow.polygon array (a list of rings of "
          "coordinates)");
    }
    if (hasNulls(array) || hasNulls(rings_array)) {
      throw std::invalid_argument(
          "Cannot split missing (null) geometries or rings: drop or fill "
          "null geometries first");
    }
    polygons = ListOffsets(schema, array, "polygons");
    rings = ListOffsets(rings_schema, rings_array, "polygon rings");
    coordinates =
        Coordinates(rings_schema->children[0], rings_array->children[0]);
  }

  int64_t size() const { return polygons.size(); }

  /// Read the rings of polygon i into the given buffer, exterior first.
  /// Unlike a flat coordinate read, the ring structure is explicit in the
  /// offsets, so rings are recovered exactly rather than inferred from
  /// where the coordinates close back on themselves.
  void read(int64_t i, std::vector<linestr> &out) const {
    out.clear();
    for (int64_t r = polygons.begin(i); r < polygons.end(i); r++) {
      linestr ring;
      int64_t begin = rings.begin(r);
      int64_t end = rings.end(r);
      ring.reserve(static_cast<std::size_t>(end - begin));
      for (int64_t v = begin; v < end; v++) {
        ring.push_back(coordinates.at(v));
      }
      out.push_back(std::move(ring));
    }
  }
};

// -- Writing GeoArrow arrays -------------------------------------------------

/// The buffers of a GeoArrow array built as output. Coordinates are held in
/// the same std::vector<Coord> the split operations fill, which is exactly
/// Arrow's interleaved coordinate buffer, so exporting copies nothing but
/// the (much smaller) offset arrays.
struct GeoArrowData {
  GeometryType type = GeometryType::linestring;
  std::vector<geo::Coord> coordinates;
  /// offsets into coordinates: one run per linestring, or per polygon ring
  std::vector<int32_t> vertex_offsets{0};
  /// offsets into the rings, one run per polygon; unused for linestrings
  std::vector<int32_t> ring_offsets{0};

  int64_t size() const {
    return static_cast<int64_t>(type == GeometryType::polygon
                                    ? ring_offsets.size()
                                    : vertex_offsets.size()) -
           1;
  }
};

struct ExportedSchema {
  std::string metadata;
  // rings is used for polygons only, and left out of the chain otherwise
  ArrowSchema rings;
  ArrowSchema vertices;
  ArrowSchema xy;
  ArrowSchema *top_child[1];
  ArrowSchema *rings_child[1];
  ArrowSchema *vertices_child[1];
};

/// Children are all owned by the one ExportedSchema block, freed when the
/// top level is released; marking them released keeps the tree consistent
/// for a consumer that walks it
static void releaseChildSchema(ArrowSchema *schema) {
  for (int64_t i = 0; i < schema->n_children; i++) {
    ArrowSchema *child = schema->children[i];
    if (child->release != nullptr) {
      child->release(child);
    }
  }
  schema->release = nullptr;
}

static void releaseSchema(ArrowSchema *schema) {
  releaseChildSchema(schema);
  delete static_cast<ExportedSchema *>(schema->private_data);
}

/// Fill in one level of nesting
static void initSchema(ArrowSchema *schema, const char *format,
                       const char *name, ArrowSchema **children,
                       int64_t n_children) {
  schema->format = format;
  schema->name = name;
  schema->metadata = nullptr;
  schema->flags = 0;
  schema->n_children = n_children;
  schema->children = children;
  schema->dictionary = nullptr;
  schema->release = releaseChildSchema;
  schema->private_data = nullptr;
}

/// Build the ArrowSchema for a GeoArrow array with interleaved coordinates:
/// list<vertices: fixed_size_list<xy: double>[2]> for linestrings, or
/// list<rings: list<vertices: fixed_size_list<xy: double>[2]>> for polygons.
/// The caller owns the result (release it, then delete it).
static ArrowSchema *exportSchema(GeometryType type) {
  auto *owned = new ExportedSchema();
  owned->metadata =
      encodeMetadata("ARROW:extension:name", extensionName(type));

  initSchema(&owned->xy, "g", "xy", nullptr, 0);
  owned->vertices_child[0] = &owned->xy;
  initSchema(&owned->vertices, "+w:2", "vertices", owned->vertices_child, 1);

  auto *schema = new ArrowSchema();
  if (type == GeometryType::polygon) {
    owned->rings_child[0] = &owned->vertices;
    initSchema(&owned->rings, "+l", "rings", owned->rings_child, 1);
    owned->top_child[0] = &owned->rings;
  } else {
    owned->top_child[0] = &owned->vertices;
  }
  initSchema(schema, "+l", "", owned->top_child, 1);
  schema->metadata = owned->metadata.data();
  schema->flags = ARROW_FLAG_NULLABLE;
  schema->release = releaseSchema;
  schema->private_data = owned;
  return schema;
}

struct ExportedArray {
  std::shared_ptr<GeoArrowData> data;
  ArrowArray rings;
  ArrowArray vertices;
  ArrowArray xy;
  ArrowArray *top_child[1];
  ArrowArray *rings_child[1];
  ArrowArray *vertices_child[1];
  const void *top_buffers[2];
  const void *rings_buffers[2];
  const void *vertices_buffers[1];
  const void *xy_buffers[2];
};

/// As for the schema, the children live in the one ExportedArray block; the
/// buffers they point at are released with the shared data it holds
static void releaseChildArray(ArrowArray *array) {
  for (int64_t i = 0; i < array->n_children; i++) {
    ArrowArray *child = array->children[i];
    if (child->release != nullptr) {
      child->release(child);
    }
  }
  array->release = nullptr;
}

static void releaseArray(ArrowArray *array) {
  releaseChildArray(array);
  delete static_cast<ExportedArray *>(array->private_data);
}

/// Fill in one level of nesting
static void initArray(ArrowArray *array, int64_t length, const void **buffers,
                      int64_t n_buffers, ArrowArray **children,
                      int64_t n_children) {
  array->length = length;
  array->null_count = 0;
  array->offset = 0;
  array->n_buffers = n_buffers;
  array->n_children = n_children;
  array->buffers = buffers;
  array->children = children;
  array->dictionary = nullptr;
  array->release = releaseChildArray;
  array->private_data = nullptr;
}

/// Build an ArrowArray over shared split data, pointing at its buffers
/// rather than copying them. The caller owns the result (release it, then
/// delete it); the buffers stay alive with the shared_ptr held here, so an
/// array may be exported any number of times and outlive its producer.
static ArrowArray *exportArray(const std::shared_ptr<GeoArrowData> &data) {
  auto *owned = new ExportedArray();
  owned->data = data;

  owned->xy_buffers[0] = nullptr;
  owned->xy_buffers[1] = data->coordinates.data();
  initArray(&owned->xy, static_cast<int64_t>(2 * data->coordinates.size()),
            owned->xy_buffers, 2, nullptr, 0);

  owned->vertices_buffers[0] = nullptr;
  owned->vertices_child[0] = &owned->xy;
  initArray(&owned->vertices, static_cast<int64_t>(data->coordinates.size()),
            owned->vertices_buffers, 1, owned->vertices_child, 1);

  auto *array = new ArrowArray();
  if (data->type == GeometryType::polygon) {
    owned->rings_buffers[0] = nullptr;
    owned->rings_buffers[1] = data->vertex_offsets.data();
    owned->rings_child[0] = &owned->vertices;
    initArray(&owned->rings,
              static_cast<int64_t>(data->vertex_offsets.size()) - 1,
              owned->rings_buffers, 2, owned->rings_child, 1);

    owned->top_buffers[0] = nullptr;
    owned->top_buffers[1] = data->ring_offsets.data();
    owned->top_child[0] = &owned->rings;
  } else {
    owned->top_buffers[0] = nullptr;
    owned->top_buffers[1] = data->vertex_offsets.data();
    owned->top_child[0] = &owned->vertices;
  }
  initArray(array, data->size(), owned->top_buffers, 2, owned->top_child, 1);
  array->release = releaseArray;
  array->private_data = owned;
  return array;
}

static void releaseSchemaCapsule(PyObject *capsule) {
  auto *schema =
      static_cast<ArrowSchema *>(PyCapsule_GetPointer(capsule, "arrow_schema"));
  if (schema != nullptr) {
    if (schema->release != nullptr) {
      schema->release(schema);
    }
    delete schema;
  }
}

static void releaseArrayCapsule(PyObject *capsule) {
  auto *array =
      static_cast<ArrowArray *>(PyCapsule_GetPointer(capsule, "arrow_array"));
  if (array != nullptr) {
    if (array->release != nullptr) {
      array->release(array);
    }
    delete array;
  }
}

/// A GeoArrow array of split geometries, exposing the Arrow PyCapsule
/// interface so that geopandas, pyarrow, shapely and other Arrow-aware
/// libraries can read it without copying the coordinates.
class GeoArrowArray {
public:
  explicit GeoArrowArray(std::shared_ptr<GeoArrowData> data)
      : data(std::move(data)) {}

  int64_t size() const { return data->size(); }

  std::string geometryType() const {
    return extensionName(data->type);
  }

  /// Arrow PyCapsule interface: hand out ("arrow_schema", "arrow_array")
  /// capsules. May be called repeatedly; every export shares ownership of
  /// the same buffers.
  py::tuple arrowCArray(const py::object &requested_schema) const {
    // requested_schema is part of the protocol and ignored here: this array
    // is always GeoArrow with interleaved coordinates
    (void)requested_schema;
    ArrowSchema *schema = exportSchema(data->type);
    ArrowArray *array = exportArray(data);
    return py::make_tuple(
        py::capsule(schema, "arrow_schema", releaseSchemaCapsule),
        py::capsule(array, "arrow_array", releaseArrayCapsule));
  }

private:
  std::shared_ptr<GeoArrowData> data;
};

// -- Reading an argument through the Arrow PyCapsule interface ---------------

/// Holds the imported capsules, and so the input buffers, for as long as
/// the split needs them
class ImportedArray {
public:
  explicit ImportedArray(const py::object &geometries) {
    if (!py::hasattr(geometries, "__arrow_c_array__")) {
      throw py::type_error(
          "Expected a GeoArrow array: an object supporting the Arrow "
          "PyCapsule interface, as produced by "
          "GeoSeries.to_arrow(geometry_encoding='geoarrow')");
    }
    capsules = geometries.attr("__arrow_c_array__")();
    schema_capsule = capsules[0].cast<py::capsule>();
    array_capsule = capsules[1].cast<py::capsule>();
    schema_ptr = static_cast<const ArrowSchema *>(
        PyCapsule_GetPointer(schema_capsule.ptr(), "arrow_schema"));
    array_ptr = static_cast<const ArrowArray *>(
        PyCapsule_GetPointer(array_capsule.ptr(), "arrow_array"));
    if (schema_ptr == nullptr || array_ptr == nullptr) {
      throw py::error_already_set();
    }
  }

  const ArrowSchema *schema() const { return schema_ptr; }
  const ArrowArray *array() const { return array_ptr; }

private:
  py::tuple capsules;
  py::capsule schema_capsule;
  py::capsule array_capsule;
  const ArrowSchema *schema_ptr = nullptr;
  const ArrowArray *array_ptr = nullptr;
};

/// Copy the parent index out as a numpy array
static py::array_t<std::int64_t>
parentArray(const std::vector<std::int64_t> &parent) {
  py::array_t<std::int64_t> out(static_cast<py::ssize_t>(parent.size()));
  if (!parent.empty()) {
    std::memcpy(out.mutable_data(), parent.data(),
                parent.size() * sizeof(std::int64_t));
  }
  return out;
}

// -- The split functions -----------------------------------------------------

static py::tuple splitLineStrings(const py::object &linestrings, int nrows,
                                  int ncols, std::vector<double> transform,
                                  bool bounded) {
  ImportedArray input(linestrings);
  LineStringReader reader(input.schema(), input.array());

  transform::Affine affine(transform[0], transform[1], transform[2],
                           transform[3], transform[4], transform[5]);
  grid::Grid grid(ncols, nrows, affine);

  auto splits = std::make_shared<GeoArrowData>();
  splits->type = GeometryType::linestring;
  std::vector<std::int64_t> parent;

  {
    // reading and writing flat buffers from here on, so let Python run
    py::gil_scoped_release unlocked;

    linestr line;
    operations::LinePieces pieces;
    for (int64_t l = 0; l < reader.size(); l++) {
      reader.read(l, line);
      pieces = operations::splitLineStringGrid(line, grid, bounded);

      std::size_t base = splits->coordinates.size();
      splits->coordinates.insert(splits->coordinates.end(),
                                 pieces.coordinates.begin(),
                                 pieces.coordinates.end());
      for (std::size_t p = 1; p < pieces.offsets.size(); p++) {
        splits->vertex_offsets.push_back(
            static_cast<int32_t>(base + pieces.offsets[p]));
        parent.push_back(l);
      }
    }
  }

  if (splits->coordinates.size() >
      static_cast<std::size_t>(std::numeric_limits<int32_t>::max())) {
    throw std::overflow_error(
        "Too many coordinates for a GeoArrow array with 32-bit offsets: "
        "split the input in smaller chunks");
  }

  return py::make_tuple(GeoArrowArray(splits), parentArray(parent));
}

static py::tuple splitPolygons(const py::object &polygons, int nrows, int ncols,
                               std::vector<double> transform) {
  ImportedArray input(polygons);
  PolygonReader reader(input.schema(), input.array());

  transform::Affine affine(transform[0], transform[1], transform[2],
                           transform[3], transform[4], transform[5]);
  grid::Grid grid(ncols, nrows, affine);

  auto splits = std::make_shared<GeoArrowData>();
  splits->type = GeometryType::polygon;
  std::vector<std::int64_t> parent;

  {
    py::gil_scoped_release unlocked;

    std::vector<linestr> rings;
    for (int64_t p = 0; p < reader.size(); p++) {
      reader.read(p, rings);
      operations::PolygonPieces pieces =
          operations::splitPolygonGridPieces(rings, grid);

      // concatenate onto the combined result, shifting the offsets
      std::size_t coordinate_base = splits->coordinates.size();
      std::size_t ring_base = splits->vertex_offsets.size() - 1;
      splits->coordinates.insert(splits->coordinates.end(),
                                 pieces.coordinates.begin(),
                                 pieces.coordinates.end());
      for (std::size_t r = 1; r < pieces.ring_offsets.size(); r++) {
        splits->vertex_offsets.push_back(
            static_cast<int32_t>(coordinate_base + pieces.ring_offsets[r]));
      }
      for (std::size_t q = 1; q < pieces.polygon_offsets.size(); q++) {
        splits->ring_offsets.push_back(
            static_cast<int32_t>(ring_base + pieces.polygon_offsets[q]));
        parent.push_back(p);
      }
    }
  }

  if (splits->coordinates.size() >
      static_cast<std::size_t>(std::numeric_limits<int32_t>::max())) {
    throw std::overflow_error(
        "Too many coordinates for a GeoArrow array with 32-bit offsets: "
        "split the input in smaller chunks");
  }

  return py::make_tuple(GeoArrowArray(splits), parentArray(parent));
}

void register_module(py::module_ &m) {
  py::class_<GeoArrowArray>(
      m, "GeoArrowArray",
      "An array of geometries in GeoArrow encoding, readable through the "
      "Arrow PyCapsule interface, e.g. by geopandas.GeoSeries.from_arrow "
      "or pyarrow.array")
      .def("__len__", &GeoArrowArray::size)
      .def_property_readonly("geometry_type", &GeoArrowArray::geometryType,
                             "The GeoArrow extension name of the geometries, "
                             "e.g. 'geoarrow.linestring'")
      .def("__arrow_c_array__", &GeoArrowArray::arrowCArray,
           py::arg("requested_schema") = py::none());

  m.def("split_linestrings", &splitLineStrings, py::arg("linestrings"),
        py::arg("nrows"), py::arg("ncols"), py::arg("transform"),
        py::arg("bounded") = false,
        R"(Split LineStrings along a grid.

Takes a geoarrow.linestring array with interleaved coordinates - any
object supporting the Arrow PyCapsule interface, such as the result of
GeoSeries.to_arrow(geometry_encoding="geoarrow") - and returns the
LineString pieces as a GeoArrowArray, together with the index of the
linestring each piece came from.)");

  m.def("split_polygons", &splitPolygons, py::arg("polygons"),
        py::arg("nrows"), py::arg("ncols"), py::arg("transform"),
        R"(Split Polygons along a grid.

Takes a geoarrow.polygon array with interleaved coordinates - any object
supporting the Arrow PyCapsule interface, such as the result of
GeoSeries.to_arrow(geometry_encoding="geoarrow") - and returns the
Polygon pieces as a GeoArrowArray, together with the index of the polygon
each piece came from.)");
}

} // namespace geoarrow
} // namespace snail
