/// GeoArrow data exchange for the Python/C++ interface (see issue #14).
///
/// Geometries cross between Python and C++ as GeoArrow, over the Arrow C
/// stream interface: the extension consumes a stream of geometry batches
/// and produces a stream of the split pieces, pulling one batch from its
/// source only when the consumer asks for the next batch of results.
///
/// Streaming rather than taking a single array is what lets any Arrow
/// source be split: a pyarrow ChunkedArray, Table, or RecordBatchReader,
/// a GeoParquet or Dataset reader, a GeoDataFrame's own Arrow export -
/// none of which are a single contiguous array, and most of which expose
/// only __arrow_c_stream__. Splitting one batch at a time also means a
/// source larger than memory never has to be materialised. A source that
/// is a single array (a GeoSeries' Arrow export, a pyarrow Array) is
/// accepted too, read as a stream of one batch.
///
/// The coordinates-and-offsets layout GeoArrow uses is the layout
/// operations::LinePieces and PolygonPieces already fill, so the pieces are
/// handed back by pointing Arrow at those buffers rather than copying them
/// out, and no geometry object is built per feature on either side. Reading
/// a source still copies each geometry's coordinates into a buffer to split
/// it, but never the source as a whole.
///
/// The Arrow C data interface is a stable C ABI of a handful of structs, so
/// nothing here links against an Arrow library: the structs are reproduced
/// below and the conventions they carry - format strings, buffer order,
/// release callbacks - are spelled out where they are relied on.
///
/// References:
/// - https://arrow.apache.org/docs/format/CDataInterface.html
/// - https://arrow.apache.org/docs/format/CStreamInterface.html
/// - https://arrow.apache.org/docs/format/CDataInterface/PyCapsuleInterface.html
/// - https://geoarrow.org/format.html

#include <algorithm>
#include <cerrno>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "geoarrow.hpp"
#include "geometry.hpp"
#include "grid.hpp"
#include "operations.hpp"
#include "transform.hpp"

// Arrow C data and stream interface structs - a frozen ABI, reproduced
// verbatim from the specification rather than depending on an Arrow library
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

#ifndef ARROW_C_STREAM_INTERFACE
#define ARROW_C_STREAM_INTERFACE

struct ArrowArrayStream {
  // Callback to get the stream type. Returns 0 on success.
  int (*get_schema)(struct ArrowArrayStream *, struct ArrowSchema *out);
  // Callback to get the next array. Returns 0 on success; on end of stream
  // the returned array is marked released.
  int (*get_next)(struct ArrowArrayStream *, struct ArrowArray *out);
  // Callback to get further details of the last error
  const char *(*get_last_error)(struct ArrowArrayStream *);

  // Release callback
  void (*release)(struct ArrowArrayStream *);
  // Opaque producer-specific data
  void *private_data;
};

#endif // ARROW_C_STREAM_INTERFACE

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

/// Read a value out of an ArrowSchema metadata blob. GeoArrow declares a
/// geometry type there, under "ARROW:extension:name".
///
/// The blob is packed, not a string: an int32 count of key/value pairs,
/// then for each pair an int32 length and that many bytes for the key,
/// and the same again for the value. Lengths are explicit, so neither keys
/// nor values are null-terminated. Returns an empty string if the key is
/// absent.
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

/// Encode a single key/value pair as an ArrowSchema metadata blob, in the
/// packed layout metadataValue describes
static std::string encodeMetadata(const std::string &key,
                                  const std::string &value) {
  std::string blob;
  auto append_int32 = [&blob](int32_t v) {
    blob.append(reinterpret_cast<const char *>(&v), sizeof(int32_t));
  };
  append_int32(1); // one pair follows
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
  // A producer may leave null_count at -1 rather than count them, so fall
  // back to the validity bitmap, which is always buffer 0 where an array
  // has one at all. It holds one bit per slot, least significant bit
  // first, set for a value that is present.
  const auto *validity = static_cast<const uint8_t *>(array->buffers[0]);
  if (validity == nullptr) {
    // no bitmap at all: every slot is valid
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

/// The offsets of an Arrow list array: for element i, where its run of
/// children begins and ends.
///
/// Offsets are buffer 1, behind the validity bitmap, and are 32-bit for a
/// list ("+l") or 64-bit for a large list ("+L"). They index the child
/// array's own elements, so they are read without regard to the child's
/// offset - but a sliced array carries a slot offset of its own, which
/// shifts where in the offsets buffer element i is described.
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

/// GeoArrow coordinates, in either of the layouts the format allows:
/// interleaved as a fixed_size_list<double>[2], which is how geopandas
/// hands over a geometry column, or separated as a struct of an x and a y
/// array, which is how GeoParquet stores them. Any further dimensions (z,
/// m) are ignored - splitting is planar.
class Coordinates {
public:
  Coordinates() = default;

  Coordinates(const ArrowSchema *schema, const ArrowArray *array) {
    checkSchema(schema);
    if (hasNulls(array)) {
      throw std::invalid_argument("Cannot split geometries with missing "
                                  "(null) coordinates");
    }
    stride = interleavedStride(schema);
    if (stride > 0) {
      const ArrowArray *values = array->children[0];
      if (hasNulls(values)) {
        throw std::invalid_argument("Cannot split geometries with missing "
                                    "(null) coordinates");
      }
      // A fixed-size list of width `stride` lays its elements end to end in
      // one child buffer, so vertex v starts at stride * v - shifted by the
      // list's own slot offset, and again by the child's.
      xy = static_cast<const double *>(values->buffers[1]);
      xy_base = values->offset + stride * array->offset;
      if (array->length > 0 && xy == nullptr) {
        throw std::invalid_argument(
            "Malformed Arrow array: no coordinates buffer");
      }
      return;
    }
    // Separated: one child array per dimension, each with its own offset
    const ArrowArray *x_array = array->children[dimension(schema, "x", 0)];
    const ArrowArray *y_array = array->children[dimension(schema, "y", 1)];
    if (hasNulls(x_array) || hasNulls(y_array)) {
      throw std::invalid_argument("Cannot split geometries with missing "
                                  "(null) coordinates");
    }
    x = static_cast<const double *>(x_array->buffers[1]);
    y = static_cast<const double *>(y_array->buffers[1]);
    x_base = x_array->offset + array->offset;
    y_base = y_array->offset + array->offset;
    if (array->length > 0 && (x == nullptr || y == nullptr)) {
      throw std::invalid_argument(
          "Malformed Arrow array: no coordinates buffer");
    }
  }

  /// Validate the coordinate layout without needing any data
  static void checkSchema(const ArrowSchema *schema) {
    int64_t width = interleavedStride(schema);
    if (width > 0) {
      if (width < 2) {
        throw std::invalid_argument(
            std::string("Expected interleaved coordinates of at least two "
                        "dimensions, got Arrow format '") +
            schema->format + "'");
      }
      if (schema->n_children != 1) {
        throw std::invalid_argument("Malformed Arrow fixed-size-list array: "
                                    "expected exactly one child");
      }
      checkDouble(schema->children[0]);
      return;
    }
    if (std::strcmp(schema->format, "+s") == 0) {
      checkDouble(schema->children[dimension(schema, "x", 0)]);
      checkDouble(schema->children[dimension(schema, "y", 1)]);
      return;
    }
    throw std::invalid_argument(
        std::string("Expected GeoArrow coordinates, either interleaved "
                    "(Arrow format '+w:2') or separated (a struct of x and "
                    "y), got '") +
        schema->format + "'");
  }

  geo::Coord at(int64_t vertex) const {
    if (xy != nullptr) {
      int64_t at = xy_base + stride * vertex;
      return {xy[at], xy[at + 1]};
    }
    return {x[x_base + vertex], y[y_base + vertex]};
  }

private:
  /// Doubles per vertex in an interleaved layout, read out of Arrow's
  /// fixed-size-list format string "+w:<width>": 2 for xy, 3 for xyz or
  /// xym, 4 for xyzm. Returns 0 for any other layout. Only x and y are
  /// read - splitting is planar - but every dimension counts towards the
  /// step from one vertex to the next.
  static int64_t interleavedStride(const ArrowSchema *schema) {
    if (std::strncmp(schema->format, "+w:", 3) != 0) {
      return 0;
    }
    return std::strtoll(schema->format + 3, nullptr, 10);
  }

  /// Index of a named dimension among a struct's children, by name where
  /// the producer gives one, else by position
  static int64_t dimension(const ArrowSchema *schema, const char *name,
                           int64_t fallback) {
    for (int64_t i = 0; i < schema->n_children; i++) {
      const char *child = schema->children[i]->name;
      if (child != nullptr && std::strcmp(child, name) == 0) {
        return i;
      }
    }
    if (fallback < schema->n_children) {
      return fallback;
    }
    throw std::invalid_argument(
        std::string("Expected a '") + name +
        "' coordinate among the separated GeoArrow coordinates");
  }

  /// "g" is Arrow's format string for a double
  static void checkDouble(const ArrowSchema *schema) {
    if (std::strcmp(schema->format, "g") != 0) {
      throw std::invalid_argument(
          std::string("Expected coordinates of type double (Arrow format "
                      "'g'), got '") +
          schema->format + "'");
    }
  }

  const double *xy = nullptr; // interleaved, stride doubles per vertex
  int64_t stride = 0;
  int64_t xy_base = 0;
  const double *x = nullptr; // separated, one array per dimension
  const double *y = nullptr;
  int64_t x_base = 0;
  int64_t y_base = 0;
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

/// Validate that a schema describes the expected GeoArrow geometry type,
/// before any data has arrived
static void checkGeometrySchema(const ArrowSchema *schema, GeometryType type) {
  checkExtensionName(schema, type);
  if (std::strcmp(schema->format, "+l") != 0 &&
      std::strcmp(schema->format, "+L") != 0) {
    throw std::invalid_argument(
        std::string("Expected a ") + extensionName(type) +
        " array (an Arrow list), got format '" + schema->format + "'");
  }
  if (schema->n_children != 1) {
    throw std::invalid_argument(
        "Malformed Arrow list array: expected exactly one child");
  }
  if (type == GeometryType::polygon) {
    const ArrowSchema *rings = schema->children[0];
    if (std::strcmp(rings->format, "+l") != 0 &&
        std::strcmp(rings->format, "+L") != 0) {
      throw std::invalid_argument(
          "Expected a geoarrow.polygon array (a list of rings of "
          "coordinates)");
    }
    if (rings->n_children != 1) {
      throw std::invalid_argument(
          "Malformed Arrow list array: expected exactly one child");
    }
    Coordinates::checkSchema(rings->children[0]);
  } else {
    Coordinates::checkSchema(schema->children[0]);
  }
}

/// A geoarrow.linestring batch: a list of coordinates per linestring,
/// list<vertices: coordinates> in either coordinate layout
struct LineStringReader {
  ListOffsets lines;
  Coordinates coordinates;

  LineStringReader(const ArrowSchema *schema, const ArrowArray *array) {
    if (array->n_children != 1) {
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

/// A geoarrow.polygon batch: a list of rings per polygon, each ring a list
/// of coordinates - list<rings: list<vertices: coordinates>>
struct PolygonReader {
  ListOffsets polygons;
  ListOffsets rings;
  Coordinates coordinates;

  PolygonReader(const ArrowSchema *schema, const ArrowArray *array) {
    if (array->n_children != 1) {
      throw std::invalid_argument(
          "Malformed Arrow list array: expected exactly one child");
    }
    const ArrowSchema *rings_schema = schema->children[0];
    const ArrowArray *rings_array = array->children[0];
    if (rings_array->n_children != 1) {
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
  /// The ring structure is explicit in the offsets, so rings are recovered
  /// exactly rather than inferred from where coordinates close back on
  /// themselves.
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

// -- Reading the source ------------------------------------------------------

/// An Arrow C struct we own - a schema, an array or a stream.
///
/// Every one of them carries its own release callback, and a null callback
/// is how the interface marks a struct that has been released or moved
/// from. Holding them in a scope guard means a throw part-way through
/// setting a stream up cannot strand what the producer handed over.
template <typename T> class Owned {
public:
  Owned() = default;
  ~Owned() { reset(); }
  Owned(const Owned &) = delete;
  Owned &operator=(const Owned &) = delete;

  void reset() {
    if (owned.release != nullptr) {
      owned.release(&owned);
      owned.release = nullptr;
    }
  }

  bool valid() const { return owned.release != nullptr; }
  T *get() { return &owned; }

  /// Take on a struct whose previous owner has given it up
  void adopt(const T &from) {
    reset();
    owned = from;
  }

  /// Give the struct up to the caller, who becomes responsible for
  /// releasing it, and leave this empty
  T take() {
    T moved = owned;
    owned.release = nullptr;
    return moved;
  }

private:
  T owned{};
};

/// Take ownership of the struct a PyCapsule holds. The PyCapsule interface
/// passes Arrow structs by move: the consumer copies the struct out and
/// nulls the producer's release callback, so that only one of them will
/// ever release it.
template <typename T>
static T movedFromCapsule(const py::capsule &capsule, const char *name) {
  auto *source = static_cast<T *>(PyCapsule_GetPointer(capsule.ptr(), name));
  if (source == nullptr) {
    throw py::error_already_set();
  }
  if (source->release == nullptr) {
    throw std::invalid_argument(std::string("The ") + name +
                                " has already been consumed");
  }
  T moved = *source;
  source->release = nullptr;
  return moved;
}

/// The stream of geometry batches to split.
///
/// Reads any source implementing the Arrow PyCapsule interface: a stream
/// (__arrow_c_stream__) of geometry batches or of record batches carrying a
/// geometry column, or a single array (__arrow_c_array__) read as one
/// batch.
class InputStream {
public:
  InputStream(const py::object &source, GeometryType type) {
    if (py::hasattr(source, "__arrow_c_stream__")) {
      py::capsule capsule =
          source.attr("__arrow_c_stream__")().cast<py::capsule>();
      stream.adopt(
          movedFromCapsule<ArrowArrayStream>(capsule, "arrow_array_stream"));
      // a stream states its type up front, before any batch arrives
      if (stream.get()->get_schema(stream.get(), schema.get()) != 0) {
        throw std::runtime_error(std::string("Could not read the schema of "
                                             "the geometry stream: ") +
                                 lastError());
      }
    } else if (py::hasattr(source, "__arrow_c_array__")) {
      // a single array comes with its schema alongside, and stands in for a
      // stream of one batch
      py::tuple capsules = source.attr("__arrow_c_array__")();
      schema.adopt(movedFromCapsule<ArrowSchema>(
          capsules[0].cast<py::capsule>(), "arrow_schema"));
      single.adopt(movedFromCapsule<ArrowArray>(
          capsules[1].cast<py::capsule>(), "arrow_array"));
    } else {
      throw py::type_error(
          "Expected GeoArrow geometries: an object supporting the Arrow "
          "PyCapsule interface, such as the result of "
          "GeoSeries.to_arrow(geometry_encoding='geoarrow'), a pyarrow "
          "ChunkedArray or Table, or any Arrow stream of geometries");
    }

    // "+s" is Arrow's format string for a struct, which is how a stream of
    // record batches describes itself: the geometries are one of its
    // columns. Anything else is a stream of the geometries themselves.
    if (std::strcmp(schema.get()->format, "+s") == 0) {
      geometry_child = findGeometryField(schema.get(), type);
      checkGeometrySchema(schema.get()->children[geometry_child], type);
    } else {
      checkGeometrySchema(schema.get(), type);
    }
  }

  /// The schema of the geometries themselves, within the batches
  const ArrowSchema *geometrySchema() {
    return geometry_child >= 0 ? schema.get()->children[geometry_child]
                               : schema.get();
  }

  /// The geometries within a batch this stream produced
  const ArrowArray *geometryArray(ArrowArray *batch) const {
    if (geometry_child < 0) {
      return batch;
    }
    if (geometry_child >= batch->n_children) {
      throw std::runtime_error("Arrow batch does not match the stream schema");
    }
    return batch->children[geometry_child];
  }

  /// Pull the next batch. Returns false once the source is exhausted.
  bool next(Owned<ArrowArray> &batch) {
    batch.reset();
    if (single.valid()) {
      // a one-batch source: hand the array over, emptying it so that the
      // next call reports the end of the stream
      batch.adopt(single.take());
      return true;
    }
    if (!stream.valid()) {
      return false;
    }
    if (stream.get()->get_next(stream.get(), batch.get()) != 0) {
      throw std::runtime_error(
          std::string("Could not read the next batch of geometries: ") +
          lastError());
    }
    // the producer marks the end of the stream with a released array
    return batch.valid();
  }

private:
  /// Pick the geometry column out of a record batch schema: the one
  /// declaring a GeoArrow extension type, or the only column there is
  static int64_t findGeometryField(const ArrowSchema *struct_schema,
                                   GeometryType type) {
    int64_t found = -1;
    for (int64_t i = 0; i < struct_schema->n_children; i++) {
      std::string name = metadataValue(struct_schema->children[i]->metadata,
                                       "ARROW:extension:name");
      if (name.rfind("geoarrow.", 0) == 0) {
        if (found >= 0) {
          throw std::invalid_argument(
              "Expected one geometry column, found several: select the "
              "column to split");
        }
        found = i;
      }
    }
    if (found >= 0) {
      return found;
    }
    if (struct_schema->n_children == 1) {
      return 0;
    }
    throw std::invalid_argument(
        std::string("Found no ") + extensionName(type) +
        " column in the Arrow stream: none of its columns declares a "
        "GeoArrow extension type");
  }

  /// A stream reports a failure by returning non-zero and leaving the
  /// detail behind for get_last_error
  const char *lastError() {
    if (!stream.valid() || stream.get()->get_last_error == nullptr) {
      return "";
    }
    const char *message = stream.get()->get_last_error(stream.get());
    return message == nullptr ? "" : message;
  }

  Owned<ArrowArrayStream> stream;
  Owned<ArrowArray> single;
  Owned<ArrowSchema> schema;
  /// which column of a record batch holds the geometries; -1 when the
  /// batches are the geometries themselves
  int64_t geometry_child = -1;
};

// -- Writing batches of split pieces -----------------------------------------

/// One batch of split pieces: the geometries in GeoArrow's flat layout,
/// and for each piece the index of the geometry it was split from.
struct BatchData {
  GeometryType type = GeometryType::linestring;
  std::vector<geo::Coord> coordinates;
  /// where each run of coordinates begins: one run per piece for
  /// linestrings, one per ring for polygons. Offsets are 32-bit, which is
  /// what a plain Arrow list takes, and are relative to this batch.
  std::vector<int32_t> vertex_offsets{0};
  /// where each polygon's run of rings begins; unused for linestrings
  std::vector<int32_t> ring_offsets{0};
  /// the geometry each piece came from, indexed across the whole stream
  std::vector<int64_t> parents;

  int64_t size() const { return static_cast<int64_t>(parents.size()); }
};

/// Backing store for one exported schema. Arrow's structs hold raw
/// pointers to their children and to the arrays listing them, so every
/// level of the tree is kept in one allocation that outlives the export
/// and is freed when the top level is released.
struct ExportedSchema {
  std::string metadata;
  ArrowSchema geometry;
  ArrowSchema rings; // polygons only
  ArrowSchema vertices;
  ArrowSchema xy;
  ArrowSchema parent;
  ArrowSchema *top_children[2];
  ArrowSchema *geometry_child[1];
  ArrowSchema *rings_child[1];
  ArrowSchema *vertices_child[1];
};

/// Releasing a struct means releasing its children and then nulling its own
/// callback, which is how the interface marks it spent. The children here
/// own nothing of their own - they are freed with the block - so this just
/// walks the tree marking it, for a consumer that inspects it afterwards.
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

/// Build the schema of the split stream: record batches of a GeoArrow
/// geometry column and the index of the geometry each piece came from.
///
/// Arrow describes a type by a format string, and nests them by children:
///
///     struct                                             "+s"
///      |- geometry  ARROW:extension:name=geoarrow.*       "+l"  list
///      |   |- (rings, polygons only)                      "+l"  list
///      |       |- vertices                              "+w:2"  2 per item
///      |           |- xy                                  "g"   double
///      |- parent                                          "l"   int64
///
/// Coordinates go back interleaved because that is what the split fills:
/// a vector of Coord is already a run of x, y, x, y doubles.
static void exportSchema(GeometryType type, ArrowSchema *out) {
  auto *owned = new ExportedSchema();
  owned->metadata = encodeMetadata("ARROW:extension:name", extensionName(type));

  initSchema(&owned->xy, "g", "xy", nullptr, 0);
  owned->vertices_child[0] = &owned->xy;
  initSchema(&owned->vertices, "+w:2", "vertices", owned->vertices_child, 1);

  if (type == GeometryType::polygon) {
    owned->rings_child[0] = &owned->vertices;
    initSchema(&owned->rings, "+l", "rings", owned->rings_child, 1);
    owned->geometry_child[0] = &owned->rings;
  } else {
    owned->geometry_child[0] = &owned->vertices;
  }
  initSchema(&owned->geometry, "+l", "geometry", owned->geometry_child, 1);
  // the extension name rides on the geometry field, which is where a
  // GeoArrow reader looks for it
  owned->geometry.metadata = owned->metadata.data();

  initSchema(&owned->parent, "l", "parent", nullptr, 0);

  owned->top_children[0] = &owned->geometry;
  owned->top_children[1] = &owned->parent;
  initSchema(out, "+s", "", owned->top_children, 2);
  out->flags = 0;
  out->release = releaseSchema;
  out->private_data = owned;
}

struct ExportedArray {
  std::shared_ptr<BatchData> data;
  ArrowArray geometry;
  ArrowArray rings; // polygons only
  ArrowArray vertices;
  ArrowArray xy;
  ArrowArray parent;
  ArrowArray *top_children[2];
  ArrowArray *geometry_child[1];
  ArrowArray *rings_child[1];
  ArrowArray *vertices_child[1];
  const void *top_buffers[1];
  const void *geometry_buffers[2];
  const void *rings_buffers[2];
  const void *vertices_buffers[1];
  const void *xy_buffers[2];
  const void *parent_buffers[2];
};

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

/// Build a record batch over a batch of split pieces, mirroring the tree
/// exportSchema describes and pointing Arrow at the split's own buffers
/// rather than copying them out. The buffers stay alive with the shared
/// data held in private_data, so the batch outlives the split that made it.
///
/// Buffer 0 is always the validity bitmap, left null here because nothing
/// a split produces is null. What follows depends on the type: a list has
/// its offsets in buffer 1, a primitive its values, and a struct or a
/// fixed-size list has no second buffer at all - its children hold
/// everything.
static void exportArray(const std::shared_ptr<BatchData> &data,
                        ArrowArray *out) {
  auto *owned = new ExportedArray();
  owned->data = data;

  // the coordinates as bare doubles: two per vertex, hence the length
  owned->xy_buffers[0] = nullptr;
  owned->xy_buffers[1] = data->coordinates.data();
  initArray(&owned->xy, static_cast<int64_t>(2 * data->coordinates.size()),
            owned->xy_buffers, 2, nullptr, 0);

  // pairs them up into vertices; a fixed-size list needs no offsets
  owned->vertices_buffers[0] = nullptr;
  owned->vertices_child[0] = &owned->xy;
  initArray(&owned->vertices, static_cast<int64_t>(data->coordinates.size()),
            owned->vertices_buffers, 1, owned->vertices_child, 1);

  if (data->type == GeometryType::polygon) {
    // a polygon is a list of rings, so the offsets nest twice: rings over
    // vertices, then polygons over rings
    owned->rings_buffers[0] = nullptr;
    owned->rings_buffers[1] = data->vertex_offsets.data();
    owned->rings_child[0] = &owned->vertices;
    initArray(&owned->rings,
              static_cast<int64_t>(data->vertex_offsets.size()) - 1,
              owned->rings_buffers, 2, owned->rings_child, 1);
    owned->geometry_buffers[1] = data->ring_offsets.data();
    owned->geometry_child[0] = &owned->rings;
  } else {
    // a linestring is a run of vertices, so one level of offsets does
    owned->geometry_buffers[1] = data->vertex_offsets.data();
    owned->geometry_child[0] = &owned->vertices;
  }
  owned->geometry_buffers[0] = nullptr;
  initArray(&owned->geometry, data->size(), owned->geometry_buffers, 2,
            owned->geometry_child, 1);

  owned->parent_buffers[0] = nullptr;
  owned->parent_buffers[1] = data->parents.data();
  initArray(&owned->parent, data->size(), owned->parent_buffers, 2, nullptr, 0);

  // the record batch itself: a struct of the two columns, whose own length
  // is the number of pieces
  owned->top_buffers[0] = nullptr;
  owned->top_children[0] = &owned->geometry;
  owned->top_children[1] = &owned->parent;
  initArray(out, data->size(), owned->top_buffers, 1, owned->top_children, 2);
  out->release = releaseArray;
  out->private_data = owned;
}

// -- Splitting a stream ------------------------------------------------------

/// Everything the split stream needs to answer its next call
struct SplitState {
  SplitState(const py::object &source, GeometryType type, const grid::Grid &grid,
             bool bounded)
      : input(source, type), grid(grid), type(type), bounded(bounded) {}

  InputStream input;
  grid::Grid grid;
  GeometryType type;
  bool bounded;
  /// how many geometries the stream has read, so that a piece's parent
  /// indexes the source as a whole rather than the batch it came from
  int64_t parent_base = 0;
  std::string error;
};

/// Split one batch of linestrings into pieces
static void splitLineStringBatch(const LineStringReader &reader,
                                 const grid::Grid &grid, bool bounded,
                                 int64_t parent_base, BatchData &out) {
  linestr line;
  for (int64_t l = 0; l < reader.size(); l++) {
    reader.read(l, line);
    operations::LinePieces pieces =
        operations::splitLineStringGrid(line, grid, bounded);

    std::size_t base = out.coordinates.size();
    out.coordinates.insert(out.coordinates.end(), pieces.coordinates.begin(),
                           pieces.coordinates.end());
    for (std::size_t p = 1; p < pieces.offsets.size(); p++) {
      out.vertex_offsets.push_back(
          static_cast<int32_t>(base + pieces.offsets[p]));
      out.parents.push_back(parent_base + l);
    }
  }
}

/// Split one batch of polygons into pieces
static void splitPolygonBatch(const PolygonReader &reader,
                              const grid::Grid &grid, int64_t parent_base,
                              BatchData &out) {
  std::vector<linestr> rings;
  for (int64_t p = 0; p < reader.size(); p++) {
    reader.read(p, rings);
    operations::PolygonPieces pieces =
        operations::splitPolygonGridPieces(rings, grid);

    // concatenate onto the batch, shifting the offsets
    std::size_t coordinate_base = out.coordinates.size();
    std::size_t ring_base = out.vertex_offsets.size() - 1;
    out.coordinates.insert(out.coordinates.end(), pieces.coordinates.begin(),
                           pieces.coordinates.end());
    for (std::size_t r = 1; r < pieces.ring_offsets.size(); r++) {
      out.vertex_offsets.push_back(
          static_cast<int32_t>(coordinate_base + pieces.ring_offsets[r]));
    }
    for (std::size_t q = 1; q < pieces.polygon_offsets.size(); q++) {
      out.ring_offsets.push_back(
          static_cast<int32_t>(ring_base + pieces.polygon_offsets[q]));
      out.parents.push_back(parent_base + p);
    }
  }
}

/// Read one batch from the source and split it, returning nothing once the
/// source is exhausted.
///
/// A batch can split to no pieces at all - every geometry outside the grid
/// with bounded splitting, say - and the loop reads on rather than emitting
/// it, both to spare consumers a batch with nothing in it and because
/// geopandas cannot read a zero-length GeoArrow array.
static std::shared_ptr<BatchData> nextSplitBatch(SplitState *state) {
  while (true) {
    Owned<ArrowArray> batch;
    if (!state->input.next(batch)) {
      return nullptr;
    }
    const ArrowArray *geometries = state->input.geometryArray(batch.get());
    const ArrowSchema *schema = state->input.geometrySchema();

    auto out = std::make_shared<BatchData>();
    out->type = state->type;
    int64_t count = 0;
    if (state->type == GeometryType::polygon) {
      PolygonReader reader(schema, geometries);
      count = reader.size();
      py::gil_scoped_release unlocked;
      splitPolygonBatch(reader, state->grid, state->parent_base, *out);
    } else {
      LineStringReader reader(schema, geometries);
      count = reader.size();
      py::gil_scoped_release unlocked;
      splitLineStringBatch(reader, state->grid, state->bounded,
                           state->parent_base, *out);
    }
    state->parent_base += count;

    if (out->coordinates.size() >
        static_cast<std::size_t>(std::numeric_limits<int32_t>::max())) {
      throw std::overflow_error(
          "One batch split to more coordinates than a GeoArrow array with "
          "32-bit offsets can hold: read the source in smaller batches");
    }
    if (out->size() > 0) {
      return out;
    }
  }
}

// -- The split stream, as an Arrow C stream ----------------------------------

/// private_data of the ArrowArrayStream we hand out
struct StreamPrivate {
  std::unique_ptr<SplitState> state;
};

static SplitState *stateOf(ArrowArrayStream *stream) {
  return static_cast<StreamPrivate *>(stream->private_data)->state.get();
}

/// Record a failure against the stream and return the code its callback
/// should report.
///
/// Stream callbacks are plain C, so a C++ exception must not escape them:
/// they signal failure by returning a non-zero errno and leaving the
/// detail for get_last_error. The consumer turns that code back into an
/// exception, and picking it deliberately keeps the type a caller sees the
/// same either side of the stream starting - EINVAL surfaces as a
/// ValueError, as a bad argument caught up front would have.
static int streamFailed(SplitState *state, const std::exception &error,
                        int code) {
  state->error = error.what();
  return code;
}

static int streamGetSchema(ArrowArrayStream *stream, ArrowSchema *out) {
  SplitState *state = stateOf(stream);
  try {
    exportSchema(state->type, out);
  } catch (const std::exception &error) {
    return streamFailed(state, error, EIO);
  }
  return 0;
}

static int streamGetNext(ArrowArrayStream *stream, ArrowArray *out) {
  SplitState *state = stateOf(stream);
  // The source and the geometries it holds may both be Python objects, and
  // the consumer may call us without the GIL, so take it for the read and
  // give it back around the splitting itself.
  py::gil_scoped_acquire locked;
  try {
    std::shared_ptr<BatchData> batch = nextSplitBatch(state);
    if (batch == nullptr) {
      // the end of a stream is a success returning a released array, not
      // an error - the same convention the source uses with us
      out->release = nullptr;
      return 0;
    }
    exportArray(batch, out);
  } catch (const std::invalid_argument &error) {
    return streamFailed(state, error, EINVAL);
  } catch (const std::overflow_error &error) {
    return streamFailed(state, error, EINVAL);
  } catch (const py::error_already_set &error) {
    return streamFailed(state, error, EIO);
  } catch (const std::exception &error) {
    return streamFailed(state, error, EIO);
  }
  return 0;
}

static const char *streamGetLastError(ArrowArrayStream *stream) {
  return stateOf(stream)->error.c_str();
}

static void streamRelease(ArrowArrayStream *stream) {
  // dropping the state releases the source stream, and with it Python
  // references the consumer may be holding no lock for
  py::gil_scoped_acquire locked;
  delete static_cast<StreamPrivate *>(stream->private_data);
  stream->release = nullptr;
}

/// Called when the consumer drops the capsule. A capsule that was handed on
/// to a reader arrives here already released - the reader moved the struct
/// out and nulled this copy's callback - so only the struct itself is left
/// to free; one abandoned unread still owns the stream, and is released
/// here.
static void releaseStreamCapsule(PyObject *capsule) {
  auto *stream = static_cast<ArrowArrayStream *>(
      PyCapsule_GetPointer(capsule, "arrow_array_stream"));
  if (stream != nullptr) {
    if (stream->release != nullptr) {
      stream->release(stream);
    }
    delete stream;
  }
}

/// The pieces a split produces, as a stream of Arrow record batches of the
/// GeoArrow geometry and the index of the geometry each piece came from.
///
/// Nothing is split until the stream is read, and only one batch of the
/// source is held at a time, so a source larger than memory can be split
/// by a consumer that takes the batches as they come.
class SplitStream {
public:
  SplitStream(const py::object &source, GeometryType type,
              const grid::Grid &grid, bool bounded)
      : state(std::make_unique<SplitState>(source, type, grid, bounded)),
        type(type) {}

  std::string geometryType() const { return extensionName(type); }

  /// Arrow PyCapsule interface. A stream is consumed once: the capsule
  /// takes the split with it, and this object is spent afterwards.
  py::capsule arrowCStream(const py::object &requested_schema) {
    // requested_schema is part of the protocol and ignored here: the
    // pieces are always GeoArrow with interleaved coordinates
    (void)requested_schema;
    if (state == nullptr) {
      throw std::invalid_argument(
          "This split has already been read: an Arrow stream can only be "
          "consumed once, so split again to read the pieces again");
    }
    auto *owned = new StreamPrivate{std::move(state)};
    auto *stream = new ArrowArrayStream();
    stream->get_schema = streamGetSchema;
    stream->get_next = streamGetNext;
    stream->get_last_error = streamGetLastError;
    stream->release = streamRelease;
    stream->private_data = owned;
    return py::capsule(stream, "arrow_array_stream", releaseStreamCapsule);
  }

private:
  std::unique_ptr<SplitState> state;
  /// kept alongside the state, which the first export takes away, so that
  /// the geometry type can still be reported afterwards
  GeometryType type;
};

// -- The split functions -----------------------------------------------------

static grid::Grid makeGrid(int nrows, int ncols,
                           const std::vector<double> &transform) {
  transform::Affine affine(transform[0], transform[1], transform[2],
                           transform[3], transform[4], transform[5]);
  return {static_cast<std::size_t>(ncols), static_cast<std::size_t>(nrows),
          affine};
}

static SplitStream splitLineStrings(const py::object &linestrings, int nrows,
                                    int ncols, std::vector<double> transform,
                                    bool bounded) {
  return {linestrings, GeometryType::linestring,
          makeGrid(nrows, ncols, transform), bounded};
}

static SplitStream splitPolygons(const py::object &polygons, int nrows,
                                 int ncols, std::vector<double> transform) {
  return {polygons, GeometryType::polygon, makeGrid(nrows, ncols, transform),
          false};
}

void register_module(py::module_ &m) {
  py::class_<SplitStream>(
      m, "SplitStream",
      "A stream of split geometries, readable through the Arrow PyCapsule "
      "stream interface, e.g. by pyarrow.RecordBatchReader or "
      "geopandas.GeoDataFrame.from_arrow. Each record batch holds a "
      "GeoArrow 'geometry' column of the pieces and a 'parent' column "
      "giving the index of the geometry each piece was split from. The "
      "split runs as the stream is read, one source batch at a time.")
      .def_property_readonly("geometry_type", &SplitStream::geometryType,
                             "The GeoArrow extension name of the pieces, "
                             "e.g. 'geoarrow.linestring'")
      .def("__arrow_c_stream__", &SplitStream::arrowCStream,
           py::arg("requested_schema") = py::none());

  m.def("split_linestrings", &splitLineStrings, py::arg("linestrings"),
        py::arg("nrows"), py::arg("ncols"), py::arg("transform"),
        py::arg("bounded") = false,
        R"(Split LineStrings along a grid.

Takes geoarrow.linestring geometries, with coordinates interleaved or
separated, from any object supporting the Arrow PyCapsule interface: a pyarrow
ChunkedArray, Table or RecordBatchReader, a GeoParquet or Dataset
reader, the result of GeoSeries.to_arrow(geometry_encoding="geoarrow"),
or a record batch stream with a GeoArrow geometry column.

Returns a SplitStream of the LineString pieces. Nothing is split until
the stream is read.)");

  m.def("split_polygons", &splitPolygons, py::arg("polygons"),
        py::arg("nrows"), py::arg("ncols"), py::arg("transform"),
        R"(Split Polygons along a grid.

Takes geoarrow.polygon geometries, with coordinates interleaved or
separated, from any object supporting the Arrow PyCapsule interface: a
pyarrow ChunkedArray,
Table or RecordBatchReader, a GeoParquet or Dataset reader, the result
of GeoSeries.to_arrow(geometry_encoding="geoarrow"), or a record batch
stream with a GeoArrow geometry column.

Returns a SplitStream of the Polygon pieces. Nothing is split until the
stream is read.)");
}

} // namespace geoarrow
} // namespace snail
