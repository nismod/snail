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
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

// Vendored, and namespaced to Snail so that loading this extension alongside
// pyarrow - or any other wheel that vendors nanoarrow - cannot collide.
// Brings the Arrow C data and stream interface structs with it.
#include "nanoarrow.hpp"

#include "geoarrow.hpp"
#include "geometry.hpp"
#include "grid.hpp"
#include "operations.hpp"
#include "transform.hpp"

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
/// The blob is packed rather than a string - an int32 count of key/value
/// pairs, then each key and value as an int32 length followed by that many
/// bytes - which nanoarrow decodes for us. Returns an empty string if the
/// key is absent, or if there is no metadata at all.
static std::string metadataValue(const char *metadata, const char *key) {
  // left untouched when the key is not found, so start it empty
  ArrowStringView value = ArrowCharView(nullptr);
  if (ArrowMetadataGetValue(metadata, ArrowCharView(key), &value) !=
          NANOARROW_OK ||
      value.data == nullptr) {
    return "";
  }
  return {value.data, static_cast<std::size_t>(value.size_bytes)};
}

// -- Reading GeoArrow arrays -------------------------------------------------

/// Does any slot of an array hold a null? A producer may leave null_count
/// at -1 rather than count them, in which case nanoarrow counts the bits of
/// the validity bitmap itself.
static bool hasNulls(const ArrowArrayView *view) {
  return ArrowArrayViewComputeNullCount(view) > 0;
}

/// Where element i of a list array's run of children begins, in the units
/// of the child array.
///
/// nanoarrow reads the offsets buffer as 32-bit for a list ("+l") or 64-bit
/// for a large list ("+L") from the view's storage type, so the two need no
/// distinguishing here. It indexes that buffer directly, though, and a
/// sliced array carries a slot offset of its own that shifts where in the
/// buffer element i is described - hence the view's offset added here.
static int64_t listBegin(const ArrowArrayView *view, int64_t i) {
  return ArrowArrayViewListChildOffset(view, view->offset + i);
}

static int64_t listEnd(const ArrowArrayView *view, int64_t i) {
  return listBegin(view, i + 1);
}

/// GeoArrow coordinates, in either of the layouts the format allows:
/// interleaved as a fixed_size_list<double>[2], which is how geopandas
/// hands over a geometry column, or separated as a struct of an x and a y
/// array, which is how GeoParquet stores them. Any further dimensions (z,
/// m) are ignored - splitting is planar.
class Coordinates {
public:
  Coordinates() = default;

  /// Read from a view of coordinates whose schema checkSchema has already
  /// passed, so the layout here is one of the two it allows. The schema
  /// comes along because a view carries no field names, and the separated
  /// layout finds its x and y by name.
  Coordinates(const ArrowArrayView *view, const ArrowSchema *schema) {
    if (hasNulls(view)) {
      throw std::invalid_argument("Cannot split geometries with missing "
                                  "(null) coordinates");
    }
    if (view->storage_type == NANOARROW_TYPE_FIXED_SIZE_LIST) {
      // A fixed-size list lays its elements end to end in one child buffer,
      // so vertex v starts at stride * v - shifted by the list's own slot
      // offset, and again by the child's.
      stride = view->layout.child_size_elements;
      const ArrowArrayView *values = view->children[0];
      if (hasNulls(values)) {
        throw std::invalid_argument("Cannot split geometries with missing "
                                    "(null) coordinates");
      }
      xy = values->buffer_views[1].data.as_double;
      xy_base = values->offset + stride * view->offset;
      return;
    }
    // Separated: one child array per dimension, each with its own offset
    const ArrowArrayView *x_view = view->children[dimension(schema, "x", 0)];
    const ArrowArrayView *y_view = view->children[dimension(schema, "y", 1)];
    if (hasNulls(x_view) || hasNulls(y_view)) {
      throw std::invalid_argument("Cannot split geometries with missing "
                                  "(null) coordinates");
    }
    x = x_view->buffer_views[1].data.as_double;
    y = y_view->buffer_views[1].data.as_double;
    x_base = x_view->offset + view->offset;
    y_base = y_view->offset + view->offset;
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
  const ArrowArrayView *lines;
  Coordinates coordinates;

  LineStringReader(const ArrowArrayView *view, const ArrowSchema *schema)
      : lines(view) {
    if (hasNulls(view)) {
      throw std::invalid_argument(
          "Cannot split missing (null) geometries: drop or fill null "
          "geometries first");
    }
    coordinates = Coordinates(view->children[0], schema->children[0]);
  }

  int64_t size() const { return lines->length; }

  /// Read linestring i into the given buffer
  void read(int64_t i, linestr &out) const {
    out.clear();
    int64_t begin = listBegin(lines, i);
    int64_t end = listEnd(lines, i);
    out.reserve(static_cast<std::size_t>(end - begin));
    for (int64_t v = begin; v < end; v++) {
      out.push_back(coordinates.at(v));
    }
  }
};

/// A geoarrow.polygon batch: a list of rings per polygon, each ring a list
/// of coordinates - list<rings: list<vertices: coordinates>>
struct PolygonReader {
  const ArrowArrayView *polygons;
  const ArrowArrayView *rings;
  Coordinates coordinates;

  PolygonReader(const ArrowArrayView *view, const ArrowSchema *schema)
      : polygons(view), rings(view->children[0]) {
    if (hasNulls(polygons) || hasNulls(rings)) {
      throw std::invalid_argument(
          "Cannot split missing (null) geometries or rings: drop or fill "
          "null geometries first");
    }
    const ArrowSchema *rings_schema = schema->children[0];
    coordinates = Coordinates(rings->children[0], rings_schema->children[0]);
  }

  int64_t size() const { return polygons->length; }

  /// Read the rings of polygon i into the given buffer, exterior first.
  /// The ring structure is explicit in the offsets, so rings are recovered
  /// exactly rather than inferred from where coordinates close back on
  /// themselves.
  void read(int64_t i, std::vector<linestr> &out) const {
    out.clear();
    for (int64_t r = listBegin(polygons, i); r < listEnd(polygons, i); r++) {
      linestr ring;
      int64_t begin = listBegin(rings, r);
      int64_t end = listEnd(rings, r);
      ring.reserve(static_cast<std::size_t>(end - begin));
      for (int64_t v = begin; v < end; v++) {
        ring.push_back(coordinates.at(v));
      }
      out.push_back(std::move(ring));
    }
  }
};

// -- Reading the source ------------------------------------------------------

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
      ArrowArrayStream moved =
          movedFromCapsule<ArrowArrayStream>(capsule, "arrow_array_stream");
      stream.reset(&moved);
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
      ArrowSchema moved_schema = movedFromCapsule<ArrowSchema>(
          capsules[0].cast<py::capsule>(), "arrow_schema");
      ArrowArray moved_array = movedFromCapsule<ArrowArray>(
          capsules[1].cast<py::capsule>(), "arrow_array");
      schema.reset(&moved_schema);
      single.reset(&moved_array);
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

    // One view over the geometry column, laid out from its schema now and
    // pointed at each batch as it arrives.
    ArrowError error;
    ArrowErrorInit(&error);
    if (ArrowArrayViewInitFromSchema(view.get(), geometrySchema(), &error) !=
        NANOARROW_OK) {
      throw std::invalid_argument(
          std::string("Could not read the geometry column: ") + error.message);
    }
  }

  /// The source stream, its schema and any batch still held may all be
  /// backed by Python objects, and a consumer may drop the split stream
  /// without holding the GIL, so take it to let them go. Releasing them here
  /// leaves the members' own destructors nothing to do.
  ~InputStream() {
    py::gil_scoped_acquire locked;
    single.reset();
    schema.reset();
    stream.reset();
  }

  /// The schema of the geometries themselves, within the batches
  const ArrowSchema *geometrySchema() const {
    return geometry_child >= 0 ? schema.get()->children[geometry_child]
                               : schema.get();
  }

  /// Pull the next batch and view its geometries, returning nullptr once the
  /// source is exhausted. The batch owns the buffers the view points at, so
  /// it has to outlive the reading.
  const ArrowArrayView *next(nanoarrow::UniqueArray &batch) {
    batch.reset();
    if (single->release != nullptr) {
      // a one-batch source: hand the array over, emptying it so that the
      // next call reports the end of the stream
      single.move(batch.get());
    } else if (stream->release == nullptr) {
      return nullptr;
    } else {
      if (stream.get()->get_next(stream.get(), batch.get()) != 0) {
        throw std::runtime_error(
            std::string("Could not read the next batch of geometries: ") +
            lastError());
      }
      // the producer marks the end of the stream with a released array
      if (batch->release == nullptr) {
        return nullptr;
      }
    }

    // Pointing the view at the batch also checks it over: that its children
    // and buffers are the shape the schema promised, and that its offsets
    // stay inside the arrays they index. Reading below can then trust it.
    ArrowError error;
    ArrowErrorInit(&error);
    if (ArrowArrayViewSetArray(view.get(), geometryArray(batch.get()),
                               &error) != NANOARROW_OK) {
      throw std::invalid_argument(
          std::string("Could not read a batch of geometries: ") +
          error.message);
    }
    return view.get();
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

  /// A stream reports a failure by returning non-zero and leaving the
  /// detail behind for get_last_error
  const char *lastError() {
    if (stream->release == nullptr || stream->get_last_error == nullptr) {
      return "";
    }
    const char *message = stream.get()->get_last_error(stream.get());
    return message == nullptr ? "" : message;
  }

  nanoarrow::UniqueArrayStream stream;
  nanoarrow::UniqueArray single;
  nanoarrow::UniqueArrayView view;
  nanoarrow::UniqueSchema schema;
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

/// Mark a schema and everything under it as holding no nulls
static void clearNullable(ArrowSchema *schema) {
  schema->flags &= ~static_cast<int64_t>(ARROW_FLAG_NULLABLE);
  for (int64_t i = 0; i < schema->n_children; i++) {
    clearNullable(schema->children[i]);
  }
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
/// nanoarrow writes those format strings and owns the children, so the tree
/// is described here by naming types rather than by hand-assembling structs.
/// Setting a list type allocates its one child, which is then typed in turn.
///
/// Coordinates go back interleaved because that is what the split fills: a
/// vector of Coord is already a run of x, y, x, y doubles.
static void exportSchema(GeometryType type, ArrowSchema *out) {
  nanoarrow::UniqueSchema schema;
  ArrowSchemaInit(schema.get());
  NANOARROW_THROW_NOT_OK(ArrowSchemaSetTypeStruct(schema.get(), 2));

  ArrowSchema *geometry = schema->children[0];
  NANOARROW_THROW_NOT_OK(ArrowSchemaSetType(geometry, NANOARROW_TYPE_LIST));
  NANOARROW_THROW_NOT_OK(ArrowSchemaSetName(geometry, "geometry"));

  // a polygon is a list of rings, so its coordinates sit one level deeper
  ArrowSchema *vertices = geometry->children[0];
  if (type == GeometryType::polygon) {
    NANOARROW_THROW_NOT_OK(ArrowSchemaSetType(vertices, NANOARROW_TYPE_LIST));
    NANOARROW_THROW_NOT_OK(ArrowSchemaSetName(vertices, "rings"));
    vertices = vertices->children[0];
  }
  NANOARROW_THROW_NOT_OK(ArrowSchemaSetTypeFixedSize(
      vertices, NANOARROW_TYPE_FIXED_SIZE_LIST, 2));
  NANOARROW_THROW_NOT_OK(ArrowSchemaSetName(vertices, "vertices"));
  NANOARROW_THROW_NOT_OK(
      ArrowSchemaSetType(vertices->children[0], NANOARROW_TYPE_DOUBLE));
  NANOARROW_THROW_NOT_OK(ArrowSchemaSetName(vertices->children[0], "xy"));

  // the extension name rides on the geometry field, which is where a
  // GeoArrow reader looks for it. SetMetadata copies the blob, so the
  // builder's buffer is free to go out of scope here.
  nanoarrow::UniqueBuffer metadata;
  NANOARROW_THROW_NOT_OK(ArrowMetadataBuilderInit(metadata.get(), nullptr));
  NANOARROW_THROW_NOT_OK(ArrowMetadataBuilderAppend(
      metadata.get(), ArrowCharView("ARROW:extension:name"),
      ArrowCharView(extensionName(type))));
  NANOARROW_THROW_NOT_OK(ArrowSchemaSetMetadata(
      geometry, reinterpret_cast<const char *>(metadata->data)));

  NANOARROW_THROW_NOT_OK(
      ArrowSchemaSetType(schema->children[1], NANOARROW_TYPE_INT64));
  NANOARROW_THROW_NOT_OK(ArrowSchemaSetName(schema->children[1], "parent"));

  // A split never produces a null - not a null piece, ring, vertex or
  // parent index - so say so. nanoarrow marks every field nullable by
  // default; GeoArrow asks that a geometry's inner arrays contain no nulls,
  // and declaring it lets a reader skip looking for validity bitmaps.
  clearNullable(schema.get());

  schema.move(out);
}

/// Hand a vector's storage to Arrow as buffer `i` of an array, without
/// copying it: nanoarrow takes the vector over and frees it with the buffer.
template <typename T>
static void adoptBuffer(ArrowArray *array, int64_t i, std::vector<T> values) {
  nanoarrow::UniqueBuffer buffer;
  nanoarrow::BufferInitSequence(buffer.get(), std::move(values));
  NANOARROW_THROW_NOT_OK(ArrowArraySetBuffer(array, i, buffer.get()));
}

/// Build a record batch over a batch of split pieces, mirroring the tree
/// exportSchema describes and giving Arrow the split's own buffers rather
/// than copying them out. The batch takes the vectors over, so it outlives
/// the split that made it.
///
/// Buffer 0 of every array is the validity bitmap, left null throughout
/// because nothing a split produces is null. What follows depends on the
/// type: a list has its offsets in buffer 1, a primitive its values, and a
/// struct or fixed-size list has no second buffer at all - its children
/// hold everything. Lengths are set per level, since the buffers arrive
/// whole rather than being appended to element by element.
static void exportArray(BatchData data, ArrowArray *out) {
  nanoarrow::UniqueSchema schema;
  exportSchema(data.type, schema.get());

  nanoarrow::UniqueArray array;
  NANOARROW_THROW_NOT_OK(
      ArrowArrayInitFromSchema(array.get(), schema.get(), nullptr));

  const int64_t pieces = data.size();
  const int64_t n_vertices = static_cast<int64_t>(data.coordinates.size());

  ArrowArray *geometry = array->children[0];
  ArrowArray *vertices = geometry->children[0];
  if (data.type == GeometryType::polygon) {
    // polygons over rings, then rings over vertices
    ArrowArray *rings = vertices;
    vertices = rings->children[0];
    rings->length = static_cast<int64_t>(data.vertex_offsets.size()) - 1;
    adoptBuffer(rings, 1, std::move(data.vertex_offsets));
    adoptBuffer(geometry, 1, std::move(data.ring_offsets));
  } else {
    // a linestring is a plain run of vertices, so one level of offsets does
    adoptBuffer(geometry, 1, std::move(data.vertex_offsets));
  }
  geometry->length = pieces;
  vertices->length = n_vertices;

  // the coordinates as bare doubles: two per vertex, hence the length
  ArrowArray *xy = vertices->children[0];
  xy->length = 2 * n_vertices;
  adoptBuffer(xy, 1, std::move(data.coordinates));

  ArrowArray *parent = array->children[1];
  parent->length = pieces;
  adoptBuffer(parent, 1, std::move(data.parents));

  // the record batch itself: a struct of the two columns, whose own length
  // is the number of pieces
  array->length = pieces;
  // MINIMAL checks every buffer against the length we just set, which is
  // exactly the mistake this hand-nested layout could make; it stops short
  // of the level that would try to reallocate the buffers we just adopted
  NANOARROW_THROW_NOT_OK(ArrowArrayFinishBuilding(
      array.get(), NANOARROW_VALIDATION_LEVEL_MINIMAL, nullptr));

  array.move(out);
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
static std::optional<BatchData> nextSplitBatch(SplitState *state) {
  while (true) {
    nanoarrow::UniqueArray batch;
    const ArrowArrayView *geometries = state->input.next(batch);
    if (geometries == nullptr) {
      return std::nullopt;
    }
    const ArrowSchema *schema = state->input.geometrySchema();

    BatchData out;
    out.type = state->type;
    int64_t count = 0;
    if (state->type == GeometryType::polygon) {
      PolygonReader reader(geometries, schema);
      count = reader.size();
      py::gil_scoped_release unlocked;
      splitPolygonBatch(reader, state->grid, state->parent_base, out);
    } else {
      LineStringReader reader(geometries, schema);
      count = reader.size();
      py::gil_scoped_release unlocked;
      splitLineStringBatch(reader, state->grid, state->bounded,
                           state->parent_base, out);
    }
    state->parent_base += count;

    if (out.coordinates.size() >
        static_cast<std::size_t>(std::numeric_limits<int32_t>::max())) {
      throw std::overflow_error(
          "One batch split to more coordinates than a GeoArrow array with "
          "32-bit offsets can hold: read the source in smaller batches");
    }
    if (out.size() > 0) {
      return out;
    }
  }
}

// -- The split stream, as an Arrow C stream ----------------------------------

/// The producer behind the ArrowArrayStream we hand out. nanoarrow adapts
/// these three methods into the C callbacks a consumer calls, and owns the
/// instance: the consumer's release callback deletes it.
///
/// Those callbacks are plain C, so a C++ exception must not escape them.
/// They signal failure by returning a non-zero errno and leaving the detail
/// for get_last_error, and the consumer turns that code back into an
/// exception. Picking the code deliberately keeps the type a caller sees the
/// same either side of the stream starting - EINVAL surfaces as a
/// ValueError, as a bad argument caught up front would have.
class SplitProducer {
public:
  /// Hand a split over to a stream the consumer owns from here on
  static void toArrayStream(std::unique_ptr<SplitState> state,
                            ArrowArrayStream *out) {
    nanoarrow::ArrayStreamFactory<SplitProducer>::InitArrayStream(
        new SplitProducer(std::move(state)), out);
  }

private:
  explicit SplitProducer(std::unique_ptr<SplitState> state)
      : state(std::move(state)) {}

  /// the methods below are called from C through the factory's callbacks
  friend class nanoarrow::ArrayStreamFactory<SplitProducer>;

  int GetSchema(ArrowSchema *out) {
    try {
      exportSchema(state->type, out);
    } catch (const std::exception &error) {
      return failed(error, EIO);
    }
    return 0;
  }

  int GetNext(ArrowArray *out) {
    // The source and the geometries it holds may both be Python objects, and
    // the consumer may call us without the GIL, so take it for the read and
    // give it back around the splitting itself.
    py::gil_scoped_acquire locked;
    try {
      std::optional<BatchData> batch = nextSplitBatch(state.get());
      if (!batch.has_value()) {
        // the end of a stream is a success returning a released array, not
        // an error - the same convention the source uses with us
        out->release = nullptr;
        return 0;
      }
      exportArray(std::move(*batch), out);
    } catch (const std::invalid_argument &error) {
      return failed(error, EINVAL);
    } catch (const std::overflow_error &error) {
      return failed(error, EINVAL);
    } catch (const std::exception &error) {
      return failed(error, EIO);
    }
    return 0;
  }

  /// The consumer reads this after a callback fails, and only needs it to
  /// stay valid until the next one, so holding the message here is enough.
  const char *GetLastError() { return last_error.c_str(); }

  /// Record a failure against the stream and return the code to report
  int failed(const std::exception &error, int code) {
    last_error = error.what();
    return code;
  }

  std::unique_ptr<SplitState> state;
  std::string last_error;
};

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
    // The capsule takes the struct before the split goes into it, marked
    // released so that failing to build the capsule frees an empty struct
    // rather than stranding a split behind one nothing owns.
    auto owned = std::make_unique<ArrowArrayStream>();
    owned->release = nullptr;
    py::capsule capsule(owned.get(), "arrow_array_stream",
                        releaseStreamCapsule);
    SplitProducer::toArrayStream(std::move(state), owned.release());
    return capsule;
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
