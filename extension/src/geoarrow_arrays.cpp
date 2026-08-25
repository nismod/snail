/// Reading GeoArrow arrays, splitting them, and writing the pieces back.
/// See geoarrow_arrays.hpp for why this is a file of its own.

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

#include "geoarrow_arrays.hpp"

#include "transform.hpp"

namespace snail {
namespace geoarrow {

const char *extensionName(GeometryType type) {
  switch (type) {
  case GeometryType::polygon:
    return "geoarrow.polygon";
  case GeometryType::mixed:
    return "geoarrow.wkb";
  default:
    return "geoarrow.linestring";
  }
}

static const char *WKB_EXTENSION_NAME = "geoarrow.wkb";

/// The name a piece of this type goes by in WKB error messages
static const char *geometryTypeName(enum GeoArrowGeometryType type) {
  switch (type) {
  case GEOARROW_GEOMETRY_TYPE_POINT:
    return "Point";
  case GEOARROW_GEOMETRY_TYPE_LINESTRING:
    return "LineString";
  case GEOARROW_GEOMETRY_TYPE_POLYGON:
    return "Polygon";
  case GEOARROW_GEOMETRY_TYPE_MULTIPOINT:
    return "MultiPoint";
  case GEOARROW_GEOMETRY_TYPE_MULTILINESTRING:
    return "MultiLineString";
  case GEOARROW_GEOMETRY_TYPE_MULTIPOLYGON:
    return "MultiPolygon";
  case GEOARROW_GEOMETRY_TYPE_GEOMETRYCOLLECTION:
    return "GeometryCollection";
  default:
    return "geometry";
  }
}

/// The one WKB geometry type a typed split accepts
static enum GeoArrowGeometryType wanted(GeometryType type) {
  return type == GeometryType::polygon ? GEOARROW_GEOMETRY_TYPE_POLYGON
                                       : GEOARROW_GEOMETRY_TYPE_LINESTRING;
}

// -- Arrow schema metadata ---------------------------------------------------

/// Read a value out of an ArrowSchema metadata blob. GeoArrow declares a
/// geometry type there, under "ARROW:extension:name".
///
/// The blob is packed rather than a string - an int32 count of key/value
/// pairs, then each key and value as an int32 length followed by that many
/// bytes - which nanoarrow decodes for us. Returns an empty string if the
/// key is absent, or if there is no metadata at all.
std::string metadataValue(const char *metadata, const char *key) {
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

  /// Vertices [begin, end) as a run the split kernels can read.
  ///
  /// Interleaved 2D coordinates already *are* a run of Coord - that is what
  /// the static_assert on sizeof(Coord) at the top of this file pins down -
  /// so the span points straight into the Arrow buffer and nothing is
  /// copied. Every other layout has a gap between one vertex's y and the
  /// next's x, or holds x and y apart altogether, so those are gathered into
  /// a buffer that is reused from one geometry to the next.
  /// Are the coordinates laid out exactly as a run of Coord? Interleaved
  /// with nothing but x and y is, provided the slice this view starts at
  /// lands on a vertex boundary rather than between a vertex's two doubles.
  bool contiguous() const {
    return xy != nullptr && stride == 2 && xy_base % 2 == 0;
  }

  operations::CoordSpan run(int64_t begin, int64_t end) {
    const std::size_t count = static_cast<std::size_t>(end - begin);
    if (contiguous()) {
      const auto *first =
          reinterpret_cast<const geo::Coord *>(xy + xy_base) + begin;
      return {first, count};
    }
    gathered.clear();
    gathered.reserve(count);
    for (int64_t v = begin; v < end; v++) {
      gathered.push_back(at(v));
    }
    return gathered;
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
  /// where a layout that is not already a run of Coord is gathered into
  std::vector<geo::Coord> gathered;
};

/// Check the GeoArrow extension name, when the producer declares one. A
/// plain nested list array of the right shape is accepted too, so that
/// arrays built directly with pyarrow work.
static Encoding checkExtensionName(const ArrowSchema *schema,
                                   GeometryType type) {
  std::string name = metadataValue(schema->metadata, "ARROW:extension:name");
  if (name.empty() || name == extensionName(type)) {
    return Encoding::native;
  }
  if (name == WKB_EXTENSION_NAME) {
    return Encoding::wkb;
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
/// before any data has arrived, and say how it is encoded.
///
/// Only the native encodings can be checked this far ahead: a WKB column
/// says nothing about what its geometries are until they are decoded, so
/// the type check for those happens per feature as they are read.
Encoding checkGeometrySchema(const ArrowSchema *schema,
                                    GeometryType type) {
  if (type == GeometryType::mixed) {
    // A mixed split takes each geometry on its own terms, so the column may
    // be anything geoarrow-c can walk - any native encoding, or WKB. Asking
    // geoarrow-c to parse the schema is both the check and the answer.
    GeoArrowSchemaView view;
    GeoArrowError error;
    if (GeoArrowSchemaViewInit(&view, schema, &error) != GEOARROW_OK) {
      throw std::invalid_argument(
          std::string("Expected GeoArrow geometries: ") + error.message);
    }
    // read through the visitor either way, since only that reports a
    // geometry's own type
    return Encoding::wkb;
  }
  const Encoding encoding = checkExtensionName(schema, type);
  if (encoding == Encoding::wkb) {
    return encoding;
  }
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
  return encoding;
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

  /// The vertices of linestring i
  operations::CoordSpan read(int64_t i) {
    return coordinates.run(listBegin(lines, i), listEnd(lines, i));
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

  /// The rings of polygon i, exterior first. The ring structure is explicit
  /// in the offsets, so rings are recovered exactly rather than inferred
  /// from where coordinates close back on themselves.
  ///
  /// Only one ring's span is valid at a time unless the coordinates are
  /// contiguous, since a gathered ring reuses the same buffer - so a
  /// polygon whose coordinates need gathering is materialised ring by ring
  /// into `out` first, and spans taken over that.
  void read(int64_t i, std::vector<operations::CoordSpan> &out,
            std::vector<linestr> &scratch) {
    out.clear();
    const int64_t first = listBegin(polygons, i);
    const int64_t last = listEnd(polygons, i);
    if (coordinates.contiguous()) {
      for (int64_t r = first; r < last; r++) {
        out.push_back(coordinates.run(listBegin(rings, r), listEnd(rings, r)));
      }
      return;
    }
    std::size_t used = 0;
    for (int64_t r = first; r < last; r++) {
      if (used == scratch.size()) {
        scratch.emplace_back();
      }
      // the ring buffers keep their capacity from the previous polygon
      linestr &ring = scratch[used++];
      ring.clear();
      const int64_t begin = listBegin(rings, r);
      const int64_t end = listEnd(rings, r);
      ring.reserve(static_cast<std::size_t>(end - begin));
      for (int64_t v = begin; v < end; v++) {
        ring.push_back(coordinates.at(v));
      }
    }
    for (std::size_t r = 0; r < used; r++) {
      out.push_back(scratch[r]);
    }
  }
};

// -- Reading WKB -------------------------------------------------------------

WkbReader::~WkbReader() {
  if (ready) {
    GeoArrowArrayReaderReset(&reader);
  }
}

void WkbReader::init(const ArrowSchema *schema) {
  GeoArrowError error;
  if (GeoArrowArrayReaderInitFromSchema(&reader, schema, &error) !=
      GEOARROW_OK) {
    throw std::invalid_argument(
        std::string("Could not read the geometry column as WKB: ") +
        error.message);
  }
  ready = true;
}

void WkbReader::setArray(const ArrowArray *array) {
  GeoArrowError error;
  if (GeoArrowArrayReaderSetArray(&reader, array, &error) != GEOARROW_OK) {
    throw std::invalid_argument(
        std::string("Could not read a batch of WKB geometries: ") +
        error.message);
  }
}

void WkbReader::visit(int64_t i, GeoArrowVisitor *visitor) {
  GeoArrowError error;
  visitor->error = &error;
  if (GeoArrowArrayReaderVisit(&reader, i, 1, visitor) != GEOARROW_OK) {
    throw std::invalid_argument(
        std::string("Could not read WKB geometry at row ") + std::to_string(i) +
        ": " + error.message);
  }
}

// -- Writing batches of split pieces -----------------------------------------

/// Append a run of coordinates geoarrow-c handed over.
///
/// They arrive in runs rather than one point at a time, and WKB gives one
/// run per ring laid out as interleaved xy - which is a run of Coord
/// already - so the common case is a single bulk append. A z or m ordinate
/// to step over, or x and y held apart, falls back to gathering. Only x and
/// y are read: splitting is planar.
static int appendCoords(const GeoArrowCoordView *c,
                        std::vector<geo::Coord> &into) {
  const std::size_t count = static_cast<std::size_t>(c->n_coords);
  if (count == 0) {
    return GEOARROW_OK;
  }
  if (c->coords_stride == 2 && c->values[1] == c->values[0] + 1) {
    const auto *first = reinterpret_cast<const geo::Coord *>(c->values[0]);
    into.insert(into.end(), first, first + count);
    return GEOARROW_OK;
  }
  const int32_t stride = c->coords_stride;
  into.reserve(into.size() + count);
  for (std::size_t i = 0; i < count; i++) {
    into.push_back({c->values[0][i * stride], c->values[1][i * stride]});
  }
  return GEOARROW_OK;
}

/// Writes pieces out as WKB, for the mixed split.
///
/// geoarrow-c's writer is driven through the same visitor protocol its
/// reader speaks, so a piece is written by calling geom_start, the
/// coordinates, and geom_end. Coordinates go in as a run rather than a
/// point at a time, and a vector of Coord is already the interleaved xy
/// buffer a run describes, so nothing is copied on the way in either.
class WkbWriter {
public:
  WkbWriter(const WkbWriter &) = delete;
  WkbWriter &operator=(const WkbWriter &) = delete;

  WkbWriter() {
    if (GeoArrowWKBWriterInit(&writer) != GEOARROW_OK) {
      throw std::runtime_error("Could not start writing WKB");
    }
    GeoArrowWKBWriterInitVisitor(&writer, &into);
  }

  ~WkbWriter() { GeoArrowWKBWriterReset(&writer); }

  /// A run of coordinates as geoarrow-c describes one: x and y interleaved,
  /// two doubles to a vertex, which is exactly a Coord array
  static GeoArrowCoordView asRun(const geo::Coord *coordinates,
                                 std::size_t count) {
    GeoArrowCoordView run{};
    const auto *values = reinterpret_cast<const double *>(coordinates);
    run.values[0] = values;
    run.values[1] = values + 1;
    run.n_coords = static_cast<int64_t>(count);
    run.n_values = 2;
    run.coords_stride = 2;
    return run;
  }

  void writeLineString(const geo::Coord *coordinates, std::size_t count) {
    begin(GEOARROW_GEOMETRY_TYPE_LINESTRING);
    if (count > 0) {
      const GeoArrowCoordView run = asRun(coordinates, count);
      check(into.coords(&into, &run));
    }
    end();
  }

  void writePoint(const geo::Coord &coordinate) {
    begin(GEOARROW_GEOMETRY_TYPE_POINT);
    const GeoArrowCoordView run = asRun(&coordinate, 1);
    check(into.coords(&into, &run));
    end();
  }

  /// Rings exterior first, as the split produces them
  void writePolygon(const geo::Coord *coordinates, const int32_t *ring_offsets,
                    std::size_t rings) {
    begin(GEOARROW_GEOMETRY_TYPE_POLYGON);
    for (std::size_t r = 0; r < rings; r++) {
      const int32_t from = ring_offsets[r];
      const int32_t to = ring_offsets[r + 1];
      check(into.ring_start(&into));
      if (to > from) {
        const GeoArrowCoordView run =
            asRun(coordinates + from, static_cast<std::size_t>(to - from));
        check(into.coords(&into, &run));
      }
      check(into.ring_end(&into));
    }
    end();
  }

  /// A geometry that held nothing goes back as itself, holding nothing
  void writeEmpty(enum GeoArrowGeometryType type) {
    begin(type);
    end();
  }

  /// Hand the written blobs over as an Arrow array
  void finish(ArrowArray *out) {
    GeoArrowError error;
    if (GeoArrowWKBWriterFinish(&writer, out, &error) != GEOARROW_OK) {
      throw std::runtime_error(std::string("Could not finish writing WKB: ") +
                               error.message);
    }
  }

private:
  void begin(enum GeoArrowGeometryType type) {
    check(into.feat_start(&into));
    check(into.geom_start(&into, type, GEOARROW_DIMENSIONS_XY));
  }

  void end() {
    check(into.geom_end(&into));
    check(into.feat_end(&into));
  }

  static void check(int code) {
    if (code != GEOARROW_OK) {
      throw std::runtime_error("Could not write a piece as WKB");
    }
  }

  GeoArrowWKBWriter writer{};
  GeoArrowVisitor into{};
};


/// One batch of split pieces: the geometries in GeoArrow's flat layout,
/// and for each piece the index of the geometry it was split from.
BatchData::BatchData() = default;
BatchData::BatchData(BatchData &&) noexcept = default;
BatchData &BatchData::operator=(BatchData &&) noexcept = default;
/// out of line, so that the header need only forward-declare WkbWriter
BatchData::~BatchData() = default;

std::size_t BatchData::coordinateCount() const {
  switch (type) {
  case GeometryType::mixed:
    return 0;
  case GeometryType::polygon:
    return polygons.coordinates.size();
  default:
    return lines.coordinates.size();
  }
}

namespace {

/// The buffers of a native batch, picked by geometry type
std::vector<geo::Coord> &coordinatesOf(BatchData &data) {
  return data.type == GeometryType::polygon ? data.polygons.coordinates
                                            : data.lines.coordinates;
}

/// Where each run of coordinates begins: one run per piece for linestrings,
/// one per ring for polygons.
std::vector<int32_t> &vertexOffsetsOf(BatchData &data) {
  return data.type == GeometryType::polygon ? data.polygons.ring_offsets
                                            : data.lines.offsets;
}

/// Where each polygon's run of rings begins; unused for linestrings
std::vector<int32_t> &ringOffsetsOf(BatchData &data) {
  return data.polygons.polygon_offsets;
}

} // namespace

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
/// The part of the output schema every encoding shares: the extension name
/// on the geometry field, the parent index alongside it, and the promise
/// that none of it is nullable.
static void exportSchemaTail(GeometryType type, nanoarrow::UniqueSchema &schema,
                             ArrowSchema *out) {
  // the extension name rides on the geometry field, which is where a
  // GeoArrow reader looks for it. SetMetadata copies the blob, so the
  // builder's buffer is free to go out of scope here.
  nanoarrow::UniqueBuffer metadata;
  NANOARROW_THROW_NOT_OK(ArrowMetadataBuilderInit(metadata.get(), nullptr));
  NANOARROW_THROW_NOT_OK(ArrowMetadataBuilderAppend(
      metadata.get(), ArrowCharView("ARROW:extension:name"),
      ArrowCharView(extensionName(type))));
  NANOARROW_THROW_NOT_OK(ArrowSchemaSetMetadata(
      schema->children[0], reinterpret_cast<const char *>(metadata->data)));

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

void exportSchema(GeometryType type, ArrowSchema *out) {
  nanoarrow::UniqueSchema schema;
  ArrowSchemaInit(schema.get());
  NANOARROW_THROW_NOT_OK(ArrowSchemaSetTypeStruct(schema.get(), 2));

  ArrowSchema *geometry = schema->children[0];
  if (type == GeometryType::mixed) {
    // WKB is one binary blob per piece: no nesting, and no coordinates Arrow
    // can see. The extension name is still what tells a reader these are
    // geometries, so the tail of this function does the rest.
    NANOARROW_THROW_NOT_OK(ArrowSchemaSetType(geometry, NANOARROW_TYPE_BINARY));
    NANOARROW_THROW_NOT_OK(ArrowSchemaSetName(geometry, "geometry"));
    return exportSchemaTail(type, schema, out);
  }
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

  exportSchemaTail(type, schema, out);
}

/// Hand a vector's storage to Arrow as buffer `i` of an array, without
/// copying it: nanoarrow takes the vector over and frees it with the buffer.
template <typename T>
static void adoptBuffer(ArrowArray *array, int64_t i, std::vector<T> values) {
  nanoarrow::UniqueBuffer buffer;
  nanoarrow::BufferInitSequence(buffer.get(), std::move(values));
  NANOARROW_THROW_NOT_OK(ArrowArraySetBuffer(array, i, buffer.get()));
}

/// Build a record batch over a batch of WKB pieces.
///
/// The writer has already produced the geometry column as an Arrow array of
/// its own, so this only has to put it beside a parent column in a struct.
/// The geometry column is *moved* in - a bitwise copy, then the source's
/// release callback nulled without calling it, which is how Arrow says a
/// struct changes hands - so the blobs are not copied either.
static void exportWkbArray(BatchData data, ArrowArray *out) {
  const int64_t pieces = data.size();

  nanoarrow::UniqueArray geometry;
  data.wkb->finish(geometry.get());
  if (geometry->length != pieces) {
    throw std::runtime_error(
        "Wrote " + std::to_string(geometry->length) +
        " WKB pieces but recorded " + std::to_string(pieces) + " parents");
  }

  nanoarrow::UniqueArray array;
  NANOARROW_THROW_NOT_OK(
      ArrowArrayInitFromType(array.get(), NANOARROW_TYPE_STRUCT));
  NANOARROW_THROW_NOT_OK(ArrowArrayAllocateChildren(array.get(), 2));
  geometry.move(array->children[0]);

  ArrowArray *parent = array->children[1];
  NANOARROW_THROW_NOT_OK(
      ArrowArrayInitFromType(parent, NANOARROW_TYPE_INT64));
  parent->length = pieces;
  adoptBuffer(parent, 1, std::move(data.parents));

  array->length = pieces;
  NANOARROW_THROW_NOT_OK(ArrowArrayFinishBuilding(
      array.get(), NANOARROW_VALIDATION_LEVEL_MINIMAL, nullptr));

  array.move(out);
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
void exportArray(BatchData data, ArrowArray *out) {
  if (data.type == GeometryType::mixed) {
    return exportWkbArray(std::move(data), out);
  }

  nanoarrow::UniqueSchema schema;
  exportSchema(data.type, schema.get());

  nanoarrow::UniqueArray array;
  NANOARROW_THROW_NOT_OK(
      ArrowArrayInitFromSchema(array.get(), schema.get(), nullptr));

  const int64_t pieces = data.size();
  const int64_t n_vertices = static_cast<int64_t>(coordinatesOf(data).size());

  ArrowArray *geometry = array->children[0];
  ArrowArray *vertices = geometry->children[0];
  if (data.type == GeometryType::polygon) {
    // polygons over rings, then rings over vertices
    ArrowArray *rings = vertices;
    vertices = rings->children[0];
    rings->length = static_cast<int64_t>(vertexOffsetsOf(data).size()) - 1;
    adoptBuffer(rings, 1, std::move(vertexOffsetsOf(data)));
    adoptBuffer(geometry, 1, std::move(ringOffsetsOf(data)));
  } else {
    // a linestring is a plain run of vertices, so one level of offsets does
    adoptBuffer(geometry, 1, std::move(vertexOffsetsOf(data)));
  }
  geometry->length = pieces;
  vertices->length = n_vertices;

  // the coordinates as bare doubles: two per vertex, hence the length
  ArrowArray *xy = vertices->children[0];
  xy->length = 2 * n_vertices;
  adoptBuffer(xy, 1, std::move(coordinatesOf(data)));

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

/// Split one batch of linestrings into pieces
static void splitLineStringBatch(LineStringReader &reader,
                                 const grid::Grid &grid, bool bounded,
                                 int64_t parent_base, BatchData &out) {
  for (int64_t l = 0; l < reader.size(); l++) {
    const std::size_t before = out.lines.size();
    operations::splitLineStringGrid(reader.read(l), grid, bounded, out.lines);
    // one parent per piece this line produced; a line that fell apart into
    // nothing produces none
    out.parents.insert(out.parents.end(), out.lines.size() - before,
                       parent_base + l);
  }
}

/// Split one batch of polygons into pieces
static void splitPolygonBatch(PolygonReader &reader, const grid::Grid &grid,
                              int64_t parent_base, BatchData &out) {
  std::vector<operations::CoordSpan> rings;
  std::vector<linestr> scratch;
  for (int64_t p = 0; p < reader.size(); p++) {
    reader.read(p, rings, scratch);
    const std::size_t before = out.polygons.size();
    operations::splitPolygonGridPieces(rings, grid, out.polygons);
    out.parents.insert(out.parents.end(), out.polygons.size() - before,
                       parent_base + p);
  }
}

/// Splits the geometries geoarrow-c decodes out of a WKB column.
///
/// The callbacks arrive nested, and that nesting is the point: a
/// MULTILINESTRING opens a geom_start of its own and then one per part, and
/// a GEOMETRYCOLLECTION one per member, each carrying that member's own
/// type. Splitting happens at the innermost level, where the coordinates
/// are, so `depth` tracks how far in we are and `type` is whatever the
/// innermost open geometry is.
///
/// This class only ever accepts one type - it backs the typed entry points,
/// which give back native GeoArrow of that type, so a feature that is
/// something else has nowhere to go and is refused with the row it is in.
class WkbSplitter {
public:
  WkbSplitter(GeometryType type, const grid::Grid &grid, bool bounded,
              BatchData &out)
      : want(wanted(type)), expected(type), grid(grid), bounded(bounded),
        out(out) {}

  /// A visitor bound to this splitter. geoarrow-c calls plain C function
  /// pointers, so each one recovers the splitter from private_data.
  GeoArrowVisitor visitor() {
    GeoArrowVisitor v;
    GeoArrowVisitorInitVoid(&v);
    v.feat_start = [](GeoArrowVisitor *v) { return self(v)->featStart(); };
    v.null_feat = [](GeoArrowVisitor *v) { return self(v)->nullFeat(); };
    v.geom_start = [](GeoArrowVisitor *v, enum GeoArrowGeometryType type,
                      enum GeoArrowDimensions) {
      return self(v)->geomStart(type);
    };
    v.ring_start = [](GeoArrowVisitor *v) { return self(v)->ringStart(); };
    v.coords = [](GeoArrowVisitor *v, const GeoArrowCoordView *c) {
      return self(v)->coords(c);
    };
    v.ring_end = [](GeoArrowVisitor *v) { return self(v)->ringEnd(); };
    v.geom_end = [](GeoArrowVisitor *v) { return self(v)->geomEnd(); };
    v.private_data = this;
    return v;
  }

  /// Which row of the source is being read, for error messages
  void startRow(int64_t index) { row = index; }

  /// How many pieces this splitter has produced
  std::size_t pieces() const {
    return expected == GeometryType::polygon ? out.polygons.size()
                                             : out.lines.size();
  }

private:
  static WkbSplitter *self(GeoArrowVisitor *v) {
    return static_cast<WkbSplitter *>(v->private_data);
  }

  /// A callback cannot let a C++ exception escape into geoarrow-c, so it
  /// records the complaint and returns a failure code; the message is
  /// raised on the way back out.
  int refuse(std::string what) {
    problem = std::move(what);
    return EINVAL;
  }

  int featStart() {
    depth = 0;
    rings.clear();
    ring_starts.clear();
    coordinates.clear();
    return GEOARROW_OK;
  }

  /// The native path refuses a null geometry rather than dropping it, and
  /// this must too: the same data in two encodings should not behave
  /// differently. WKB knows which row it is on, so it says.
  int nullFeat() {
    return refuse("Cannot split missing (null) geometries (row " +
                  std::to_string(row) +
                  "): drop or fill null geometries first");
  }

  int geomStart(enum GeoArrowGeometryType type) {
    depth++;
    if (depth == 1 && type != want) {
      std::string message = std::string("Cannot split a ") +
                            geometryTypeName(type) + " (row " +
                            std::to_string(row) + ") as a " +
                            extensionName(expected);
      if (type == GEOARROW_GEOMETRY_TYPE_MULTILINESTRING ||
          type == GEOARROW_GEOMETRY_TYPE_MULTIPOLYGON ||
          type == GEOARROW_GEOMETRY_TYPE_MULTIPOINT ||
          type == GEOARROW_GEOMETRY_TYPE_GEOMETRYCOLLECTION) {
        message += "; merge or explode multi-part geometries before splitting";
      }
      return refuse(std::move(message));
    }
    return GEOARROW_OK;
  }

  int ringStart() {
    ring_starts.push_back(coordinates.size());
    return GEOARROW_OK;
  }

  int coords(const GeoArrowCoordView *c) {
    return appendCoords(c, coordinates);
  }

  int ringEnd() { return GEOARROW_OK; }

  int geomEnd() {
    depth--;
    if (depth != 0) {
      return GEOARROW_OK;
    }
    // the outermost geometry has closed, so split what it held
    if (expected == GeometryType::polygon) {
      rings.clear();
      for (std::size_t r = 0; r < ring_starts.size(); r++) {
        const std::size_t begin = ring_starts[r];
        const std::size_t end = r + 1 < ring_starts.size()
                                    ? ring_starts[r + 1]
                                    : coordinates.size();
        rings.push_back({coordinates.data() + begin, end - begin});
      }
      operations::splitPolygonGridPieces(rings, grid, out.polygons);
    } else {
      operations::splitLineStringGrid({coordinates.data(), coordinates.size()},
                                      grid, bounded, out.lines);
    }
    return GEOARROW_OK;
  }

  const enum GeoArrowGeometryType want;
  const GeometryType expected;
  const grid::Grid &grid;
  const bool bounded;
  BatchData &out;

  int64_t row = 0;
  int depth = 0;
  /// every coordinate of the geometry being read, rings end to end
  std::vector<geo::Coord> coordinates;
  /// where each ring begins within them
  std::vector<std::size_t> ring_starts;
  std::vector<operations::CoordSpan> rings;

public:
  /// set when a callback refused a geometry, so the caller can raise it
  std::string problem;
};

/// Split one batch of WKB geometries into pieces
void splitWkbBatch(WkbReader &reader, int64_t count, GeometryType type,
                   const grid::Grid &grid, bool bounded, int64_t parent_base,
                   BatchData &out) {
  WkbSplitter splitter(type, grid, bounded, out);
  GeoArrowVisitor visitor = splitter.visitor();
  for (int64_t i = 0; i < count; i++) {
    const std::size_t before = splitter.pieces();
    splitter.startRow(i);
    try {
      reader.visit(i, &visitor);
    } catch (const std::invalid_argument &) {
      // a refusal from one of our own callbacks says more than the error
      // geoarrow-c wraps it in
      if (!splitter.problem.empty()) {
        throw std::invalid_argument(splitter.problem);
      }
      throw;
    }
    out.parents.insert(out.parents.end(), splitter.pieces() - before,
                       parent_base + i);
  }
}

/// Splits a column holding geometries of more than one type.
///
/// Where WkbSplitter accepts one type and refuses the rest, this one takes
/// each geometry as it comes: a LineString and a Polygon are split by the
/// kernel for their type, a Point has nothing to split and goes through
/// unchanged, and a MULTI* or a GEOMETRYCOLLECTION is a container whose
/// members are each handled on their own terms. Since geoarrow-c reports a
/// container's members as nested geom_starts carrying their own types, that
/// recursion is the library's rather than ours - all this has to do is
/// notice which level it is on.
///
/// Pieces go out as WKB, which is what lets one stream carry all of them.
class MixedSplitter {
public:
  MixedSplitter(const grid::Grid &grid, bool bounded, WkbWriter &writer)
      : grid(grid), bounded(bounded), writer(writer) {}

  GeoArrowVisitor visitor() {
    GeoArrowVisitor v;
    GeoArrowVisitorInitVoid(&v);
    v.feat_start = [](GeoArrowVisitor *v) { return self(v)->featStart(); };
    v.null_feat = [](GeoArrowVisitor *v) { return self(v)->nullFeat(); };
    v.geom_start = [](GeoArrowVisitor *v, enum GeoArrowGeometryType type,
                      enum GeoArrowDimensions) {
      return self(v)->geomStart(type);
    };
    v.ring_start = [](GeoArrowVisitor *v) { return self(v)->ringStart(); };
    v.coords = [](GeoArrowVisitor *v, const GeoArrowCoordView *c) {
      return self(v)->coords(c);
    };
    v.geom_end = [](GeoArrowVisitor *v) { return self(v)->geomEnd(); };
    v.private_data = this;
    return v;
  }

  void startRow(int64_t index) { row = index; }

  /// How many pieces have been written
  std::size_t pieces() const { return written; }

  /// set when a callback refused a geometry, so the caller can raise it
  std::string problem;

private:
  static MixedSplitter *self(GeoArrowVisitor *v) {
    return static_cast<MixedSplitter *>(v->private_data);
  }

  int refuse(std::string what) {
    problem = std::move(what);
    return EINVAL;
  }

  /// One open geometry. Whether it held anything is what says an empty
  /// geometry apart from a container: a MULTILINESTRING has no coordinates
  /// of its own but does open children, while MULTILINESTRING EMPTY opens
  /// neither.
  struct Frame {
    enum GeoArrowGeometryType type = GEOARROW_GEOMETRY_TYPE_GEOMETRY;
    std::size_t coordinate_base = 0;
    std::size_t ring_base = 0;
    bool had_children = false;
  };

  int featStart() {
    open.clear();
    coordinates.clear();
    ring_offsets.assign(1, 0);
    return GEOARROW_OK;
  }

  int nullFeat() {
    return refuse("Cannot split missing (null) geometries (row " +
                  std::to_string(row) +
                  "): drop or fill null geometries first");
  }

  int geomStart(enum GeoArrowGeometryType type) {
    if (!open.empty()) {
      open.back().had_children = true;
    }
    open.push_back({type, coordinates.size(), ring_offsets.size() - 1, false});
    return GEOARROW_OK;
  }

  int ringStart() {
    ring_offsets.push_back(static_cast<int32_t>(coordinates.size()));
    if (!open.empty()) {
      open.back().had_children = true;
    }
    return GEOARROW_OK;
  }

  int coords(const GeoArrowCoordView *c) {
    return appendCoords(c, coordinates);
  }

  int geomEnd() {
    Frame frame = open.back();
    open.pop_back();
    const geo::Coord *from = coordinates.data() + frame.coordinate_base;
    const std::size_t count = coordinates.size() - frame.coordinate_base;

    try {
      if (!frame.had_children && count == 0) {
        // an empty geometry goes back as itself, whatever it was
        writer.writeEmpty(frame.type);
        written++;
      } else {
        switch (frame.type) {
        case GEOARROW_GEOMETRY_TYPE_POINT:
          if (count > 0) {
            writer.writePoint(from[0]);
            written++;
          }
          break;
        case GEOARROW_GEOMETRY_TYPE_LINESTRING:
          splitLine(from, count);
          break;
        case GEOARROW_GEOMETRY_TYPE_POLYGON:
          splitPolygon(frame);
          break;
        default:
          // a container: its members have already been dealt with
          break;
        }
      }
    } catch (const std::exception &error) {
      return refuse(std::string("Could not split the geometry at row ") +
                    std::to_string(row) + ": " + error.what());
    }

    // the geometry's own coordinates are spent
    coordinates.resize(frame.coordinate_base);
    ring_offsets.resize(frame.ring_base + 1);
    return GEOARROW_OK;
  }

  void splitLine(const geo::Coord *from, std::size_t count) {
    pieces_scratch.offsets.assign(1, 0);
    pieces_scratch.coordinates.clear();
    operations::splitLineStringGrid({from, count}, grid, bounded,
                                    pieces_scratch);
    for (std::size_t p = 0; p + 1 < pieces_scratch.offsets.size(); p++) {
      const int32_t begin = pieces_scratch.offsets[p];
      const int32_t end = pieces_scratch.offsets[p + 1];
      writer.writeLineString(pieces_scratch.coordinates.data() + begin,
                             static_cast<std::size_t>(end - begin));
      written++;
    }
  }

  void splitPolygon(const Frame &frame) {
    rings.clear();
    for (std::size_t r = frame.ring_base; r + 1 < ring_offsets.size(); r++) {
      const int32_t begin = ring_offsets[r + 1];
      const int32_t end = r + 2 < ring_offsets.size()
                              ? ring_offsets[r + 2]
                              : static_cast<int32_t>(coordinates.size());
      rings.push_back({coordinates.data() + begin,
                       static_cast<std::size_t>(end - begin)});
    }
    polygon_scratch.coordinates.clear();
    polygon_scratch.ring_offsets.assign(1, 0);
    polygon_scratch.polygon_offsets.assign(1, 0);
    operations::splitPolygonGridPieces(rings, grid, polygon_scratch);
    for (std::size_t q = 0; q + 1 < polygon_scratch.polygon_offsets.size();
         q++) {
      const int32_t first = polygon_scratch.polygon_offsets[q];
      const int32_t last = polygon_scratch.polygon_offsets[q + 1];
      writer.writePolygon(polygon_scratch.coordinates.data(),
                          polygon_scratch.ring_offsets.data() + first,
                          static_cast<std::size_t>(last - first));
      written++;
    }
  }

  const grid::Grid &grid;
  const bool bounded;
  WkbWriter &writer;

  int64_t row = 0;
  std::size_t written = 0;
  std::vector<Frame> open;
  /// every coordinate of the geometry being read, rings and members end to
  /// end; a frame owns the tail it added and drops it when it closes
  std::vector<geo::Coord> coordinates;
  std::vector<int32_t> ring_offsets{0};
  std::vector<operations::CoordSpan> rings;
  operations::LinePieces pieces_scratch;
  operations::PolygonPieces polygon_scratch;
};

/// Split one batch of geometries of any type, writing the pieces as WKB
void splitMixedBatch(WkbReader &reader, int64_t count, const grid::Grid &grid,
                     bool bounded, int64_t parent_base, BatchData &out) {
  if (!out.wkb) {
    out.wkb = std::make_unique<WkbWriter>();
  }
  MixedSplitter splitter(grid, bounded, *out.wkb);
  GeoArrowVisitor visitor = splitter.visitor();
  for (int64_t i = 0; i < count; i++) {
    const std::size_t before = splitter.pieces();
    splitter.startRow(i);
    try {
      reader.visit(i, &visitor);
    } catch (const std::invalid_argument &) {
      if (!splitter.problem.empty()) {
        throw std::invalid_argument(splitter.problem);
      }
      throw;
    }
    out.parents.insert(out.parents.end(), splitter.pieces() - before,
                       parent_base + i);
  }
}


/// Split one batch of geometries held in Arrow buffers, of the one type
void splitNativeBatch(const ArrowArrayView *geometries,
                      const ArrowSchema *schema, GeometryType type,
                      const grid::Grid &grid, bool bounded,
                      int64_t parent_base, BatchData &out) {
  if (type == GeometryType::polygon) {
    PolygonReader reader(geometries, schema);
    splitPolygonBatch(reader, grid, parent_base, out);
  } else {
    LineStringReader reader(geometries, schema);
    splitLineStringBatch(reader, grid, bounded, parent_base, out);
  }
}

} // namespace geoarrow
} // namespace snail
