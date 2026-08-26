/// Reading GeoArrow arrays, splitting them, and writing the pieces back.
/// See geoarrow_arrays.hpp for why this is a file of its own.

#include <algorithm>
#include <array>
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

/// How the pieces of a split of this type are written back out. Coordinates
/// go back interleaved because that is what the split fills: a vector of
/// Coord is already a run of x, y, x, y doubles.
static enum GeoArrowType writtenAs(GeometryType type) {
  switch (type) {
  case GeometryType::polygon:
    return GEOARROW_TYPE_INTERLEAVED_POLYGON;
  case GeometryType::mixed:
    return GEOARROW_TYPE_WKB;
  default:
    return GEOARROW_TYPE_INTERLEAVED_LINESTRING;
  }
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

// -- Reading native GeoArrow arrays ------------------------------------------

/// Does any slot of an array, or of anything below it, hold a null?
///
/// GeoArrow asks that the arrays under a geometry hold no nulls, and a split
/// has nothing sensible to do with one, so this refuses rather than guesses.
/// geoarrow-c tracks a validity bitmap for the outermost array only, so the
/// inner levels are checked here. A producer may leave null_count at -1
/// rather than count them, hence the fall back to counting the bitmap.
static bool hasNulls(const ArrowArray *array) {
  if (array->null_count > 0) {
    return true;
  }
  if (array->null_count < 0 && array->n_buffers > 0) {
    const auto *validity = static_cast<const uint8_t *>(array->buffers[0]);
    if (validity != nullptr &&
        ArrowBitCountSet(validity, array->offset, array->length) !=
            array->length) {
      return true;
    }
  }
  for (int64_t i = 0; i < array->n_children; i++) {
    if (hasNulls(array->children[i])) {
      return true;
    }
  }
  return false;
}

/// Refuse a column whose list offsets are 64 bits wide.
///
/// The GeoArrow specification asks a reader to take either width - "
/// Implementations SHOULD accept LargeList int64 offset buffers but MAY
/// produce only List int32 offset buffers" - and geoarrow-c does not: its
/// schema parser rejects any list that is not "+l" and its array view holds
/// int32 offsets. Narrowing them here was tried and cost more code than the
/// hand-written reader it replaced, so this says so plainly instead. Nothing
/// geopandas writes is affected; an array built directly with pyarrow can be
/// a large list, and casting it is one call.
static void refuseLargeOffsets(const ArrowSchema *schema) {
  for (const ArrowSchema *at = schema; at != nullptr; at = at->n_children == 1
                                                              ? at->children[0]
                                                              : nullptr) {
    if (std::strcmp(at->format, "+L") == 0) {
      throw std::invalid_argument(
          "Expected 32-bit Arrow list offsets (\"+l\"), got 64-bit "
          "(\"+L\"): cast the geometry column before splitting");
    }
  }
}

NativeReader::NativeReader() = default;
NativeReader::~NativeReader() = default;

void NativeReader::init(const ArrowSchema *schema, GeometryType type) {
  // geoarrow-c insists on an extension name, where an array built directly
  // with pyarrow may carry none - hence InitFromStorage with the name we are
  // looking for, rather than InitFromSchema. Passing the name we want is
  // also the shape check: a column nested the wrong number of levels deep
  // for that name does not parse.
  refuseLargeOffsets(schema);
  const char *declared = extensionName(type);
  GeoArrowStringView name;
  name.data = declared;
  name.size_bytes = static_cast<int64_t>(std::strlen(declared));

  GeoArrowSchemaView schema_view;
  GeoArrowError error;
  if (GeoArrowSchemaViewInitFromStorage(&schema_view, schema, name, &error) !=
      GEOARROW_OK) {
    throw std::invalid_argument(std::string("Expected a ") + declared +
                                " array: " + error.message);
  }
  if (GeoArrowArrayViewInitFromType(&view, schema_view.type) != GEOARROW_OK) {
    throw std::invalid_argument(std::string("Could not read the ") + declared +
                                " column");
  }
}

void NativeReader::setArray(const ArrowArray *array) {
  if (hasNulls(array)) {
    throw std::invalid_argument(
        "Cannot split missing (null) geometries: drop or fill null "
        "geometries first");
  }
  GeoArrowError error;
  if (GeoArrowArrayViewSetArray(&view, array, &error) != GEOARROW_OK) {
    throw std::invalid_argument(
        std::string("Could not read a batch of geometries: ") + error.message);
  }

  // The innermost level needs the same shift offsetAt applies to the list
  // levels, and geoarrow-c leaves it to the consumer in the same way: its
  // coordinate pointers stop at the innermost double child's offset, and
  // the coordinate array's own - the fixed-size list's, or the struct's for
  // separated coordinates - is recorded in the array view for us to add.
  // Its own visitors add it per geometry; applying it once to the base
  // pointers here comes to the same thing and costs nothing per piece.
  const ArrowArray *coordinates = array;
  for (int32_t level = 0; level < view.n_offsets; level++) {
    coordinates = coordinates->children[0];
  }
  if (coordinates->offset != 0) {
    for (int32_t i = 0; i < view.coords.n_values; i++) {
      view.coords.values[i] +=
          view.coords.coords_stride * coordinates->offset;
    }
  }
}

int64_t NativeReader::length() const { return view.length[0]; }

/// Where element i of the list at `level` begins.
///
/// An Arrow list's offsets are logical indices into its child, so the
/// child's own slice offset has to be added before its offsets buffer is
/// read in turn. geoarrow-c records each level's offset in the array view
/// for exactly this, and every index into `offsets[level]` goes through
/// here so that no level can be forgotten.
int64_t NativeReader::offsetAt(int level, int64_t i) const {
  return view.offsets[level][view.offset[level] + i];
}

/// Are the coordinates laid out exactly as a run of Coord? Interleaved with
/// nothing but x and y is, which the static_assert on sizeof(Coord) pins
/// down.
bool NativeReader::contiguous() const {
  return view.coords.coords_stride == 2 &&
         view.coords.values[1] == view.coords.values[0] + 1;
}

geo::Coord NativeReader::at(int64_t vertex) const {
  const int32_t stride = view.coords.coords_stride;
  return {view.coords.values[0][vertex * stride],
          view.coords.values[1][vertex * stride]};
}

/// Vertices [begin, end) as a run the split kernels can read, pointing
/// straight into the Arrow buffer where the layout allows it and gathering
/// into a reused buffer where it does not.
operations::CoordSpan NativeReader::run(int64_t begin, int64_t end) {
  const std::size_t count = static_cast<std::size_t>(end - begin);
  if (contiguous()) {
    const auto *first =
        reinterpret_cast<const geo::Coord *>(view.coords.values[0]) + begin;
    return {first, count};
  }
  gathered.clear();
  gathered.reserve(count);
  for (int64_t v = begin; v < end; v++) {
    gathered.push_back(at(v));
  }
  return gathered;
}

operations::CoordSpan NativeReader::vertices(int64_t i) {
  return run(offsetAt(0, i), offsetAt(0, i + 1));
}

/// The ring structure is explicit in the offsets, so rings are recovered
/// exactly rather than inferred from where coordinates close back on
/// themselves.
///
/// Only one gathered ring's span is valid at a time, since they share a
/// buffer - so a polygon whose coordinates need gathering is materialised
/// ring by ring into `scratch` first, and spans taken over that.
void NativeReader::rings(int64_t i, std::vector<operations::CoordSpan> &out,
                         std::vector<linestr> &scratch) {
  out.clear();
  const int64_t first = offsetAt(0, i);
  const int64_t last = offsetAt(0, i + 1);
  if (contiguous()) {
    for (int64_t r = first; r < last; r++) {
      out.push_back(run(offsetAt(1, r), offsetAt(1, r + 1)));
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
    const int64_t begin = offsetAt(1, r);
    const int64_t end = offsetAt(1, r + 1);
    ring.reserve(static_cast<std::size_t>(end - begin));
    for (int64_t v = begin; v < end; v++) {
      ring.push_back(at(v));
    }
  }
  for (std::size_t r = 0; r < used; r++) {
    out.push_back(scratch[r]);
  }
}

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
Encoding checkGeometrySchema(const ArrowSchema *schema, GeometryType type) {
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
  return checkExtensionName(schema, type);
}

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
/// Everything from "geometry" down is geoarrow-c's to describe - the nesting,
/// the child names, the coordinate layout, and the extension metadata a
/// GeoArrow reader looks for. WKB comes out of the same call as a binary
/// column carrying the same kind of metadata, so all three encodings are one
/// line apart. nanoarrow puts the field beside the parent index in a struct.
void exportSchema(GeometryType type, ArrowSchema *out) {
  nanoarrow::UniqueSchema geometry;
  if (GeoArrowSchemaInitExtension(geometry.get(), writtenAs(type)) !=
      GEOARROW_OK) {
    throw std::runtime_error(std::string("Could not describe a column of ") +
                             extensionName(type));
  }
  NANOARROW_THROW_NOT_OK(ArrowSchemaSetName(geometry.get(), "geometry"));

  nanoarrow::UniqueSchema schema;
  ArrowSchemaInit(schema.get());
  NANOARROW_THROW_NOT_OK(ArrowSchemaSetTypeStruct(schema.get(), 2));
  // SetTypeStruct leaves each child initialised but untyped, so the one
  // geoarrow-c filled in replaces it rather than being written over
  ArrowSchemaRelease(schema->children[0]);
  geometry.move(schema->children[0]);

  NANOARROW_THROW_NOT_OK(
      ArrowSchemaSetType(schema->children[1], NANOARROW_TYPE_INT64));
  NANOARROW_THROW_NOT_OK(ArrowSchemaSetName(schema->children[1], "parent"));

  // A split never produces a null - not a null piece, ring, vertex or
  // parent index - so say so. Both libraries mark the outermost field
  // nullable by default; GeoArrow asks that a geometry's inner arrays
  // contain no nulls, and declaring it lets a reader skip looking for
  // validity bitmaps.
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

/// Hand a vector's storage to geoarrow-c as buffer `i` of the geometry
/// column, without copying it: the builder takes the vector over and frees
/// it when the array it becomes is released.
template <typename T>
static void adoptGeoBuffer(GeoArrowBuilder *builder, int64_t i,
                           std::vector<T> values) {
  auto *owned = new std::vector<T>(std::move(values));
  GeoArrowBufferView view;
  view.data = reinterpret_cast<const uint8_t *>(owned->data());
  view.size_bytes = static_cast<int64_t>(owned->size() * sizeof(T));
  const auto release = [](uint8_t *, int64_t, void *held) {
    delete static_cast<std::vector<T> *>(held);
  };
  if (GeoArrowBuilderSetOwnedBuffer(builder, i, view, release, owned) !=
      GEOARROW_OK) {
    delete owned;
    throw std::runtime_error("Could not hand a buffer to the geometry column");
  }
}

/// Build the geometry column of a native batch, giving geoarrow-c the
/// split's own buffers rather than copying them out. The column takes the
/// vectors over, so it outlives the split that made it.
///
/// The builder numbers buffers from the outside in: buffer 0 is the
/// validity bitmap, left unset because nothing a split produces is null,
/// then one offsets buffer per level of nesting, then the coordinates -
/// which is the shape LinePieces and PolygonPieces already hold. Every
/// array's length is worked out from the buffer sizes, so none is set here.
static void exportGeometryColumn(BatchData &data, ArrowArray *out) {
  GeoArrowBuilder builder{};
  if (GeoArrowBuilderInitFromType(&builder, writtenAs(data.type)) !=
      GEOARROW_OK) {
    throw std::runtime_error(std::string("Could not start a column of ") +
                             extensionName(data.type));
  }
  // adopting a buffer can throw, and the builder owns those already handed to
  // it, so it is reset however this returns
  std::unique_ptr<GeoArrowBuilder, void (*)(GeoArrowBuilder *)> owned(
      &builder, GeoArrowBuilderReset);

  int64_t buffer = 1;
  if (data.type == GeometryType::polygon) {
    // polygons over rings, then rings over vertices
    adoptGeoBuffer(&builder, buffer++,
                   std::move(data.polygons.polygon_offsets));
  }
  adoptGeoBuffer(&builder, buffer++, std::move(vertexOffsetsOf(data)));
  adoptGeoBuffer(&builder, buffer, std::move(coordinatesOf(data)));

  GeoArrowError error;
  if (GeoArrowBuilderFinish(&builder, out, &error) != GEOARROW_OK) {
    throw std::runtime_error(std::string("Could not finish a column of ") +
                             extensionName(data.type) + ": " + error.message);
  }
}

/// Build a record batch over a batch of split pieces: the geometry column
/// beside the index of the geometry each piece came from.
///
/// The geometry column arrives finished - built by geoarrow-c from the
/// split's buffers, or written by the WKB writer - and is *moved* into the
/// struct, a bitwise copy then the source's release callback nulled without
/// calling it, which is how Arrow says an array changes hands. So nothing is
/// copied here either.
void exportArray(BatchData data, ArrowArray *out) {
  nanoarrow::UniqueArray geometry;
  if (data.type == GeometryType::mixed) {
    data.wkb->finish(geometry.get());
  } else {
    exportGeometryColumn(data, geometry.get());
  }

  // the two columns are filled independently - the pieces by the split, the
  // parent indices beside them - so this is where a disagreement would show
  const int64_t pieces = data.size();
  if (geometry->length != pieces) {
    throw std::runtime_error("Wrote " + std::to_string(geometry->length) +
                             " pieces but recorded " + std::to_string(pieces) +
                             " parents");
  }

  nanoarrow::UniqueArray array;
  NANOARROW_THROW_NOT_OK(
      ArrowArrayInitFromType(array.get(), NANOARROW_TYPE_STRUCT));
  NANOARROW_THROW_NOT_OK(ArrowArrayAllocateChildren(array.get(), 2));
  geometry.move(array->children[0]);

  ArrowArray *parent = array->children[1];
  NANOARROW_THROW_NOT_OK(ArrowArrayInitFromType(parent, NANOARROW_TYPE_INT64));
  parent->length = pieces;
  adoptBuffer(parent, 1, std::move(data.parents));

  array->length = pieces;
  // MINIMAL checks every buffer against the length we just set; it stops
  // short of the level that would try to reallocate the buffers adopted here
  NANOARROW_THROW_NOT_OK(ArrowArrayFinishBuilding(
      array.get(), NANOARROW_VALIDATION_LEVEL_MINIMAL, nullptr));

  array.move(out);
}

// -- Splitting a stream ------------------------------------------------------

/// Split one batch of linestrings into pieces
static void splitLineStringBatch(NativeReader &reader, int64_t count,
                                 const grid::Grid &grid, bool bounded,
                                 int64_t parent_base, BatchData &out) {
  for (int64_t l = 0; l < count; l++) {
    const std::size_t before = out.lines.size();
    operations::splitLineStringGrid(reader.vertices(l), grid, bounded,
                                    out.lines);
    // one parent per piece this line produced; a line that fell apart into
    // nothing produces none
    out.parents.insert(out.parents.end(), out.lines.size() - before,
                       parent_base + l);
  }
}

/// Split one batch of polygons into pieces
static void splitPolygonBatch(NativeReader &reader, int64_t count,
                              const grid::Grid &grid, int64_t parent_base,
                              BatchData &out) {
  std::vector<operations::CoordSpan> rings;
  std::vector<linestr> scratch;
  for (int64_t p = 0; p < count; p++) {
    reader.rings(p, rings, scratch);
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


void splitNativeBatch(NativeReader &reader, int64_t count, GeometryType type,
                      const grid::Grid &grid, bool bounded,
                      int64_t parent_base, BatchData &out) {
  if (type == GeometryType::polygon) {
    splitPolygonBatch(reader, count, grid, parent_base, out);
  } else {
    splitLineStringBatch(reader, count, grid, bounded, parent_base, out);
  }
}

} // namespace geoarrow
} // namespace snail
