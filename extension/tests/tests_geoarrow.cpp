/// Tests for the GeoArrow layer: reading arrays, splitting them, writing the
/// pieces back.
///
/// The Python suite covers this through the extension, which is the only way
/// it could be reached while it lived beside the pybind11 bindings. These go
/// at it directly, so a failure points at the Arrow code rather than at a
/// round trip through geopandas - and so the cases that are awkward to build
/// from Python (a batch whose WKB is hand-written, a schema with no
/// extension name) can be built as they are.

#include <catch2/catch.hpp>
#include <cstring>
#include <string>
#include <vector>

#include "geoarrow_arrays.hpp"

using snail::geoarrow::BatchData;
using snail::geoarrow::Encoding;
using snail::geoarrow::exportArray;
using snail::geoarrow::exportSchema;
using snail::geoarrow::GeometryType;
using snail::geoarrow::splitMixedBatch;
using snail::geoarrow::splitWkbBatch;
using snail::geoarrow::WkbReader;

namespace {

snail::grid::Grid unitGrid(int cells = 4) {
  return snail::grid::Grid(cells, cells, snail::transform::Affine());
}

/// A little-endian WKB writer, so that the tests state their input as bytes
/// rather than borrowing the code under test to produce it
struct Wkb {
  std::vector<uint8_t> bytes;

  void u8(uint8_t v) { bytes.push_back(v); }
  void u32(uint32_t v) {
    for (int i = 0; i < 4; i++) {
      bytes.push_back(static_cast<uint8_t>((v >> (8 * i)) & 0xff));
    }
  }
  void f64(double v) {
    uint8_t raw[8];
    std::memcpy(raw, &v, 8);
    bytes.insert(bytes.end(), raw, raw + 8);
  }
  /// every geometry begins with a byte order flag and a type code
  void header(uint32_t type) {
    u8(1);
    u32(type);
  }
  void point(double x, double y) {
    header(1);
    f64(x);
    f64(y);
  }
  void lineString(const std::vector<std::pair<double, double>> &points) {
    header(2);
    u32(static_cast<uint32_t>(points.size()));
    for (const auto &p : points) {
      f64(p.first);
      f64(p.second);
    }
  }
  void ring(const std::vector<std::pair<double, double>> &points) {
    u32(static_cast<uint32_t>(points.size()));
    for (const auto &p : points) {
      f64(p.first);
      f64(p.second);
    }
  }
  void emptyCollection() {
    header(7);
    u32(0);
  }
};

/// A geoarrow.wkb array over the given blobs, and a reader positioned on it
class WkbColumn {
public:
  explicit WkbColumn(const std::vector<Wkb> &geometries,
                     bool trailing_null = false) {
    ArrowSchemaInit(schema.get());
    REQUIRE(ArrowSchemaSetType(schema.get(), NANOARROW_TYPE_BINARY) ==
            NANOARROW_OK);
    nanoarrow::UniqueBuffer metadata;
    REQUIRE(ArrowMetadataBuilderInit(metadata.get(), nullptr) == NANOARROW_OK);
    REQUIRE(ArrowMetadataBuilderAppend(
                metadata.get(), ArrowCharView("ARROW:extension:name"),
                ArrowCharView("geoarrow.wkb")) == NANOARROW_OK);
    REQUIRE(ArrowSchemaSetMetadata(
                schema.get(),
                reinterpret_cast<const char *>(metadata->data)) ==
            NANOARROW_OK);

    REQUIRE(ArrowArrayInitFromSchema(array.get(), schema.get(), nullptr) ==
            NANOARROW_OK);
    REQUIRE(ArrowArrayStartAppending(array.get()) == NANOARROW_OK);
    for (const Wkb &geometry : geometries) {
      ArrowBufferView view;
      view.data.as_uint8 = geometry.bytes.data();
      view.size_bytes = static_cast<int64_t>(geometry.bytes.size());
      REQUIRE(ArrowArrayAppendBytes(array.get(), view) == NANOARROW_OK);
    }
    if (trailing_null) {
      REQUIRE(ArrowArrayAppendNull(array.get(), 1) == NANOARROW_OK);
    }
    REQUIRE(ArrowArrayFinishBuildingDefault(array.get(), nullptr) ==
            NANOARROW_OK);

    reader.init(schema.get());
    reader.setArray(array.get());
    count = array->length;
  }

  WkbReader &get() { return reader; }
  int64_t size() const { return count; }
  const ArrowSchema *arrowSchema() const { return schema.get(); }

private:
  nanoarrow::UniqueSchema schema;
  nanoarrow::UniqueArray array;
  WkbReader reader;
  int64_t count = 0;
};

/// The blobs a mixed split wrote, as bytes
std::vector<std::vector<uint8_t>> writtenPieces(BatchData &&data) {
  nanoarrow::UniqueArray array;
  exportArray(std::move(data), array.get());

  const ArrowArray *geometry = array->children[0];
  const auto *offsets = static_cast<const int32_t *>(geometry->buffers[1]);
  const auto *values = static_cast<const uint8_t *>(geometry->buffers[2]);
  std::vector<std::vector<uint8_t>> pieces;
  for (int64_t i = 0; i < geometry->length; i++) {
    pieces.emplace_back(values + offsets[i], values + offsets[i + 1]);
  }
  return pieces;
}

/// The geometry type code in a WKB blob's header
uint32_t typeOf(const std::vector<uint8_t> &blob) {
  REQUIRE(blob.size() >= 5);
  uint32_t type = 0;
  std::memcpy(&type, blob.data() + 1, 4);
  return type;
}

} // namespace

TEST_CASE("A WKB column reports itself as WKB-encoded", "[geoarrow]") {
  Wkb line;
  line.lineString({{0.5, 0.5}, {3.5, 0.5}});
  WkbColumn column({line});

  REQUIRE(snail::geoarrow::checkGeometrySchema(
              column.arrowSchema(), GeometryType::linestring) == Encoding::wkb);
  REQUIRE(snail::geoarrow::metadataValue(column.arrowSchema()->metadata,
                                         "ARROW:extension:name") ==
          "geoarrow.wkb");
  REQUIRE(snail::geoarrow::metadataValue(column.arrowSchema()->metadata,
                                         "nothing:here") == "");
}

TEST_CASE("A typed WKB split refuses a geometry of another type",
          "[geoarrow]") {
  Wkb line;
  line.lineString({{0.5, 0.5}, {3.5, 0.5}});
  Wkb point;
  point.point(2.5, 2.5);
  WkbColumn column({line, point});

  BatchData out;
  out.type = GeometryType::linestring;
  // the row index is what a caller cannot work out for itself, since a WKB
  // column does not say which of its geometries is which
  REQUIRE_THROWS_WITH(
      splitWkbBatch(column.get(), column.size(), GeometryType::linestring,
                    unitGrid(), false, 0, out),
      Catch::Contains("Point") && Catch::Contains("row 1"));
}

TEST_CASE("A mixed split takes each geometry on its own terms", "[geoarrow]") {
  Wkb point;
  point.point(2.5, 2.5);
  Wkb line;
  line.lineString({{0.5, 0.5}, {3.5, 0.5}});
  Wkb empty;
  empty.emptyCollection();
  WkbColumn column({point, line, empty});

  BatchData out;
  out.type = GeometryType::mixed;
  splitMixedBatch(column.get(), column.size(), unitGrid(), false, 0, out);

  // the point is not split, the line crosses x = 1, 2 and 3, and the empty
  // collection comes back as itself
  REQUIRE(out.size() == 6);
  REQUIRE(out.parents == std::vector<int64_t>{0, 1, 1, 1, 1, 2});

  auto pieces = writtenPieces(std::move(out));
  REQUIRE(pieces.size() == 6);
  REQUIRE(typeOf(pieces[0]) == 1); // Point
  for (std::size_t i = 1; i < 5; i++) {
    REQUIRE(typeOf(pieces[i]) == 2); // LineString
  }
  REQUIRE(typeOf(pieces[5]) == 7); // GeometryCollection
  // an empty collection is a header and a count of zero, nothing more
  REQUIRE(pieces[5].size() == 9);
}

TEST_CASE("A mixed split explodes multi-part geometries", "[geoarrow]") {
  Wkb multi;
  multi.header(5); // MultiLineString
  multi.u32(2);
  multi.lineString({{0.5, 0.5}, {2.5, 0.5}});
  multi.lineString({{0.5, 2.5}, {2.5, 2.5}});
  WkbColumn column({multi});

  BatchData out;
  out.type = GeometryType::mixed;
  splitMixedBatch(column.get(), column.size(), unitGrid(), false, 0, out);

  // three pieces per part, and both parts came out of the one row
  REQUIRE(out.size() == 6);
  REQUIRE(out.parents == std::vector<int64_t>(6, 0));
  for (const auto &piece : writtenPieces(std::move(out))) {
    REQUIRE(typeOf(piece) == 2);
  }
}

TEST_CASE("A mixed split recurses through geometry collections", "[geoarrow]") {
  Wkb nested;
  nested.header(7); // GeometryCollection
  nested.u32(1);
  nested.header(7); // holding another
  nested.u32(2);
  nested.point(1.5, 1.5);
  nested.lineString({{0.5, 0.5}, {2.5, 0.5}});
  WkbColumn column({nested});

  BatchData out;
  out.type = GeometryType::mixed;
  splitMixedBatch(column.get(), column.size(), unitGrid(), false, 0, out);

  auto pieces = writtenPieces(std::move(out));
  REQUIRE(pieces.size() == 4);
  REQUIRE(typeOf(pieces[0]) == 1); // the Point, unsplit
  for (std::size_t i = 1; i < 4; i++) {
    REQUIRE(typeOf(pieces[i]) == 2); // the LineString, in three pieces
  }
}

TEST_CASE("A mixed split splits polygons with their holes", "[geoarrow]") {
  Wkb polygon;
  polygon.header(3); // Polygon
  polygon.u32(2);    // exterior and one hole
  polygon.ring({{0.2, 0.2}, {3.8, 0.2}, {3.8, 3.8}, {0.2, 3.8}, {0.2, 0.2}});
  polygon.ring({{1.2, 1.2}, {1.2, 2.8}, {2.8, 2.8}, {2.8, 1.2}, {1.2, 1.2}});
  WkbColumn column({polygon});

  BatchData out;
  out.type = GeometryType::mixed;
  splitMixedBatch(column.get(), column.size(), unitGrid(), false, 0, out);

  REQUIRE(out.size() > 0);
  for (const auto &piece : writtenPieces(std::move(out))) {
    REQUIRE(typeOf(piece) == 3);
  }
}

TEST_CASE("The exported schema declares its encoding and no nulls",
          "[geoarrow]") {
  struct Expected {
    GeometryType type;
    const char *name;
    const char *format;
  };
  const Expected cases[] = {
      {GeometryType::linestring, "geoarrow.linestring", "+l"},
      {GeometryType::polygon, "geoarrow.polygon", "+l"},
      {GeometryType::mixed, "geoarrow.wkb", "z"},
  };

  for (const Expected &expected : cases) {
    nanoarrow::UniqueSchema schema;
    exportSchema(expected.type, schema.get());

    REQUIRE(std::string(schema->format) == "+s");
    REQUIRE(schema->n_children == 2);
    const ArrowSchema *geometry = schema->children[0];
    REQUIRE(std::string(geometry->name) == "geometry");
    REQUIRE(std::string(geometry->format) == expected.format);
    REQUIRE(snail::geoarrow::metadataValue(geometry->metadata,
                                           "ARROW:extension:name") ==
            expected.name);
    REQUIRE(std::string(schema->children[1]->name) == "parent");
    REQUIRE(std::string(schema->children[1]->format) == "l");
    // a split never produces a null, at any level
    REQUIRE((geometry->flags & ARROW_FLAG_NULLABLE) == 0);
    REQUIRE((schema->children[1]->flags & ARROW_FLAG_NULLABLE) == 0);
  }
}

TEST_CASE("A null geometry is refused rather than dropped", "[geoarrow]") {
  // Dropping it would lose the row silently, and geoarrow-c reports a null
  // through a visitor callback that does nothing unless it is given one.
  Wkb line;
  line.lineString({{0.5, 0.5}, {3.5, 0.5}});
  WkbColumn column({line}, /*trailing_null=*/true);

  BatchData out;
  out.type = GeometryType::mixed;
  REQUIRE_THROWS_WITH(
      splitMixedBatch(column.get(), column.size(), unitGrid(), false, 0, out),
      Catch::Contains("null") && Catch::Contains("row 1"));
}
