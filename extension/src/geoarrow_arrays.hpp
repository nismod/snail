#ifndef GEOARROW_ARRAYS_H
#define GEOARROW_ARRAYS_H

/// Reading GeoArrow arrays, splitting them, and writing the pieces back.
///
/// Everything here works on Arrow structures and knows nothing about
/// Python: the source of the batches, the GIL, and the stream handed back
/// over the PyCapsule interface are all geoarrow.cpp's. Keeping the line
/// there is what lets this half be built and tested on its own - the Catch2
/// suite compiles it directly, which the Arrow code had no way to do while
/// it lived in the same file as the pybind11 bindings.

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

// Vendored, and namespaced to Snail so that loading this extension alongside
// pyarrow - or any other wheel that vendors them - cannot collide. nanoarrow
// brings the Arrow C data and stream interface structs with it; geoarrow-c
// decodes WKB and walks any GeoArrow encoding through a visitor.
#include "nanoarrow.hpp"

#include "geoarrow/geoarrow.h"

#include "geometry.hpp"
#include "grid.hpp"
#include "operations.hpp"

namespace snail {
namespace geoarrow {

namespace geo = geometry;

using linestr = std::vector<geo::Coord>;

static_assert(sizeof(geo::Coord) == 2 * sizeof(double),
              "Coord must be a bare pair of doubles to alias Arrow's "
              "interleaved coordinate buffer");

/// What a split produces.
///
/// The typed splits give back GeoArrow of one type, with the coordinates in
/// Arrow buffers. A mixed split cannot: an Arrow stream has one schema for
/// every batch, so pieces of several types need one encoding that carries
/// any of them, and that is WKB.
enum class GeometryType { linestring, polygon, mixed };

const char *extensionName(GeometryType type);

/// How a source holds its geometries.
///
/// GeoArrow's native encodings put the coordinates in Arrow buffers, which
/// are read in place. Its serialised WKB encoding puts each geometry in a
/// binary blob instead, which has to be decoded - and only says what type a
/// geometry is once decoded, per feature rather than per column.
enum class Encoding { native, wkb };

/// Read a value out of an ArrowSchema metadata blob. GeoArrow declares a
/// geometry type there, under "ARROW:extension:name". Returns an empty
/// string if the key is absent, or if there is no metadata at all.
std::string metadataValue(const char *metadata, const char *key);

/// Validate that a schema describes the expected GeoArrow geometry type,
/// before any data has arrived, and say how it is encoded.
///
/// Only the native encodings can be checked this far ahead: a WKB column
/// says nothing about what its geometries are until they are decoded, so
/// the type check for those happens per feature as they are read.
Encoding checkGeometrySchema(const ArrowSchema *schema, GeometryType type);

/// A geoarrow.wkb batch, read through geoarrow-c.
///
/// WKB is a serialised encoding: the coordinates sit inside each blob, so
/// unlike the native encodings they cannot be pointed at where they lie and
/// have to be decoded. geoarrow-c does that, handing each feature to a
/// visitor - which is also the only place a WKB geometry's type is known,
/// since the column as a whole does not declare one.
class WkbReader {
public:
  WkbReader() = default;
  WkbReader(const WkbReader &) = delete;
  WkbReader &operator=(const WkbReader &) = delete;
  ~WkbReader();

  /// Lay the reader out from the column's schema, once
  void init(const ArrowSchema *schema);

  /// Point it at a batch
  void setArray(const ArrowArray *array);

  /// Walk feature i, handing its parts to the visitor
  void visit(int64_t i, GeoArrowVisitor *visitor);

private:
  GeoArrowArrayReader reader{};
  bool ready = false;
};

/// A batch of native GeoArrow geometries - the coordinates in Arrow buffers
/// rather than serialised - read through geoarrow-c.
///
/// Its array view collapses the layouts the format allows: interleaved
/// coordinates and separated x and y arrays, over any number of dimensions.
/// Splitting is planar, so a z or m ordinate is stepped over rather than
/// refused. It cannot read 64-bit ("+L") list offsets - see
/// refuseLargeOffsets in the implementation for why those are turned away.
class NativeReader {
public:
  NativeReader();
  NativeReader(const NativeReader &) = delete;
  NativeReader &operator=(const NativeReader &) = delete;
  ~NativeReader();

  /// Lay the reader out from the column's schema, once, reading it as the
  /// given geometry type - an array built directly with pyarrow may declare
  /// no extension name of its own, so it is told which to expect.
  void init(const ArrowSchema *schema, GeometryType type);

  /// Point it at a batch
  void setArray(const ArrowArray *array);

  /// How many geometries the current batch holds
  int64_t length() const;

  /// The vertices of linestring i
  operations::CoordSpan vertices(int64_t i);

  /// The rings of polygon i, exterior first
  void rings(int64_t i, std::vector<operations::CoordSpan> &out,
             std::vector<linestr> &scratch);

private:
  int64_t offsetAt(int level, int64_t i) const;
  bool contiguous() const;
  geo::Coord at(int64_t vertex) const;
  operations::CoordSpan run(int64_t begin, int64_t end);

  GeoArrowArrayView view{};
  std::vector<geo::Coord> gathered;
};

class WkbWriter;

/// One batch of split pieces: the geometries, and for each piece the index
/// of the geometry it was split from.
struct BatchData {
  BatchData();
  BatchData(BatchData &&) noexcept;
  BatchData &operator=(BatchData &&) noexcept;
  ~BatchData();

  GeometryType type = GeometryType::linestring;
  /// The split kernels append straight into these, one geometry after
  /// another, so a batch's buffers are the ones the kernel filled. Nothing
  /// is concatenated on the way out, and no offset is rebased: the kernels
  /// record where a piece starts in the buffer they are filling, which for
  /// a whole batch is already the offset Arrow wants. Only the one matching
  /// `type` is used.
  operations::LinePieces lines;
  operations::PolygonPieces polygons;
  /// where a mixed split's pieces go instead: one WKB blob each, since an
  /// Arrow stream has one schema and only WKB can carry every type
  std::unique_ptr<WkbWriter> wkb;
  /// the geometry each piece came from, indexed across the whole stream
  std::vector<int64_t> parents;

  int64_t size() const { return static_cast<int64_t>(parents.size()); }

  /// How many coordinates the pieces hold; zero for a mixed batch, whose
  /// coordinates are inside the blobs
  std::size_t coordinateCount() const;
};

/// Build the schema of the split stream: record batches of a geometry
/// column and the index of the geometry each piece came from
void exportSchema(GeometryType type, ArrowSchema *out);

/// Build a record batch over a batch of split pieces, giving Arrow the
/// split's own buffers rather than copying them out
void exportArray(BatchData data, ArrowArray *out);

/// Split one batch of geometries held in Arrow buffers, of the one type
void splitNativeBatch(NativeReader &reader, int64_t count, GeometryType type,
                      const grid::Grid &grid, bool bounded,
                      int64_t parent_base, BatchData &out);

/// Split one batch of WKB geometries, refusing any that is not `type`
void splitWkbBatch(WkbReader &reader, int64_t count, GeometryType type,
                   const grid::Grid &grid, bool bounded, int64_t parent_base,
                   BatchData &out);

/// Split one batch of geometries of any type, writing the pieces as WKB.
/// Sets up `out`'s writer, so a caller need never name it - which is what
/// keeps WkbWriter out of this header.
void splitMixedBatch(WkbReader &reader, int64_t count, const grid::Grid &grid,
                     bool bounded, int64_t parent_base, BatchData &out);

} // namespace geoarrow
} // namespace snail

#endif // GEOARROW_ARRAYS_H
