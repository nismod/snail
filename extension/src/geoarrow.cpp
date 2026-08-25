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
/// This file is the Python side of that: where the batches come from, who
/// holds the GIL, and the stream handed back over the PyCapsule interface.
/// Reading the Arrow arrays, splitting them and writing the pieces are in
/// geoarrow_arrays.cpp, which knows nothing about Python.
///
/// References:
/// - https://arrow.apache.org/docs/format/CDataInterface.html
/// - https://arrow.apache.org/docs/format/CStreamInterface.html
/// - https://arrow.apache.org/docs/format/CDataInterface/PyCapsuleInterface.html
/// - https://geoarrow.org/format.html

#include <cerrno>
#include <cstdint>
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

#include "geoarrow.hpp"
#include "geoarrow_arrays.hpp"
#include "grid.hpp"
#include "transform.hpp"

namespace snail {
namespace geoarrow {

namespace py = pybind11;

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
      encoding = checkGeometrySchema(schema.get()->children[geometry_child], type);
    } else {
      encoding = checkGeometrySchema(schema.get(), type);
    }

    // One reader over the geometry column, laid out from its schema now and
    // pointed at each batch as it arrives. Which reader depends on how the
    // column is encoded: the native encodings are read in place through an
    // Arrow view, WKB is decoded by geoarrow-c.
    if (encoding == Encoding::wkb) {
      wkb.init(geometrySchema());
      return;
    }
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

  /// How this source holds its geometries
  Encoding source() const { return encoding; }

  /// The Arrow view of the current batch's geometries; native encodings only
  const ArrowArrayView *geometryView() { return view.get(); }

  /// The WKB reader positioned on the current batch; WKB sources only
  WkbReader &wkbReader() { return wkb; }

  /// How many geometries the current batch holds
  int64_t length() const { return batch_length; }

  /// Pull the next batch and point the reader at its geometries, returning
  /// false once the source is exhausted. The batch owns what the reader
  /// reads, so it has to outlive the reading.
  bool next(nanoarrow::UniqueArray &batch) {
    batch.reset();
    if (single->release != nullptr) {
      // a one-batch source: hand the array over, emptying it so that the
      // next call reports the end of the stream
      single.move(batch.get());
    } else if (stream->release == nullptr) {
      return false;
    } else {
      if (stream.get()->get_next(stream.get(), batch.get()) != 0) {
        throw std::runtime_error(
            std::string("Could not read the next batch of geometries: ") +
            lastError());
      }
      // the producer marks the end of the stream with a released array
      if (batch->release == nullptr) {
        return false;
      }
    }

    const ArrowArray *geometries = geometryArray(batch.get());
    batch_length = geometries->length;
    if (encoding == Encoding::wkb) {
      wkb.setArray(geometries);
      return true;
    }
    // Pointing the view at the batch also checks it over: that its children
    // and buffers are the shape the schema promised, and that its offsets
    // stay inside the arrays they index. Reading below can then trust it.
    ArrowError error;
    ArrowErrorInit(&error);
    if (ArrowArrayViewSetArray(view.get(), geometries, &error) !=
        NANOARROW_OK) {
      throw std::invalid_argument(
          std::string("Could not read a batch of geometries: ") +
          error.message);
    }
    return true;
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
  WkbReader wkb;
  Encoding encoding = Encoding::native;
  int64_t batch_length = 0;
  nanoarrow::UniqueSchema schema;
  /// which column of a record batch holds the geometries; -1 when the
  /// batches are the geometries themselves
  int64_t geometry_child = -1;
};

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
    if (!state->input.next(batch)) {
      return std::nullopt;
    }

    BatchData out;
    out.type = state->type;
    const int64_t count = state->input.length();
    // The GIL is held to read the source, which may be a Python object, and
    // given back around the splitting, which is not.
    if (state->type == GeometryType::mixed) {
      py::gil_scoped_release unlocked;
      splitMixedBatch(state->input.wkbReader(), count, state->grid,
                      state->bounded, state->parent_base, out);
    } else if (state->input.source() == Encoding::wkb) {
      py::gil_scoped_release unlocked;
      splitWkbBatch(state->input.wkbReader(), count, state->type, state->grid,
                    state->bounded, state->parent_base, out);
    } else {
      const ArrowArrayView *geometries = state->input.geometryView();
      const ArrowSchema *schema = state->input.geometrySchema();
      py::gil_scoped_release unlocked;
      splitNativeBatch(geometries, schema, state->type, state->grid,
                       state->bounded, state->parent_base, out);
    }
    state->parent_base += count;

    if (out.coordinateCount() >
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

static SplitStream splitGeometries(const py::object &geometries, int nrows,
                                   int ncols, std::vector<double> transform,
                                   bool bounded) {
  return {geometries, GeometryType::mixed, makeGrid(nrows, ncols, transform),
          bounded};
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

  m.def("split_geometries", &splitGeometries, py::arg("geometries"),
        py::arg("nrows"), py::arg("ncols"), py::arg("transform"),
        py::arg("bounded") = false,
        R"(Split geometries of any type along a grid.

Takes geometries of any GeoArrow encoding, including geoarrow.wkb and
multi-part types, from any object supporting the Arrow PyCapsule interface.
Unlike split_linestrings and split_polygons it does not require the column to
hold a single geometry type, so it can split a layer of mixed geometries -
which only WKB can carry, since geopandas will not write a mixed frame as
geoarrow-encoded at all.

Each geometry is handled on its own terms: LineStrings and Polygons are
split, Points pass through unchanged, multi-part geometries are split part by
part, and a GeometryCollection is split member by member. Every piece is
attributed to the row it came from, whatever it came out of. An empty
geometry comes back as itself.

Returns a SplitStream whose pieces are geoarrow.wkb, the one encoding that
can carry every type in a single Arrow stream. The typed functions give back
GeoArrow with the coordinates in Arrow buffers, and are cheaper where the
column really does hold one type. Nothing is split until the stream is
read.)");

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
