"""Tests for GeoArrow data exchange with the C++ extension (issue #14)

Geometries cross into the extension as a stream of GeoArrow batches and the
split pieces come back as a stream of record batches. These tests cover that
interface: which sources can be split, how the stream behaves, and how the
pieces come back. That the splits themselves are right is covered in
test_intersection.py.
"""

import geopandas as gpd
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from numpy.testing import assert_array_equal
from shapely.geometry import LineString, Polygon
from snail.core.intersections import split_linestring as core_split_linestring
from snail.core.intersections import split_linestrings as core_split_linestrings
from snail.core.intersections import split_polygon as core_split_polygon
from snail.core.intersections import split_polygons as core_split_polygons

from snail.intersection import read_split_stream, to_geoarrow

NROWS = NCOLS = 4
TRANSFORM = (1, 0, 0, 0, 1, 0)


def batches_of(stream):
    """Every record batch of a split stream"""
    reader = pa.RecordBatchReader._import_from_c_capsule(stream.__arrow_c_stream__())
    return list(reader)


def geometry_of(stream):
    """The split pieces of a stream, as shapely geometries"""
    return read_split_stream(stream)[0]


def assert_same_pieces(actual, expected):
    """Two splits produced the same pieces, in the same order"""
    assert len(actual) == len(expected)
    for piece, want in zip(actual, expected):
        assert piece.equals_exact(want, 1e-12)


def vertices_type():
    """fixed_size_list<xy: double>[2] - interleaved 2D coordinates"""
    return pa.list_(pa.field("xy", pa.float64(), nullable=False), 2)


def large_list_linestring_type():
    """A geoarrow.linestring shape carrying 64-bit ("+L") list offsets"""
    return pa.large_list(pa.field("vertices", vertices_type(), nullable=False))


def large_list_polygon_type():
    """A geoarrow.polygon shape carrying 64-bit ("+L") offsets at both levels"""
    return pa.large_list(
        pa.field(
            "rings",
            pa.large_list(pa.field("vertices", vertices_type(), nullable=False)),
            nullable=False,
        )
    )


def assert_not_nullable(data_type):
    """No field anywhere below this type admits a null. Nested types report
    their children through num_fields/field(i); leaves report none."""
    for i in range(data_type.num_fields):
        field = data_type.field(i)
        assert not field.nullable, f"{field.name} should not be nullable"
        assert_not_nullable(field.type)


def as_nested_lists(geometries):
    """Geometries as the plain nested lists pyarrow.array builds from, so that
    an array of any list type can be constructed from them"""
    if geometries.geom_type.iloc[0] == "Polygon":
        return [
            [[list(xy) for xy in ring.coords] for ring in [p.exterior, *p.interiors]]
            for p in geometries
        ]
    return [[list(xy) for xy in line.coords] for line in geometries]


def with_z(geometries, height=100.0):
    """The same geometries with a z ordinate added to every vertex"""
    if geometries.geom_type.iloc[0] == "Polygon":
        return gpd.GeoSeries(
            [
                Polygon(
                    [(x, y, height) for x, y in p.exterior.coords],
                    [[(x, y, height) for x, y in r.coords] for r in p.interiors],
                )
                for p in geometries
            ]
        )
    return gpd.GeoSeries(
        [LineString([(x, y, height) for x, y in line.coords]) for line in geometries]
    )


@pytest.fixture
def linestrings():
    return gpd.GeoSeries(
        [
            LineString([(0.5, 0.5), (0.75, 0.5), (1.5, 0.5), (1.5, 1.5)]),
            LineString([(0.5, 0.5), (0.75, 0.5), (1.5, 1.5)]),
        ]
    )


@pytest.fixture
def polygons():
    return gpd.GeoSeries(
        [
            Polygon([(0.5, 0.5), (2.5, 0.5), (2.5, 2.5), (0.5, 2.5)]),
            # a polygon with a hole: its rings must survive the round trip
            Polygon(
                [(0.2, 0.2), (3.8, 0.2), (3.8, 3.8), (0.2, 3.8)],
                [[(1.2, 1.2), (1.2, 2.8), (2.8, 2.8), (2.8, 1.2)]],
            ),
        ]
    )


@pytest.fixture
def many_linestrings():
    return gpd.GeoSeries(
        [LineString([(i + 0.5, 0.5), (i + 0.5, 3.5)]) for i in range(6)]
    )


class TestSourceKinds:
    """Any Arrow source of geometries can be split, however it is batched"""

    def sources(self, geometries):
        array = pa.array(geometries.to_arrow(geometry_encoding="geoarrow"))
        frame = gpd.GeoDataFrame({"a": range(len(geometries))}, geometry=geometries)
        return {
            # a single array: exposes only __arrow_c_array__
            "GeoSeries.to_arrow": geometries.to_arrow(geometry_encoding="geoarrow"),
            "pyarrow.Array": array,
            # batched sources: expose only __arrow_c_stream__
            "pyarrow.ChunkedArray": pa.chunked_array(
                [array.slice(i, 1) for i in range(len(array))]
            ),
            "pyarrow.Table": pa.table({"geometry": array}),
            "GeoDataFrame.to_arrow": frame.to_arrow(geometry_encoding="geoarrow"),
            "snail to_geoarrow": to_geoarrow(geometries),
        }

    def test_batched_sources_offer_no_array_interface(self, linestrings):
        """The reason for consuming a stream: the batched sources cannot be
        read through __arrow_c_array__ at all"""
        sources = self.sources(linestrings)

        for name in ("pyarrow.ChunkedArray", "pyarrow.Table", "GeoDataFrame.to_arrow"):
            assert not hasattr(sources[name], "__arrow_c_array__"), name
            assert hasattr(sources[name], "__arrow_c_stream__"), name
        for name in ("GeoSeries.to_arrow", "pyarrow.Array"):
            assert hasattr(sources[name], "__arrow_c_array__"), name

    def test_linestring_sources_agree(self, linestrings):
        expected = None
        for name, source in self.sources(linestrings).items():
            actual = geometry_of(
                core_split_linestrings(source, NROWS, NCOLS, TRANSFORM)
            )
            if expected is None:
                expected = actual
                assert len(expected) == 6
            assert len(actual) == len(expected), name
            for piece, want in zip(actual, expected):
                assert piece.equals_exact(want, 1e-12), name

    def test_polygon_sources_agree(self, polygons):
        expected = None
        for name, source in self.sources(polygons).items():
            actual = geometry_of(core_split_polygons(source, NROWS, NCOLS, TRANSFORM))
            if expected is None:
                expected = actual
                assert len(expected) == 25
            assert len(actual) == len(expected), name
            assert sorted(round(p.area, 9) for p in actual) == sorted(
                round(p.area, 9) for p in expected
            ), name

    def test_record_batch_reader(self, many_linestrings):
        """A reader that only yields batches once, as an external source
        would - it has no array interface and cannot be rewound"""
        array = pa.array(many_linestrings.to_arrow(geometry_encoding="geoarrow"))
        schema = pa.schema([pa.field("geometry", array.type)])
        reader = pa.RecordBatchReader.from_batches(
            schema,
            (
                pa.RecordBatch.from_arrays([array.slice(i, 2)], schema=schema)
                for i in range(0, len(array), 2)
            ),
        )
        assert not hasattr(reader, "__arrow_c_array__")

        pieces = geometry_of(core_split_linestrings(reader, NROWS, NCOLS, TRANSFORM))
        assert len(pieces) == 24

    def test_three_dimensional_coordinates(self, many_linestrings):
        """Interleaved coordinates may carry a z (or m) alongside x and y,
        which widens the step from one vertex to the next. Splitting is
        planar, so the pieces must match those of the 2D geometries."""
        arrow = with_z(many_linestrings).to_arrow(
            geometry_encoding="geoarrow", include_z=True
        )
        field = pa.Field._import_from_c_capsule(arrow.__arrow_c_array__()[0])
        assert field.type.value_type.list_size == 3

        actual = geometry_of(core_split_linestrings(arrow, NROWS, NCOLS, TRANSFORM))
        expected = geometry_of(
            core_split_linestrings(
                to_geoarrow(many_linestrings), NROWS, NCOLS, TRANSFORM
            )
        )

        assert_same_pieces(actual, expected)

    def test_geoparquet_with_separated_coordinates(self, tmp_path, many_linestrings):
        """GeoParquet stores coordinates separated into x and y columns
        rather than interleaved, and must split to the same pieces"""
        frame = gpd.GeoDataFrame(
            {"a": range(len(many_linestrings))},
            geometry=many_linestrings,
            crs="EPSG:4326",
        )
        path = tmp_path / "lines.parquet"
        frame.to_parquet(path, geometry_encoding="geoarrow")
        parquet = pq.ParquetFile(path)
        assert pa.types.is_struct(
            parquet.schema_arrow.field("geometry").type.value_type
        )

        reader = pa.RecordBatchReader.from_batches(
            parquet.schema_arrow, parquet.iter_batches(batch_size=2)
        )
        actual = geometry_of(core_split_linestrings(reader, NROWS, NCOLS, TRANSFORM))
        expected = geometry_of(
            core_split_linestrings(
                to_geoarrow(many_linestrings), NROWS, NCOLS, TRANSFORM
            )
        )

        assert_same_pieces(actual, expected)

    def test_three_dimensional_polygons(self, polygons):
        """As for linestrings: a z ordinate widens the step from one vertex to
        the next, and splitting is planar, so the pieces must be unchanged"""
        arrow = with_z(polygons).to_arrow(geometry_encoding="geoarrow", include_z=True)
        field = pa.Field._import_from_c_capsule(arrow.__arrow_c_array__()[0])
        assert field.type.value_type.value_type.list_size == 3

        actual = geometry_of(core_split_polygons(arrow, NROWS, NCOLS, TRANSFORM))
        expected = geometry_of(
            core_split_polygons(to_geoarrow(polygons), NROWS, NCOLS, TRANSFORM)
        )

        assert_same_pieces(actual, expected)

    def test_geoparquet_polygons_with_separated_coordinates(self, tmp_path, polygons):
        """The separated-coordinate layout again, but nested one level deeper:
        a polygon's rings each hold their own run of x and y"""
        frame = gpd.GeoDataFrame({"a": range(len(polygons))}, geometry=polygons)
        path = tmp_path / "polygons.parquet"
        frame.to_parquet(path, geometry_encoding="geoarrow")
        parquet = pq.ParquetFile(path)
        rings = parquet.schema_arrow.field("geometry").type.value_type
        assert pa.types.is_struct(rings.value_type)

        reader = pa.RecordBatchReader.from_batches(
            parquet.schema_arrow, parquet.iter_batches(batch_size=1)
        )
        actual = geometry_of(core_split_polygons(reader, NROWS, NCOLS, TRANSFORM))
        expected = geometry_of(
            core_split_polygons(to_geoarrow(polygons), NROWS, NCOLS, TRANSFORM)
        )

        assert_same_pieces(actual, expected)

    def test_large_list_offsets_linestrings(self, many_linestrings):
        """Arrow lists may carry 64-bit offsets ("+L") instead of 32-bit.
        geopandas never emits them, but an array built with pyarrow can, and
        the reader dispatches on both."""
        arrow = pa.array(
            as_nested_lists(many_linestrings), type=large_list_linestring_type()
        )
        assert pa.types.is_large_list(arrow.type)

        actual = geometry_of(core_split_linestrings(arrow, NROWS, NCOLS, TRANSFORM))
        expected = geometry_of(
            core_split_linestrings(
                to_geoarrow(many_linestrings), NROWS, NCOLS, TRANSFORM
            )
        )

        assert_same_pieces(actual, expected)

    def test_large_list_offsets_polygons(self, polygons):
        """64-bit offsets at both levels of a polygon's nesting"""
        arrow = pa.array(as_nested_lists(polygons), type=large_list_polygon_type())
        assert pa.types.is_large_list(arrow.type)
        assert pa.types.is_large_list(arrow.type.value_type)

        actual = geometry_of(core_split_polygons(arrow, NROWS, NCOLS, TRANSFORM))
        expected = geometry_of(
            core_split_polygons(to_geoarrow(polygons), NROWS, NCOLS, TRANSFORM)
        )

        assert_same_pieces(actual, expected)

    def test_sliced_arrays(self, many_linestrings, polygons):
        """A slice shares the buffers of the array it came from and records
        where in them it starts. That offset shifts which offsets describe an
        element, so a slice must split to the pieces of its geometries alone
        and not of the ones before it."""
        for geometries, split in (
            (many_linestrings, core_split_linestrings),
            (polygons, core_split_polygons),
        ):
            array = pa.array(geometries.to_arrow(geometry_encoding="geoarrow"))
            sliced = array.slice(1, len(array) - 1)
            assert sliced.offset == 1

            actual = geometry_of(split(sliced, NROWS, NCOLS, TRANSFORM))
            expected = geometry_of(
                split(to_geoarrow(geometries[1:]), NROWS, NCOLS, TRANSFORM)
            )

            assert_same_pieces(actual, expected)


class TestStream:
    def test_nothing_is_split_until_the_stream_is_read(self, many_linestrings):
        """The split runs as the pieces are read, pulling one source batch
        at a time"""
        array = pa.array(many_linestrings.to_arrow(geometry_encoding="geoarrow"))
        schema = pa.schema([pa.field("geometry", array.type)])
        pulled = []

        def source_batches():
            for i in range(len(array)):
                pulled.append(i)
                yield pa.RecordBatch.from_arrays([array.slice(i, 1)], schema=schema)

        stream = core_split_linestrings(
            pa.RecordBatchReader.from_batches(schema, source_batches()),
            NROWS,
            NCOLS,
            TRANSFORM,
        )
        assert pulled == []

        reader = pa.RecordBatchReader._import_from_c_capsule(
            stream.__arrow_c_stream__()
        )
        assert pulled == []

        pieces = iter(reader)
        next(pieces)
        assert pulled == [0]
        next(pieces)
        assert pulled == [0, 1]
        list(pieces)
        assert pulled == [0, 1, 2, 3, 4, 5]

    def test_one_result_batch_per_source_batch(self, many_linestrings):
        array = pa.array(many_linestrings.to_arrow(geometry_encoding="geoarrow"))
        source = pa.chunked_array([array.slice(i, 2) for i in range(0, len(array), 2)])

        batches = batches_of(core_split_linestrings(source, NROWS, NCOLS, TRANSFORM))

        assert source.num_chunks == 3
        assert len(batches) == 3

    def test_parents_index_the_whole_source(self, many_linestrings):
        """A piece's parent indexes the source, not the batch it was in"""
        array = pa.array(many_linestrings.to_arrow(geometry_encoding="geoarrow"))
        source = pa.chunked_array([array.slice(i, 2) for i in range(0, len(array), 2)])

        _, parent = read_split_stream(
            core_split_linestrings(source, NROWS, NCOLS, TRANSFORM)
        )

        assert_array_equal(np.unique(parent), [0, 1, 2, 3, 4, 5])
        assert_array_equal(np.bincount(parent), [4, 4, 4, 4, 4, 4])

    def test_batching_does_not_change_the_pieces(self, many_linestrings):
        """Where the source batch boundaries fall makes no difference"""
        array = pa.array(many_linestrings.to_arrow(geometry_encoding="geoarrow"))
        results = []
        for size in (1, 2, 4, len(array)):
            source = pa.chunked_array(
                [array.slice(i, size) for i in range(0, len(array), size)]
            )
            results.append(
                read_split_stream(
                    core_split_linestrings(source, NROWS, NCOLS, TRANSFORM)
                )
            )

        first_geometry, first_parent = results[0]
        for geometry, parent in results[1:]:
            assert len(geometry) == len(first_geometry)
            for piece, want in zip(geometry, first_geometry):
                assert piece.equals_exact(want, 1e-12)
            assert_array_equal(parent, first_parent)

    def test_stream_is_consumed_once(self, linestrings):
        """Arrow streams are one-shot: reading the pieces takes them"""
        stream = core_split_linestrings(
            to_geoarrow(linestrings), NROWS, NCOLS, TRANSFORM
        )
        assert len(batches_of(stream)) == 1

        with pytest.raises(ValueError, match="already been read"):
            stream.__arrow_c_stream__()

    def test_result_schema(self, linestrings, polygons):
        """Each batch carries the GeoArrow geometry and the parent index"""
        for geometries, split, extension in (
            (linestrings, core_split_linestrings, b"geoarrow.linestring"),
            (polygons, core_split_polygons, b"geoarrow.polygon"),
        ):
            stream = split(to_geoarrow(geometries), NROWS, NCOLS, TRANSFORM)
            assert stream.geometry_type == extension.decode()

            reader = pa.RecordBatchReader._import_from_c_capsule(
                stream.__arrow_c_stream__()
            )
            schema = reader.schema
            assert schema.names == ["geometry", "parent"]
            assert (
                schema.field("geometry").metadata[b"ARROW:extension:name"] == extension
            )
            assert schema.field("parent").type == pa.int64()
            # A split never produces a null piece, and GeoArrow asks that the
            # arrays under a geometry hold no nulls either, so nothing in the
            # exported schema is nullable
            assert not schema.field("geometry").nullable
            assert not schema.field("parent").nullable
            assert_not_nullable(schema.field("geometry").type)
            list(reader)

    def test_pieces_outlive_the_stream(self, linestrings):
        """A batch owns its buffers, so it survives the stream closing"""
        batches = batches_of(
            core_split_linestrings(to_geoarrow(linestrings), NROWS, NCOLS, TRANSFORM)
        )
        table = pa.Table.from_batches(batches)
        expected = table.to_pylist()
        del batches

        assert table.to_pylist() == expected

    def test_empty_source(self):
        empty = pa.chunked_array(
            [],
            type=pa.list_(
                pa.field(
                    "vertices",
                    pa.list_(pa.field("xy", pa.float64(), nullable=False), 2),
                    nullable=False,
                )
            ),
        )
        geometry, parent = read_split_stream(
            core_split_linestrings(empty, NROWS, NCOLS, TRANSFORM)
        )

        assert len(geometry) == 0
        assert len(parent) == 0


class TestSplits:
    def test_linestrings_match_single_geometry_split(self, linestrings):
        actual = geometry_of(
            core_split_linestrings(to_geoarrow(linestrings), NROWS, NCOLS, TRANSFORM)
        )

        expected = []
        for geometry in linestrings:
            expected.extend(core_split_linestring(geometry, NROWS, NCOLS, TRANSFORM))
        assert len(actual) == len(expected)
        for piece, want in zip(actual, expected):
            assert piece.equals_exact(want, 1e-12)

    def test_polygons_match_single_geometry_split(self, polygons):
        actual = geometry_of(
            core_split_polygons(to_geoarrow(polygons), NROWS, NCOLS, TRANSFORM)
        )

        expected = []
        for geometry in polygons:
            expected.extend(core_split_polygon(geometry, NROWS, NCOLS, TRANSFORM))
        assert len(actual) == len(expected)
        assert sorted(round(p.area, 9) for p in actual) == sorted(
            round(p.area, 9) for p in expected
        )

    def test_polygon_pieces_are_valid_and_conserve_area(self, polygons):
        """Splitting cuts a polygon up without losing or duplicating area,
        holes included"""
        pieces = gpd.GeoSeries(
            geometry_of(
                core_split_polygons(to_geoarrow(polygons), NROWS, NCOLS, TRANSFORM)
            )
        )

        assert pieces.is_valid.all()
        assert pieces.area.sum() == pytest.approx(polygons.area.sum())

    def test_bounded_passed_through(self):
        outside = gpd.GeoSeries([LineString([(5.0, 0.5), (16.0, 1.5)])])

        bounded = geometry_of(
            core_split_linestrings(to_geoarrow(outside), NROWS, NCOLS, TRANSFORM, True)
        )
        unbounded = geometry_of(
            core_split_linestrings(to_geoarrow(outside), NROWS, NCOLS, TRANSFORM, False)
        )

        assert len(bounded) == 1
        assert len(unbounded) > 1


class TestErrors:
    def test_rejects_wrong_geometry_type(self, polygons, linestrings):
        with pytest.raises(ValueError, match="geoarrow.polygon"):
            batches_of(
                core_split_linestrings(to_geoarrow(polygons), NROWS, NCOLS, TRANSFORM)
            )
        with pytest.raises(ValueError, match="geoarrow.linestring"):
            batches_of(
                core_split_polygons(to_geoarrow(linestrings), NROWS, NCOLS, TRANSFORM)
            )

    def test_rejects_multi_geometries(self):
        """Multi-part geometries must be merged or exploded first - the
        error should say so rather than complain about Arrow types"""
        multi = gpd.GeoSeries.from_wkt(
            ["MULTILINESTRING ((0.5 0.5, 1.5 1.5), (2.5 2.5, 3.5 3.5))"]
        )
        with pytest.raises(ValueError, match="explode"):
            core_split_linestrings(to_geoarrow(multi), NROWS, NCOLS, TRANSFORM)

    def test_rejects_a_batch_that_does_not_match_the_schema(self, linestrings):
        """A stream states its type up front and is taken at its word, so the
        batches that follow are checked against it rather than read blindly"""
        array = pa.array(linestrings.to_arrow(geometry_encoding="geoarrow"))
        schema = pa.schema(
            [
                pa.field(
                    "geometry",
                    array.type,
                    nullable=False,
                    metadata={b"ARROW:extension:name": b"geoarrow.linestring"},
                )
            ]
        )
        mismatched = pa.RecordBatch.from_arrays(
            [pa.array([[1, 2]], type=pa.list_(pa.int32()))], names=["geometry"]
        )
        reader = pa.RecordBatchReader.from_batches(schema, iter([mismatched]))

        with pytest.raises(ValueError, match="Could not read a batch"):
            batches_of(core_split_linestrings(reader, NROWS, NCOLS, TRANSFORM))

    def test_rejects_wkb_encoding(self, linestrings):
        arrow = linestrings.to_arrow(geometry_encoding="WKB")
        with pytest.raises(ValueError, match="geoarrow.wkb"):
            core_split_linestrings(arrow, NROWS, NCOLS, TRANSFORM)

    def test_rejects_null_geometries(self, linestrings):
        with_null = gpd.GeoSeries([linestrings[0], None])
        with pytest.raises(ValueError, match="null"):
            batches_of(
                core_split_linestrings(to_geoarrow(with_null), NROWS, NCOLS, TRANSFORM)
            )

    def test_rejects_non_arrow_source(self, linestrings):
        with pytest.raises(TypeError, match="Arrow"):
            core_split_linestrings(linestrings.to_numpy(), NROWS, NCOLS, TRANSFORM)

    def test_rejects_stream_without_a_geometry_column(self):
        table = pa.table({"a": [1, 2], "b": [3, 4]})
        with pytest.raises(ValueError, match="GeoArrow"):
            core_split_linestrings(table, NROWS, NCOLS, TRANSFORM)
