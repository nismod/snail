"""Tests for GeoArrow data exchange with the C++ extension (issue #14)

Geometries cross into the extension as a stream of GeoArrow batches and the
split pieces come back as a stream of record batches. These tests cover that
interface: which sources can be split, how the stream behaves, and how the
pieces come back. That the splits themselves are right is covered in
test_intersection.py.
"""

import json

import geopandas as gpd
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from numpy.testing import assert_array_equal
from shapely.geometry import LineString, Point, Polygon
from snail.core.intersections import split_geometries as core_split_geometries
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

    def test_wkb_agrees_with_geoarrow(self, linestrings, polygons):
        """WKB holds each geometry as a serialised blob rather than in Arrow
        buffers, so it is decoded rather than read in place - and must give
        exactly the pieces the geoarrow-encoded column gives"""
        for geometries, split in (
            (linestrings, core_split_linestrings),
            (polygons, core_split_polygons),
        ):
            wkb = geometries.to_arrow(geometry_encoding="WKB")
            field = pa.Field._import_from_c_capsule(wkb.__arrow_c_array__()[0])
            assert field.metadata[b"ARROW:extension:name"] == b"geoarrow.wkb"

            actual = geometry_of(split(wkb, NROWS, NCOLS, TRANSFORM))
            expected = geometry_of(
                split(to_geoarrow(geometries), NROWS, NCOLS, TRANSFORM)
            )

            assert_same_pieces(actual, expected)

    def test_geoparquet_written_with_default_encoding(self, tmp_path, many_linestrings):
        """GeoDataFrame.to_parquet writes geoarrow.wkb unless told otherwise,
        so this is what a GeoParquet file looks like by default"""
        frame = gpd.GeoDataFrame(
            {"a": range(len(many_linestrings))}, geometry=many_linestrings
        )
        path = tmp_path / "default.parquet"
        frame.to_parquet(path)
        parquet = pq.ParquetFile(path)
        assert (
            parquet.schema_arrow.field("geometry").metadata[b"ARROW:extension:name"]
            == b"geoarrow.wkb"
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

    def test_three_dimensional_wkb(self, many_linestrings, polygons):
        """A z ordinate widens each coordinate in the blob too; splitting is
        planar, so the pieces must be unchanged"""
        for geometries, split in (
            (many_linestrings, core_split_linestrings),
            (polygons, core_split_polygons),
        ):
            actual = geometry_of(
                split(
                    with_z(geometries).to_arrow(geometry_encoding="WKB"),
                    NROWS,
                    NCOLS,
                    TRANSFORM,
                )
            )
            expected = geometry_of(
                split(to_geoarrow(geometries), NROWS, NCOLS, TRANSFORM)
            )

            assert_same_pieces(actual, expected)

    def test_rejects_large_list_offsets(self, many_linestrings, polygons):
        """Arrow lists may carry 64-bit offsets ("+L") instead of 32-bit. The
        GeoArrow spec asks a reader to accept them, but geoarrow-c - which
        reads the coordinates here - does not, and working around that cost
        more code than it saved. geopandas never emits one; an array built
        with pyarrow can be one, and casting it is a single call."""
        cases = (
            (many_linestrings, large_list_linestring_type(), core_split_linestrings),
            (polygons, large_list_polygon_type(), core_split_polygons),
        )
        for geometries, large_type, split in cases:
            arrow = pa.array(as_nested_lists(geometries), type=large_type)
            assert pa.types.is_large_list(arrow.type)

            with pytest.raises(ValueError, match=r"cast the geometry column"):
                batches_of(split(arrow, NROWS, NCOLS, TRANSFORM))

    def test_large_list_offsets_split_once_cast(self, many_linestrings):
        """...and casting really is all it takes"""
        arrow = pa.array(
            as_nested_lists(many_linestrings), type=large_list_linestring_type()
        )
        cast = arrow.cast(pa.list_(arrow.type.field(0)))

        actual = geometry_of(core_split_linestrings(cast, NROWS, NCOLS, TRANSFORM))
        expected = geometry_of(
            core_split_linestrings(
                to_geoarrow(many_linestrings), NROWS, NCOLS, TRANSFORM
            )
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


class TestMixed:
    """split_geometries takes each geometry on its own terms, and gives the
    pieces back as WKB - the one encoding an Arrow stream can carry every
    type in"""

    def test_result_is_wkb(self, linestrings):
        stream = core_split_geometries(
            linestrings.to_arrow(geometry_encoding="WKB"), NROWS, NCOLS, TRANSFORM
        )
        assert stream.geometry_type == "geoarrow.wkb"

        reader = pa.RecordBatchReader._import_from_c_capsule(
            stream.__arrow_c_stream__()
        )
        schema = reader.schema
        assert schema.names == ["geometry", "parent"]
        assert schema.field("geometry").type == pa.binary()
        assert (
            schema.field("geometry").metadata[b"ARROW:extension:name"]
            == b"geoarrow.wkb"
        )
        assert schema.field("parent").type == pa.int64()
        assert not schema.field("geometry").nullable
        assert not schema.field("parent").nullable
        list(reader)

    def test_agrees_with_the_typed_splits(self, linestrings, polygons):
        """A column that does hold one type must split to the same pieces
        either way - the typed path is an optimisation, not a variant"""
        for geometries, typed in (
            (linestrings, core_split_linestrings),
            (polygons, core_split_polygons),
        ):
            for source in (
                geometries.to_arrow(geometry_encoding="WKB"),
                to_geoarrow(geometries),
            ):
                actual = geometry_of(
                    core_split_geometries(source, NROWS, NCOLS, TRANSFORM)
                )
                expected = geometry_of(
                    typed(to_geoarrow(geometries), NROWS, NCOLS, TRANSFORM)
                )
                assert_same_pieces(actual, expected)

    def test_splits_a_column_of_several_types(self):
        """The case geopandas cannot even write as geoarrow-encoded:
        ValueError('Geometry type combination is not supported')"""
        mixed = gpd.GeoSeries(
            [
                Point(2.5, 2.5),
                LineString([(0.5, 0.5), (3.5, 0.5)]),
                Polygon([(0.5, 0.5), (2.5, 0.5), (2.5, 2.5), (0.5, 2.5)]),
            ]
        )
        with pytest.raises(ValueError, match="not supported"):
            mixed.to_arrow(geometry_encoding="geoarrow")

        pieces, parents = read_split_stream(
            core_split_geometries(
                mixed.to_arrow(geometry_encoding="WKB"), NROWS, NCOLS, TRANSFORM
            )
        )

        by_parent = {}
        for piece, parent in zip(pieces, parents):
            by_parent.setdefault(int(parent), []).append(piece)
        # a point has nothing to split, so it comes back as itself
        assert [p.wkt for p in by_parent[0]] == ["POINT (2.5 2.5)"]
        assert {p.geom_type for p in by_parent[1]} == {"LineString"}
        assert {p.geom_type for p in by_parent[2]} == {"Polygon"}
        # every piece is attributed to the row it came from
        assert sorted(by_parent) == [0, 1, 2]

    def test_multi_part_geometries_split_part_by_part(self):
        multi = gpd.GeoSeries.from_wkt(
            ["MULTILINESTRING ((0.5 0.5, 3.5 0.5), (0.5 2.5, 3.5 2.5))"]
        )
        parts = gpd.GeoSeries.from_wkt(
            ["LINESTRING (0.5 0.5, 3.5 0.5)", "LINESTRING (0.5 2.5, 3.5 2.5)"]
        )

        actual, parents = read_split_stream(
            core_split_geometries(
                multi.to_arrow(geometry_encoding="WKB"), NROWS, NCOLS, TRANSFORM
            )
        )
        expected = geometry_of(
            core_split_linestrings(to_geoarrow(parts), NROWS, NCOLS, TRANSFORM)
        )

        assert_same_pieces(actual, expected)
        # both parts came out of the one row, so both are attributed to it
        assert list(parents) == [0] * len(actual)

    def test_geometry_collections_recurse(self):
        """A collection's members are split on their own terms, however deep
        they are nested"""
        collections = gpd.GeoSeries.from_wkt(
            [
                "GEOMETRYCOLLECTION (POINT (1.5 1.5), LINESTRING (0.5 3.5, 3.5 3.5))",
                (
                    "GEOMETRYCOLLECTION (GEOMETRYCOLLECTION "
                    "(LINESTRING (0.5 0.5, 2.5 0.5)))"
                ),
            ]
        )

        pieces, parents = read_split_stream(
            core_split_geometries(
                collections.to_arrow(geometry_encoding="WKB"),
                NROWS,
                NCOLS,
                TRANSFORM,
            )
        )

        # the point is not split; the line crosses x = 1, 2 and 3
        first = [p for p, r in zip(pieces, parents) if r == 0]
        assert [p.geom_type for p in first] == ["Point"] + ["LineString"] * 4
        # nested two deep, and still split as an ordinary line: x = 1 and 2
        nested = [p for p, r in zip(pieces, parents) if r == 1]
        assert [p.geom_type for p in nested] == ["LineString"] * 3

    @pytest.mark.parametrize(
        "wkt",
        [
            "GEOMETRYCOLLECTION EMPTY",
            "LINESTRING EMPTY",
            "POLYGON EMPTY",
            "MULTILINESTRING EMPTY",
        ],
    )
    def test_empty_geometries_come_back_as_themselves(self, wkt):
        """An empty geometry has nothing to split, and dropping it would lose
        the row - so it goes back unchanged"""
        empty = gpd.GeoSeries.from_wkt([wkt])

        pieces, parents = read_split_stream(
            core_split_geometries(
                empty.to_arrow(geometry_encoding="WKB"), NROWS, NCOLS, TRANSFORM
            )
        )

        assert len(pieces) == 1
        assert pieces[0].is_empty
        assert pieces[0].wkt == wkt
        assert list(parents) == [0]

    def test_rejects_null_geometries(self, linestrings):
        with_null = gpd.GeoSeries([linestrings[0], None])
        with pytest.raises(ValueError, match="null"):
            batches_of(
                core_split_geometries(
                    with_null.to_arrow(geometry_encoding="WKB"),
                    NROWS,
                    NCOLS,
                    TRANSFORM,
                )
            )


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
            metadata = schema.field("geometry").metadata
            assert metadata[b"ARROW:extension:name"] == extension
            # GeoArrow carries a geometry's CRS and edge type in a JSON object
            # beside its name. A split says nothing about either - the pieces
            # are in whatever the source's CRS was - so the object is empty.
            assert json.loads(metadata[b"ARROW:extension:metadata"]) == {}
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

    def test_wkb_names_the_row_and_type_it_could_not_split(self, linestrings):
        """A WKB column says nothing about its geometry types until they are
        decoded, so a feature of the wrong type is only found part-way
        through: the error has to say which row, since the caller cannot see
        it from the schema"""
        mixed = gpd.GeoSeries([linestrings[0], Point(2.5, 2.5)])
        with pytest.raises(ValueError, match=r"Cannot split a Point \(row 1\)"):
            batches_of(
                core_split_linestrings(
                    mixed.to_arrow(geometry_encoding="WKB"), NROWS, NCOLS, TRANSFORM
                )
            )

    def test_wkb_multi_part_geometries_say_what_to_do(self, linestrings):
        multi = gpd.GeoSeries.from_wkt(
            ["MULTILINESTRING ((0.5 0.5, 1.5 1.5), (2.5 2.5, 3.5 3.5))"]
        )
        with pytest.raises(ValueError, match="explode"):
            batches_of(
                core_split_linestrings(
                    multi.to_arrow(geometry_encoding="WKB"), NROWS, NCOLS, TRANSFORM
                )
            )

    def test_rejects_null_geometries(self, linestrings):
        """Both encodings refuse a null rather than dropping it: the same
        data read two ways should not behave differently"""
        with_null = gpd.GeoSeries([linestrings[0], None])
        for source in (
            to_geoarrow(with_null),
            with_null.to_arrow(geometry_encoding="WKB"),
        ):
            with pytest.raises(ValueError, match="null"):
                batches_of(core_split_linestrings(source, NROWS, NCOLS, TRANSFORM))

    def test_rejects_non_arrow_source(self, linestrings):
        with pytest.raises(TypeError, match="Arrow"):
            core_split_linestrings(linestrings.to_numpy(), NROWS, NCOLS, TRANSFORM)

    def test_rejects_stream_without_a_geometry_column(self):
        table = pa.table({"a": [1, 2], "b": [3, 4]})
        with pytest.raises(ValueError, match="GeoArrow"):
            core_split_linestrings(table, NROWS, NCOLS, TRANSFORM)
