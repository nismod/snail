"""Tests for GeoArrow data exchange with the C++ extension (issue #14)

Geometry columns cross into the extension as GeoArrow arrays and the split
pieces come back the same way. These tests cover that interface itself: the
equivalence of the splits with the single-geometry functions is covered in
test_intersection.py.
"""

import geopandas as gpd
import pyarrow as pa
import pytest
from numpy.testing import assert_array_equal
from shapely.geometry import LineString, Polygon
from snail.core.intersections import split_linestring as core_split_linestring
from snail.core.intersections import split_linestrings as core_split_linestrings
from snail.core.intersections import split_polygon as core_split_polygon
from snail.core.intersections import split_polygons as core_split_polygons

from snail.intersection import GridDefinition, from_geoarrow, to_geoarrow

NROWS = NCOLS = 4
TRANSFORM = (1, 0, 0, 0, 1, 0)


def linestring_type():
    """list<vertices: fixed_size_list<xy: double>[2]>"""
    return pa.list_(pa.field("vertices", vertices_type(), nullable=False))


def polygon_type():
    """list<rings: list<vertices: fixed_size_list<xy: double>[2]>>"""
    return pa.list_(
        pa.field(
            "rings",
            pa.list_(pa.field("vertices", vertices_type(), nullable=False)),
            nullable=False,
        )
    )


def vertices_type():
    return pa.list_(pa.field("xy", pa.float64(), nullable=False), 2)


def field_of(array):
    """The Arrow field an array exports, carrying its GeoArrow metadata"""
    schema_capsule, _ = array.__arrow_c_array__()
    return pa.Field._import_from_c_capsule(schema_capsule)


@pytest.fixture
def grid():
    return GridDefinition(crs=None, width=NCOLS, height=NROWS, transform=TRANSFORM)


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


class TestSplitLineStrings:
    def test_matches_single_geometry_split(self, linestrings):
        """Splitting a whole GeoArrow array gives what splitting each
        linestring in turn gives"""
        pieces, parent = core_split_linestrings(
            to_geoarrow(linestrings), NROWS, NCOLS, TRANSFORM
        )
        actual = gpd.GeoSeries.from_arrow(pieces)

        assert_array_equal(parent, [0, 0, 0, 1, 1, 1])
        expected = []
        for geometry in linestrings:
            expected.extend(core_split_linestring(geometry, NROWS, NCOLS, TRANSFORM))
        assert len(actual) == len(expected)
        for actual_piece, expected_piece in zip(actual, expected):
            assert actual_piece.equals_exact(expected_piece, 1e-12)

    def test_returns_geoarrow_linestrings(self, linestrings):
        """The pieces come back as a geoarrow.linestring array, so any
        Arrow-aware reader can interpret them"""
        pieces, _ = core_split_linestrings(
            to_geoarrow(linestrings), NROWS, NCOLS, TRANSFORM
        )
        field = field_of(pieces)

        assert pieces.geometry_type == "geoarrow.linestring"
        assert field.metadata[b"ARROW:extension:name"] == b"geoarrow.linestring"
        assert field.type == linestring_type()

    def test_accepts_plain_arrow_array(self):
        """An array of the right shape is read even without GeoArrow
        extension metadata, as produced by pyarrow directly"""
        arrow = pa.array([[[0.5, 0.5], [1.5, 1.5]]], type=linestring_type())
        pieces, parent = core_split_linestrings(arrow, NROWS, NCOLS, TRANSFORM)

        assert len(pieces) == 2
        assert_array_equal(parent, [0, 0])

    def test_reads_sliced_array(self, linestrings):
        """Slices carry a non-zero Arrow offset, which must be honoured"""
        arrow = pa.array(to_geoarrow(linestrings))
        pieces, parent = core_split_linestrings(
            arrow.slice(1, 1), NROWS, NCOLS, TRANSFORM
        )
        actual = gpd.GeoSeries.from_arrow(pieces)
        expected = core_split_linestring(linestrings[1], NROWS, NCOLS, TRANSFORM)

        assert_array_equal(parent, [0, 0, 0])
        for actual_piece, expected_piece in zip(actual, expected):
            assert actual_piece.equals_exact(expected_piece, 1e-12)

    def test_empty_array(self):
        pieces, parent = core_split_linestrings(
            pa.array([], type=linestring_type()), NROWS, NCOLS, TRANSFORM
        )
        assert len(pieces) == 0
        assert len(parent) == 0
        assert len(from_geoarrow(pieces)) == 0

    def test_bounded_passed_through(self):
        """bounded=True leaves the parts outside the grid unsplit"""
        outside = gpd.GeoSeries([LineString([(5.0, 0.5), (16.0, 1.5)])])

        bounded, _ = core_split_linestrings(
            to_geoarrow(outside), NROWS, NCOLS, TRANSFORM, True
        )
        unbounded, _ = core_split_linestrings(
            to_geoarrow(outside), NROWS, NCOLS, TRANSFORM, False
        )
        assert len(bounded) == 1
        assert len(unbounded) > 1


class TestSplitPolygons:
    def test_matches_single_geometry_split(self, polygons):
        pieces, parent = core_split_polygons(
            to_geoarrow(polygons), NROWS, NCOLS, TRANSFORM
        )
        actual = gpd.GeoSeries.from_arrow(pieces)

        expected = []
        for geometry in polygons:
            expected.extend(core_split_polygon(geometry, NROWS, NCOLS, TRANSFORM))
        assert len(actual) == len(expected)
        assert sorted(round(piece.area, 9) for piece in actual) == sorted(
            round(piece.area, 9) for piece in expected
        )
        assert_array_equal(parent[:1], [0])
        assert parent.max() == 1

    def test_returns_geoarrow_polygons(self, polygons):
        pieces, _ = core_split_polygons(to_geoarrow(polygons), NROWS, NCOLS, TRANSFORM)
        field = field_of(pieces)

        assert pieces.geometry_type == "geoarrow.polygon"
        assert field.metadata[b"ARROW:extension:name"] == b"geoarrow.polygon"
        assert field.type == polygon_type()

    def test_pieces_are_valid_and_conserve_area(self, polygons):
        """Splitting cuts a polygon up without losing or duplicating area,
        holes included"""
        pieces, _ = core_split_polygons(to_geoarrow(polygons), NROWS, NCOLS, TRANSFORM)
        actual = gpd.GeoSeries.from_arrow(pieces)

        assert actual.is_valid.all()
        assert actual.area.sum() == pytest.approx(polygons.area.sum())

    def test_empty_array(self):
        pieces, parent = core_split_polygons(
            pa.array([], type=polygon_type()), NROWS, NCOLS, TRANSFORM
        )
        assert len(pieces) == 0
        assert len(parent) == 0
        assert len(from_geoarrow(pieces)) == 0


class TestArrowInterface:
    def test_export_is_repeatable(self, linestrings):
        """Exporting shares the buffers rather than handing them over, so an
        array can be read any number of times"""
        pieces, _ = core_split_linestrings(
            to_geoarrow(linestrings), NROWS, NCOLS, TRANSFORM
        )
        first = pa.array(pieces)
        second = pa.array(pieces)

        assert first.equals(second)
        assert len(first) == len(pieces)

    def test_pieces_outlive_reader(self, linestrings):
        """The exported buffers stay alive with the Arrow array, not with
        the object it came from"""
        pieces, _ = core_split_linestrings(
            to_geoarrow(linestrings), NROWS, NCOLS, TRANSFORM
        )
        arrow = pa.array(pieces)
        expected = arrow.to_pylist()
        del pieces

        assert arrow.to_pylist() == expected

    def test_rejects_wrong_geometry_type(self, polygons, linestrings):
        with pytest.raises(ValueError, match="geoarrow.polygon"):
            core_split_linestrings(to_geoarrow(polygons), NROWS, NCOLS, TRANSFORM)
        with pytest.raises(ValueError, match="geoarrow.linestring"):
            core_split_polygons(to_geoarrow(linestrings), NROWS, NCOLS, TRANSFORM)

    def test_rejects_multi_geometries(self):
        """MultiLineStrings must be merged or exploded first - the error
        should say so rather than complain about Arrow types"""
        multi = gpd.GeoSeries.from_wkt(
            ["MULTILINESTRING ((0.5 0.5, 1.5 1.5), (2.5 2.5, 3.5 3.5))"]
        )
        with pytest.raises(ValueError, match="explode"):
            core_split_linestrings(to_geoarrow(multi), NROWS, NCOLS, TRANSFORM)

    def test_rejects_wkb_encoding(self, linestrings):
        arrow = linestrings.to_arrow(geometry_encoding="WKB")
        with pytest.raises(ValueError, match="geoarrow.wkb"):
            core_split_linestrings(arrow, NROWS, NCOLS, TRANSFORM)

    def test_rejects_separated_coordinates(self, linestrings):
        arrow = linestrings.to_arrow(geometry_encoding="geoarrow", interleaved=False)
        with pytest.raises(ValueError, match="separated"):
            core_split_linestrings(arrow, NROWS, NCOLS, TRANSFORM)

    def test_rejects_null_geometries(self, linestrings):
        with_null = gpd.GeoSeries([linestrings[0], None])
        with pytest.raises(ValueError, match="null"):
            core_split_linestrings(to_geoarrow(with_null), NROWS, NCOLS, TRANSFORM)

    def test_rejects_non_arrow_input(self, linestrings):
        with pytest.raises(TypeError, match="GeoArrow"):
            core_split_linestrings(linestrings.to_numpy(), NROWS, NCOLS, TRANSFORM)
