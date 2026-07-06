"""Tests for GeoArrow-based vectorised linestring splitting (issue #14)"""

import geopandas as gpd
import pyarrow as pa
import pytest
from numpy.testing import assert_array_equal
from shapely.geometry import LineString

from snail.core.intersections import (
    split_linestring as core_split_linestring,
)
from snail.core.intersections import (
    split_linestrings as core_split_linestrings,
)
from snail.intersection import (
    GridDefinition,
    split_linestrings,
    split_linestrings_geoarrow,
)


def _interleaved_linestring_type():
    """Arrow storage type of a geoarrow.linestring array with interleaved
    coordinates: list<vertices: fixed_size_list<xy: double>[2]>"""
    return pa.list_(
        pa.field(
            "vertices",
            pa.list_(pa.field("xy", pa.float64(), nullable=False), 2),
            nullable=False,
        )
    )


@pytest.fixture
def grid():
    return GridDefinition(crs=None, width=4, height=4, transform=(1, 0, 0, 0, 1, 0))


@pytest.fixture
def linestrings():
    test_linestrings = [
        LineString([(0.5, 0.5), (0.75, 0.5), (1.5, 0.5), (1.5, 1.5)]),
        LineString([(0.5, 0.5), (0.75, 0.5), (1.5, 1.5)]),
    ]
    return gpd.GeoDataFrame({"col1": ["name1", "name2"], "geometry": test_linestrings})


class TestCoreSplitLineStrings:
    def test_round_trip_matches_split_linestring(self, grid, linestrings):
        """Splitting a whole GeoArrow array matches splitting each
        linestring in turn with the single-geometry function"""
        arrow_geometry = linestrings.geometry.to_arrow(geometry_encoding="geoarrow")
        splits, index = core_split_linestrings(
            arrow_geometry, grid.width, grid.height, grid.transform
        )
        actual = gpd.GeoSeries.from_arrow(splits)

        assert_array_equal(index, [0, 0, 0, 1, 1, 1])
        expected = []
        for geom in linestrings.geometry:
            expected.extend(
                core_split_linestring(geom, grid.width, grid.height, grid.transform)
            )
        assert len(actual) == len(expected)
        for actual_geom, expected_geom in zip(actual, expected):
            assert actual_geom.equals_exact(expected_geom, 1e-9)

    def test_returns_geoarrow_extension_type(self, grid, linestrings):
        """The returned array declares the geoarrow.linestring extension
        type, so generic Arrow consumers can interpret it"""
        splits, _ = core_split_linestrings(
            linestrings.geometry.to_arrow(geometry_encoding="geoarrow"),
            grid.width,
            grid.height,
            grid.transform,
        )
        schema_capsule, _ = splits.__arrow_c_array__()
        field = pa.Field._import_from_c_capsule(schema_capsule)
        assert field.metadata[b"ARROW:extension:name"] == b"geoarrow.linestring"
        assert field.type == _interleaved_linestring_type()

    def test_consumable_by_pyarrow(self, grid, linestrings):
        splits, index = core_split_linestrings(
            linestrings.geometry.to_arrow(geometry_encoding="geoarrow"),
            grid.width,
            grid.height,
            grid.transform,
        )
        arr = pa.array(splits)
        assert len(arr) == len(splits) == len(index) == 6
        # exporting is zero-copy and repeatable
        arr_again = pa.array(splits)
        assert arr.equals(arr_again)

    def test_sliced_input(self, grid, linestrings):
        """Sliced Arrow arrays (with a non-zero offset) are read correctly"""
        full = pa.array(linestrings.geometry.to_arrow(geometry_encoding="geoarrow"))
        splits, index = core_split_linestrings(
            full.slice(1, 1), grid.width, grid.height, grid.transform
        )
        actual = gpd.GeoSeries.from_arrow(splits)
        expected = core_split_linestring(
            linestrings.geometry[1], grid.width, grid.height, grid.transform
        )
        assert_array_equal(index, [0, 0, 0])
        for actual_geom, expected_geom in zip(actual, expected):
            assert actual_geom.equals_exact(expected_geom, 1e-9)

    def test_empty_input(self, grid):
        # geopandas cannot yet convert an empty GeoSeries to geoarrow
        # encoding, so build the empty Arrow array directly
        empty = pa.array([], type=_interleaved_linestring_type())
        splits, index = core_split_linestrings(
            empty, grid.width, grid.height, grid.transform
        )
        assert len(splits) == 0
        assert len(index) == 0
        # geopandas.GeoSeries.from_arrow fails on empty arrays (upstream
        # bug reshaping zero-length coordinates), so check via pyarrow
        assert len(pa.array(splits)) == 0

    def test_rejects_separated_coordinates(self, grid, linestrings):
        arrow_geometry = linestrings.geometry.to_arrow(
            geometry_encoding="geoarrow", interleaved=False
        )
        with pytest.raises(ValueError, match="separated"):
            core_split_linestrings(
                arrow_geometry, grid.width, grid.height, grid.transform
            )

    def test_rejects_other_geometry_types(self, grid, linestrings):
        arrow_geometry = linestrings.geometry.to_arrow(geometry_encoding="WKB")
        with pytest.raises(ValueError, match="geoarrow.wkb"):
            core_split_linestrings(
                arrow_geometry, grid.width, grid.height, grid.transform
            )

    def test_rejects_null_geometries(self, grid, linestrings):
        with_null = gpd.GeoSeries([linestrings.geometry[0], None])
        arrow_geometry = with_null.to_arrow(geometry_encoding="geoarrow")
        with pytest.raises(ValueError, match="null"):
            core_split_linestrings(
                arrow_geometry, grid.width, grid.height, grid.transform
            )


class TestSplitLineStringsGeoArrow:
    def test_matches_split_linestrings(self, grid, linestrings):
        """The GeoArrow-based split gives the same features as the
        row-by-row split_linestrings"""
        actual = split_linestrings_geoarrow(linestrings, grid)
        expected = split_linestrings(linestrings, grid)

        assert len(actual) == len(expected)
        assert (
            actual.geometry.reset_index(drop=True)
            .geom_equals_exact(
                expected.geometry.reset_index(drop=True),
                tolerance=1e-9,
                align=False,
            )
            .values.all()
        )
        assert_array_equal(actual["col1"].values, expected["col1"].values)
        assert_array_equal(actual["split"].values, expected["split"].values)
