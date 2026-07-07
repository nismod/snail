import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
import rasterio
from numpy.testing import assert_array_equal
from shapely.geometry import MultiLineString, Point

from snail.intersection import GridDefinition, split_linestrings
from snail.overlay import overlay_raster, overlay_rasters, parse_bands, split_features


class TestOverlayRaster:
    def test_single_band_from_path(self, two_band_raster, lines_over_raster):
        splits = overlay_raster(lines_over_raster, two_band_raster, bands=[1])
        # column named by raster filename stem for a single band
        assert "two_band" in splits.columns
        # first line splits into two cells, second stays in one
        assert len(splits) == 3
        assert_array_equal(splits["two_band"].values, [0.0, 1.0, 22.0])
        assert_array_equal(splits["index_i"].values, [0, 1, 2])
        assert_array_equal(splits["index_j"].values, [0, 0, 2])
        assert splits.crs == lines_over_raster.crs

    def test_all_bands_by_default(self, two_band_raster, lines_over_raster):
        splits = overlay_raster(lines_over_raster, two_band_raster)
        # multi-band rasters attribute one column per band
        assert "two_band_band_1" in splits.columns
        assert "two_band_band_2" in splits.columns
        assert_array_equal(splits["two_band_band_1"].values, [0.0, 1.0, 22.0])
        assert_array_equal(splits["two_band_band_2"].values, [100.0, 101.0, 122.0])

    def test_custom_column(self, two_band_raster, lines_over_raster):
        splits = overlay_raster(
            lines_over_raster, two_band_raster, bands=[2], column="depth"
        )
        assert_array_equal(splits["depth"].values, [100.0, 101.0, 122.0])

    def test_open_dataset(self, two_band_raster, lines_over_raster):
        with rasterio.open(two_band_raster) as dataset:
            splits = overlay_raster(lines_over_raster, dataset, bands=[1])
        assert_array_equal(splits["two_band"].values, [0.0, 1.0, 22.0])

    def test_dataarray(self, two_band_raster, lines_over_raster):
        rioxarray = pytest.importorskip("rioxarray")
        data_array = rioxarray.open_rasterio(two_band_raster)
        splits = overlay_raster(
            lines_over_raster, data_array, bands=[1], column="two_band"
        )
        assert_array_equal(splits["two_band"].values, [0.0, 1.0, 22.0])

    def test_lazy_reads_the_same_values(self, two_band_raster, lines_over_raster):
        pytest.importorskip("rioxarray")
        splits = overlay_raster(
            lines_over_raster, two_band_raster, bands=[1], lazy=True
        )
        assert_array_equal(splits["two_band"].values, [0.0, 1.0, 22.0])

    def test_reprojects_features_to_raster_crs(
        self, two_band_raster, lines_over_raster
    ):
        projected = lines_over_raster.to_crs("EPSG:3857")
        splits = overlay_raster(projected, two_band_raster, bands=[1])
        # implicitly reprojected to the raster CRS for splitting and lookup,
        # then returned in the original CRS
        assert splits.crs == projected.crs
        assert_array_equal(splits["two_band"].values, [0.0, 1.0, 22.0])

    def test_points(self, two_band_raster):
        points = gpd.GeoDataFrame(
            {"geometry": [Point(0.5, 3.5), Point(2.5, 1.5)]}, crs="EPSG:4326"
        )
        splits = overlay_raster(points, two_band_raster, bands=[1])
        assert_array_equal(splits["two_band"].values, [0.0, 22.0])

    def test_outside_raster_is_nan(self, two_band_raster):
        points = gpd.GeoDataFrame({"geometry": [Point(10.5, 10.5)]}, crs="EPSG:4326")
        splits = overlay_raster(points, two_band_raster, bands=[1])
        assert np.isnan(splits["two_band"].values[0])
        assert_array_equal(splits["index_i"].values, [-1])


class TestOverlayRasters:
    def test_list_of_paths(self, two_band_raster, lines_over_raster):
        splits = overlay_rasters(lines_over_raster, [two_band_raster])
        assert_array_equal(splits["two_band_band_1"].values, [0.0, 1.0, 22.0])
        assert_array_equal(splits["two_band_band_2"].values, [100.0, 101.0, 122.0])
        # cell indices per distinct grid
        assert_array_equal(splits["i_0"].values, [0, 1, 2])
        assert_array_equal(splits["j_0"].values, [0, 0, 2])

    def test_dataframe_with_keys_and_bands(self, two_band_raster, lines_over_raster):
        rasters = pd.DataFrame(
            {
                "path": [two_band_raster, two_band_raster],
                "key": ["flood", "heat"],
                "bands": ["1", "1,2"],
            }
        )
        splits = overlay_rasters(lines_over_raster, rasters)
        assert_array_equal(splits["flood"].values, [0.0, 1.0, 22.0])
        assert_array_equal(splits["heat_band_1"].values, [0.0, 1.0, 22.0])
        assert_array_equal(splits["heat_band_2"].values, [100.0, 101.0, 122.0])

    def test_dataframe_missing_bands_defaults_to_all(
        self, two_band_raster, lines_over_raster
    ):
        rasters = pd.DataFrame(
            {
                "path": [two_band_raster, two_band_raster],
                "key": ["flood", "heat"],
                "bands": ["1", None],
            }
        )
        splits = overlay_rasters(lines_over_raster, rasters)
        assert_array_equal(splits["flood"].values, [0.0, 1.0, 22.0])
        assert_array_equal(splits["heat_band_1"].values, [0.0, 1.0, 22.0])
        assert_array_equal(splits["heat_band_2"].values, [100.0, 101.0, 122.0])

    def test_multiple_distinct_grids(
        self, two_band_raster, half_cell_raster, lines_over_raster
    ):
        # splitting on a second, different grid re-splits the pieces from the
        # first (regression test: this failed with duplicate index labels)
        splits = overlay_rasters(
            lines_over_raster, [str(two_band_raster), str(half_cell_raster)]
        )
        assert "i_0" in splits.columns and "i_1" in splits.columns
        # line "a" (0.5,3.5)->(1.5,3.5) is split at x=1 on the coarse grid
        # and no further on the fine grid (its ends lie on fine cell edges)
        first = splits[splits["name"] == "a"]
        assert len(first) == 2
        assert_array_equal(first["two_band_band_1"].values, [0.0, 1.0])
        assert_array_equal(first["half_cell"].values, [101.0, 102.0])
        # line "b" (2.25,1.5)->(2.75,1.5) is within one coarse cell, and is
        # re-split at x=2.5 on the fine grid
        second = splits[splits["name"] == "b"]
        assert len(second) == 2
        assert_array_equal(second["two_band_band_1"].values, [22.0, 22.0])
        assert_array_equal(second["half_cell"].values, [504.0, 505.0])

    def test_requires_path_column(self, lines_over_raster):
        with pytest.raises(ValueError, match="path"):
            overlay_rasters(lines_over_raster, pd.DataFrame({"key": ["flood"]}))

    def test_lazy_reads_the_same_values(self, two_band_raster, lines_over_raster):
        pytest.importorskip("rioxarray")
        splits = overlay_rasters(lines_over_raster, [two_band_raster], lazy=True)
        assert_array_equal(splits["two_band_band_1"].values, [0.0, 1.0, 22.0])
        assert_array_equal(splits["two_band_band_2"].values, [100.0, 101.0, 122.0])


class TestSplitFeatures:
    def test_split_lines(self, lines_over_raster):
        grid = GridDefinition(
            crs="EPSG:4326", width=4, height=4, transform=(1, 0, 0, 0, -1, 4)
        )
        splits = split_features(lines_over_raster, grid)
        assert len(splits) == 3
        assert_array_equal(splits["index_i"].values, [0, 1, 2])
        assert_array_equal(splits["index_j"].values, [0, 0, 2])

    def test_empty_features(self):
        grid = GridDefinition(
            crs="EPSG:4326", width=4, height=4, transform=(1, 0, 0, 0, -1, 4)
        )
        empty = gpd.GeoDataFrame({"geometry": []}, crs="EPSG:4326")
        splits = split_features(empty, grid)
        assert len(splits) == 0
        assert "index_i" in splits.columns
        assert "index_j" in splits.columns


def test_split_linestrings_coerces_multilinestring(caplog):
    grid = GridDefinition(crs=None, width=4, height=4, transform=(1, 0, 0, 0, 1, 0))
    features = gpd.GeoDataFrame(
        {
            "name": ["multi"],
            "geometry": [
                MultiLineString(
                    [
                        [(0.5, 0.5), (1.5, 0.5)],
                        [(2.5, 2.5), (2.5, 3.5)],
                    ]
                )
            ],
        }
    )
    with caplog.at_level("WARNING"):
        splits = split_linestrings(features, grid)
    assert "MultiLineString" in caplog.text
    # each part crosses one cell boundary, so two pieces per part
    assert len(splits) == 4
    assert (splits.geometry.geom_type == "LineString").all()
    assert_array_equal(splits["name"].values, ["multi"] * 4)


def test_parse_bands():
    assert parse_bands("1,2,3") == (1, 2, 3)
    assert parse_bands("1") == (1,)
    assert parse_bands(2) == (2,)
    assert parse_bands(2.0) == (2,)
    assert parse_bands([1, 2]) == (1, 2)
    assert parse_bands((1, 2)) == (1, 2)
    assert parse_bands(None) is None
    assert parse_bands(float("nan")) is None
    assert parse_bands("") is None
