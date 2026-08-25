from pathlib import Path

import geopandas as gpd
import pandas as pd
import pytest
from numpy.testing import assert_array_equal
from shapely import from_wkt
from shapely.geometry import LineString, Point, Polygon

from snail.cli import _default_output_path, _expand_layers, snail
from snail.overlay import _geom_kinds


def test_split_multiband_attributes_column_per_band(
    tmp_path, two_band_raster, lines_over_raster
):
    features_path = tmp_path / "lines.geojson"
    lines_over_raster.to_file(features_path)
    output_path = tmp_path / "splits.gpkg"

    snail(
        [
            "split",
            "--features",
            str(features_path),
            "--raster",
            str(two_band_raster),
            "--attribute",
            "--band",
            "1",
            "2",
            "--output",
            str(output_path),
        ]
    )

    splits = gpd.read_file(output_path)
    assert_array_equal(splits["two_band_band_1"].values, [0.0, 1.0, 22.0])
    assert_array_equal(splits["two_band_band_2"].values, [100.0, 101.0, 122.0])


def test_split_parquet_output(tmp_path, two_band_raster, lines_over_raster):
    features_path = tmp_path / "lines.geojson"
    lines_over_raster.to_file(features_path)
    output_path = tmp_path / "splits.parquet"

    snail(
        [
            "split",
            "--features",
            str(features_path),
            "--raster",
            str(two_band_raster),
            "--attribute",
            "--output",
            str(output_path),
        ]
    )

    splits = gpd.read_parquet(output_path)
    assert len(splits) == 3
    assert "two_band_band_1" in splits.columns


def test_split_all_layers(tmp_path, two_band_raster, lines_over_raster):
    features_path = tmp_path / "features.gpkg"
    lines_over_raster.to_file(features_path, layer="first")
    lines_over_raster.to_file(features_path, layer="second")
    output_path = tmp_path / "splits.gpkg"

    snail(
        [
            "split",
            "--features",
            str(features_path),
            "--all-layers",
            "--raster",
            str(two_band_raster),
            "--attribute",
            "--band",
            "1",
            "--output",
            str(output_path),
        ]
    )

    for layer in ("first", "second"):
        splits = gpd.read_file(output_path, layer=layer)
        assert_array_equal(splits["two_band_band_1"].values, [0.0, 1.0, 22.0])


def test_process_default_output_and_wildcard_layers(
    tmp_path, two_band_raster, lines_over_raster
):
    features_path = tmp_path / "features.gpkg"
    lines_over_raster.to_file(features_path, layer="first")
    lines_over_raster.to_file(features_path, layer="second")

    features_csv = tmp_path / "features.csv"
    pd.DataFrame({"path": [str(features_path)], "layer": ["*"]}).to_csv(
        features_csv, index=False
    )
    rasters_csv = tmp_path / "rasters.csv"
    pd.DataFrame(
        {"path": [str(two_band_raster)], "key": ["hazard"], "bands": ["1,2"]}
    ).to_csv(rasters_csv, index=False)

    snail(["process", "-fs", str(features_csv), "-rs", str(rasters_csv)])

    # one output per layer, named "{features_stem}_{layer}__{rasters_stem}.parquet"
    for layer in ("first", "second"):
        output_path = tmp_path / f"features_{layer}__rasters.parquet"
        assert output_path.exists()
        splits = gpd.read_parquet(output_path)
        assert_array_equal(splits["hazard_band_1"].values, [0.0, 1.0, 22.0])
        assert_array_equal(splits["hazard_band_2"].values, [100.0, 101.0, 122.0])


def test_expand_layers_with_explicit_output(tmp_path, lines_over_raster):
    features_path = tmp_path / "features.gpkg"
    lines_over_raster.to_file(features_path, layer="first")
    lines_over_raster.to_file(features_path, layer="second")

    vector_layers = pd.DataFrame(
        {
            "path": [str(features_path)],
            "layer": ["*"],
            "output_path": [str(tmp_path / "out.parquet")],
        }
    )
    expanded = _expand_layers(vector_layers)
    assert expanded.layer.tolist() == ["first", "second"]
    # explicit output paths gain a layer name per row
    assert expanded.output_path.tolist() == [
        str(tmp_path / "out_first.parquet"),
        str(tmp_path / "out_second.parquet"),
    ]


def test_default_output_path():
    assert _default_output_path("dir/net.gpkg", "roads", "hazards") == str(
        Path("dir/net_roads__hazards.parquet")
    )
    assert _default_output_path("dir/net.gpkg", None, "hazards") == str(
        Path("dir/net__hazards.parquet")
    )
    assert _default_output_path("dir/net.gpkg", float("nan"), "hazards") == str(
        Path("dir/net__hazards.parquet")
    )


def test_cli_without_command_prints_help(capsys):
    snail([])

    captured = capsys.readouterr()
    assert "usage: snail" in captured.out


class TestGeomKinds:
    """Which split a layer gets is decided by the kinds of geometry in it,
    not by whichever one row 0 happens to hold"""

    def test_multi_part_counts_as_its_single_part_kind(self):
        """prepare_linestrings turns the one into the other, so a layer of
        both is still one kind of thing to split"""
        both = gpd.GeoSeries.from_wkt(
            [
                "LINESTRING (0.5 0.5, 1.5 1.5)",
                "MULTILINESTRING ((2.5 2.5, 3.5 3.5))",
            ]
        )
        assert _geom_kinds(gpd.GeoDataFrame(geometry=both)) == {"LineString"}

    def test_different_kinds_are_counted_separately(self):
        mixed = gpd.GeoSeries([Point(0.5, 0.5), LineString([(1.5, 1.5), (2.5, 2.5)])])
        assert _geom_kinds(gpd.GeoDataFrame(geometry=mixed)) == {
            "Point",
            "LineString",
        }

    def test_nulls_are_ignored(self):
        with_null = gpd.GeoSeries([LineString([(0.5, 0.5), (1.5, 1.5)]), None])
        assert _geom_kinds(gpd.GeoDataFrame(geometry=with_null)) == {"LineString"}


@pytest.fixture
def mixed_features(tmp_path):
    """A layer whose first row says nothing about the rest of it"""
    features = gpd.GeoDataFrame(
        {"name": ["a", "b", "c"]},
        geometry=[
            Point(2.5, 2.5),
            LineString([(0.5, 0.5), (3.5, 0.5)]),
            Polygon([(0.5, 0.5), (2.5, 0.5), (2.5, 2.5), (0.5, 2.5)]),
        ],
        crs="EPSG:4326",
    )
    path = tmp_path / "mixed.gpkg"
    features.to_file(path)
    return path


def run_split(features_path, output_path):
    snail(
        [
            "split",
            "--features",
            str(features_path),
            "--transform",
            "1",
            "0",
            "0",
            "0",
            "1",
            "0",
            "--width",
            "4",
            "--height",
            "4",
            "--output",
            str(output_path),
        ]
    )


def test_split_of_a_mixed_layer_keeps_every_feature(mixed_features, tmp_path):
    """Row 0 is a Point, so before the mixed path this layer was processed as
    points throughout: the line and the polygon came back unsplit"""
    output = tmp_path / "split.gpkg"
    run_split(mixed_features, output)

    splits = gpd.read_file(output)
    by_name = splits.groupby("name").size()
    assert set(by_name.index) == {"a", "b", "c"}
    # the point is not split, the line crosses x = 1, 2 and 3, and the
    # polygon covers nine whole cells
    assert by_name["a"] == 1
    assert by_name["b"] == 4
    assert by_name["c"] == 9
    assert set(splits[splits.name == "c"].geometry.geom_type) == {"Polygon"}


def test_split_of_a_single_kind_layer_is_unchanged(tmp_path):
    """A layer of one kind keeps its typed split, multi-parts included"""
    features = gpd.GeoDataFrame(
        {"name": ["a", "b"]},
        geometry=[
            LineString([(0.5, 0.5), (3.5, 0.5)]),
            from_wkt("MULTILINESTRING ((0.5 2.5, 3.5 2.5))"),
        ],
        crs="EPSG:4326",
    )
    path = tmp_path / "lines.gpkg"
    features.to_file(path)
    output = tmp_path / "split.gpkg"
    run_split(path, output)

    splits = gpd.read_file(output)
    assert set(splits.geometry.geom_type) == {"LineString"}
    assert len(splits) == 8
