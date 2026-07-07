from pathlib import Path

import geopandas as gpd
import pandas as pd
from numpy.testing import assert_array_equal

from snail.cli import _default_output_path, _expand_layers, snail


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
        assert_array_equal(splits["two_band"].values, [0.0, 1.0, 22.0])


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
