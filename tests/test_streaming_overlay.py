import geopandas as gpd
import numpy as np
import pandas as pd
import pyarrow as pa
import rasterio
from geopandas.testing import assert_geodataframe_equal
from shapely.geometry import Point

from snail.intersection import read_raster_values_for_splits
from snail.overlay import (
    iter_overlay_raster_batches,
    iter_split_features_batches,
    overlay_raster,
)


def _reader(frame, batches):
    table = pa.table(frame.to_arrow(geometry_encoding="WKB"))
    return pa.RecordBatchReader.from_batches(table.schema, batches(table))


def test_streaming_overlay_is_lazy_and_preserves_source_batches(
    two_band_raster, lines_over_raster
):
    pulled = []

    def batches(table):
        for position in range(len(table)):
            pulled.append(position)
            yield table.slice(position, 1).to_batches()[0]

    result = iter_overlay_raster_batches(
        _reader(lines_over_raster, batches),
        two_band_raster,
        bands=[1],
        split_batch_size=1,
    )
    assert pulled == []

    output = iter(result)
    first = next(output)
    assert pulled == [0]
    assert first.num_rows == 2  # one feature is never divided between batches
    second = next(output)
    assert pulled == [0, 1]
    assert second.num_rows == 1


def test_streaming_and_materialized_overlay_are_equivalent(
    two_band_raster, lines_over_raster
):
    streamed = gpd.GeoDataFrame.from_arrow(
        iter_overlay_raster_batches(
            pa.table(lines_over_raster.to_arrow(geometry_encoding="WKB")),
            two_band_raster,
            split_batch_size=1,
        ).read_all()
    )
    materialized = overlay_raster(lines_over_raster, two_band_raster)

    assert_geodataframe_equal(streamed, materialized)


def test_streaming_split_reader_has_stable_empty_schema(lines_over_raster):
    table = pa.table(lines_over_raster.iloc[:0].to_arrow(geometry_encoding="WKB"))
    grid = rasterio.transform.Affine(1, 0, 0, 0, -1, 4)
    from snail.intersection import GridDefinition

    reader = iter_split_features_batches(
        table,
        GridDefinition("EPSG:4326", 4, 4, tuple(grid)[:6]),
    )

    assert reader.schema.names[-3:] == ["split", "index_i", "index_j"]
    assert reader.read_all().num_rows == 0


def test_streaming_split_drops_null_geometries():
    frame = gpd.GeoDataFrame(
        {"name": ["missing", "point"], "geometry": [None, Point(0.5, 0.5)]},
        crs="EPSG:4326",
    )
    from snail.intersection import GridDefinition

    result = iter_split_features_batches(
        pa.table(frame.to_arrow(geometry_encoding="WKB")),
        GridDefinition("EPSG:4326", 2, 2, (1, 0, 0, 0, 1, 0)),
    ).read_all()

    assert result.column("name").to_pylist() == ["point"]


def test_reader_groups_mixed_dtype_vrt_bands(tmp_path):
    for name, dtype in (("byte.tif", "uint8"), ("float.tif", "float32")):
        with rasterio.open(
            tmp_path / name,
            "w",
            driver="GTiff",
            width=2,
            height=2,
            count=1,
            dtype=dtype,
            crs="EPSG:4326",
            transform=rasterio.transform.from_origin(0, 2, 1, 1),
        ) as dataset:
            dataset.write(np.ones((2, 2), dtype=dtype), 1)

    vrt = tmp_path / "mixed.vrt"
    vrt.write_text(
        """<VRTDataset rasterXSize="2" rasterYSize="2">
  <SRS>EPSG:4326</SRS><GeoTransform>0,1,0,2,0,-1</GeoTransform>
  <VRTRasterBand dataType="Byte" band="1"><SimpleSource>
    <SourceFilename relativeToVRT="1">byte.tif</SourceFilename><SourceBand>1</SourceBand>
    <SrcRect xOff="0" yOff="0" xSize="2" ySize="2"/><DstRect xOff="0" yOff="0" xSize="2" ySize="2"/>
  </SimpleSource></VRTRasterBand>
  <VRTRasterBand dataType="Float32" band="2"><SimpleSource>
    <SourceFilename relativeToVRT="1">float.tif</SourceFilename><SourceBand>1</SourceBand>
    <SrcRect xOff="0" yOff="0" xSize="2" ySize="2"/><DstRect xOff="0" yOff="0" xSize="2" ySize="2"/>
  </SimpleSource></VRTRasterBand>
</VRTDataset>"""
    )
    splits = pd.DataFrame({"index_i": [0, 1], "index_j": [0, 1]})

    values = read_raster_values_for_splits(splits, vrt, [1, 2])

    assert values[1].dtype == np.dtype("uint8")
    assert values[2].dtype == np.dtype("float32")
    assert values[1].tolist() == [1, 1]
    assert values[2].tolist() == [1.0, 1.0]
