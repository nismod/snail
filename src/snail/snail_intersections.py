import geopandas as gpd

from snail.intersections import split as split_one_geom


def split(vector_data, raster_data):
    all_splits = []
    all_idx = []
    for idx, linestring in zip(vector_data.index, vector_data["geometry"]):
        splits = split_one_geom(
            linestring,
            raster_data.width,
            raster_data.height,
            list(raster_data.transform),
        )
        all_splits.extend(splits)
        all_idx.extend([idx] * len(splits))

    return gpd.GeoDataFrame({"line index": all_idx, "geometry": all_splits})
