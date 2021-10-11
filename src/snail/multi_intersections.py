import logging
import os

import geopandas as gpd
from shapely.geometry import LineString, Polygon
from shapely.ops import polygonize

# optional progress bars
if "SNAIL_PROGRESS" in os.environ and os.environ["SNAIL_PROGRESS"] in (
    "1",
    "TRUE",
):
    try:
        from tqdm import tqdm
    except ImportError:
        from snail.tqdm_standin import tqdm_standin as tqdm
else:
    from snail.tqdm_standin import tqdm_standin as tqdm

from snail.core.intersections import split_linestring
from snail.core.intersections import split_polygon
from snail.core.intersections import get_cell_indices


def split_linestrings(vector_data, raster_data):
    all_splits = []
    all_idx = []
    logging.info("Split linestrings")
    for edge in tqdm(vector_data.itertuples(), total=len(vector_data)):
        if type(edge.geometry) != LineString:
            msg = f"Incorrect geometry type {type(edge.geometry)}, expected LineString."
            raise ValueError(msg)
        split_geoms = split_linestring(
            edge.geometry,
            raster_data.width,
            raster_data.height,
            list(raster_data.transform),
        )
        all_splits.extend(split_geoms)
        all_idx.extend([edge.Index] * len(split_geoms))

    return gpd.GeoDataFrame({"line index": all_idx, "geometry": all_splits})


def split_polygons(vector_data, raster_data):
    all_splits = []
    all_idx = []
    logging.info("Split polygons")
    for edge in tqdm(vector_data.itertuples(), total=len(vector_data)):
        if type(edge.geometry) != Polygon:
            msg = f"Incorrect geometry type {type(edge.geometry)}, expected Polygon."
            raise ValueError(msg)
        splits = split_polygon(
            edge.geometry,
            raster_data.width,
            raster_data.height,
            list(raster_data.transform),
        )
        split_geoms = list(polygonize(splits))
        all_splits.extend(split_geoms)
        all_idx.extend([edge.Index] * len(split_geoms))

    return gpd.GeoDataFrame({"line index": all_idx, "geometry": all_splits})


def raster2split(vector_data, raster_data, bands=[1], inplace=False):
    returned_gdf = vector_data if inplace else vector_data.copy()
    for band in bands:
        band_data = raster_data.read(band)
        geom_raster_values = []
        for split in vector_data.geometry:
            cell_x, cell_y = get_cell_indices(
                split,
                raster_data.width,
                raster_data.height,
                list(raster_data.transform),
            )
            geom_raster_values.append(band_data[cell_x, cell_y])
        returned_gdf.insert(
            len(vector_data.columns), f"band{band}", geom_raster_values
        )
    return returned_gdf
