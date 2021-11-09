import logging
import os

import geopandas as gpd
import rasterio
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


def raster2split(
    vector_data,
    rasters,
    width,
    height,
    transform,
    band_number=1,
    inplace=False,
):
    """Associate raster data to split vector data.

    Positional arguments:
    vector_data -- Split vector data (geometries) according to raster
    grid (geopandas.GeoDataFrame)
    rasters -- Mapping of key to raster file name (dict of str: str)
    width -- Raster data width (int)
    height -- Raster data height (int)
    transform -- Raster data transform (list[float])
    band_number -- Band number to be read from raster data files (int)
    inplace -- Whether or not to modify the input vector data in place

    Returns: Split vector data with added "cell_index" and "<key>"
    columns with one <key> column for each item in rasters
    dictionary. (geopandas.GeoDataFrame)

    """
    df = vector_data if inplace else vector_data.copy()

    def get_indices(geom):
        x, y = get_cell_indices(geom, width, height, transform)
        x = x % width
        y = y % height
        return [x, y]

    df["cell_index"] = df.geometry.apply(get_indices)
    for key, fname in rasters.items():
        with rasterio.open(fname) as dataset:
            band_data = dataset.read(band_number)
            df[key] = df.cell_index.apply(lambda i: band_data[i[1], i[0]])
    return df
