import geopandas as gpd
from shapely.geometry import LineString, Polygon
from shapely.ops import polygonize

from snail.intersections import split_linestring
from snail.intersections import split_polygon
from snail.intersections import get_cell_indices


def split_linestrings(vector_data, raster_data):
    all_splits = []
    all_idx = []
    for idx, geom in zip(vector_data.index, vector_data.geometry):
        if type(geom) != LineString:
            msg = f"Incorrect geometry type {type(geom)}, expected LineString."
            raise ValueError(msg)
        split_geoms = split_linestring(
            geom,
            raster_data.width,
            raster_data.height,
            list(raster_data.transform),
        )
        all_splits.extend(split_geoms)
        all_idx.extend([idx] * len(split_geoms))

    return gpd.GeoDataFrame({"line index": all_idx, "geometry": all_splits})


def split_polygons(vector_data, raster_data):
    all_splits = []
    all_idx = []
    for idx, geom in zip(vector_data.index, vector_data.geometry):
        if type(geom) != Polygon:
            msg = f"Incorrect geometry type {type(geom)}, expected LineString."
            raise ValueError(msg)
        splits = split_polygon(
            geom,
            raster_data.width,
            raster_data.height,
            list(raster_data.transform),
        )
        split_geoms = list(polygonize(splits))
        all_splits.extend(split_geoms)
        all_idx.extend([idx] * len(split_geoms))

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
