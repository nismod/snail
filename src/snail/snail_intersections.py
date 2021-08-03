import geopandas as gpd
from shapely.geometry import LineString, Polygon
from shapely.ops import polygonize

from snail.intersections import split as split_one_geom
from snail.intersections import split_polygon as split_one_polygon
from snail.intersections import get_cell_indices


def split(vector_data, raster_data):
    all_splits = []
    all_idx = []
    for idx, geom in zip(vector_data.index, vector_data["geometry"]):
        if type(geom) == LineString:
            split_geoms = split_one_geom(
                geom,
                raster_data.width,
                raster_data.height,
                list(raster_data.transform),
            )
        elif type(geom) == Polygon:
            splits = split_one_polygon(
                geom,
                raster_data.width,
                raster_data.height,
                list(raster_data.transform),
            )
            split_geoms = list(polygonize(splits))
        else:
            msg = f"""Incorrect geometry type {type(geom)}: only LineString and Polygon are supported."""
            raise ValueError(msg)
        all_splits.extend(split_geoms)
        all_idx.extend([idx] * len(split_geoms))

    return gpd.GeoDataFrame({"line index": all_idx, "geometry": all_splits})


def raster2split(vector_data, raster_data, bands=[1], inplace=False):
    returned_gdf = vector_data.copy() if inplace else vector_data
    for band in bands:
        band_data = raster_data.read(band)
        geom_raster_values = []
        for split in vector_data["geometry"]:
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
