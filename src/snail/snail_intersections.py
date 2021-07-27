import geopandas as gpd

from snail.intersections import split as split_one_geom
from snail.intersections import get_cell_indices


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


def raster2split(vector_data, raster_data, bands=[1]):
    # Let's not catch the user by surprise by modifying the
    # dataframe in place. Return an enriched copy instead.
    enriched_gdf = vector_data.copy()
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
        enriched_gdf.insert(len(vector_data.columns), f"band{band}", geom_raster_values)
    return enriched_gdf
