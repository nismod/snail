import geopandas as gpd
import rasterio

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


def raster2split(raster_file, splits_file, output_file, bands=[1]):
    raster_data = rasterio.open(raster_file)
    splits_data = gpd.read_file(splits_file)

    for band in bands:
        band_data = raster_data.read(band)
        geom_raster_values = []
        for split in splits_data["geometry"]:
            cell_x, cell_y = get_cell_indices(
                split,
                raster_data.width,
                raster_data.height,
                list(raster_data.transform),
            )
            geom_raster_values.append(band_data[cell_x, cell_y])
        splits_data.insert(
            len(splits_data.columns), f"band{band}", geom_raster_values
        )
    splits_data.to_file(output_file)
