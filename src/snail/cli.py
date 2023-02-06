import argparse
import logging
import os
import warnings

from collections import namedtuple
from os.path import splitext

import fiona
import geopandas
import numpy
import pandas
import rasterio

from shapely.geometry import mapping, shape
from shapely.ops import linemerge, polygonize
from snail.core.intersections import (
    get_cell_indices,
    split_linestring,
    split_polygon,
)
from snail.multi_intersections import (
    split_linestrings,
    split_polygons,
    raster2split,
)


def snail(args=None):
    parser = argparse.ArgumentParser(prog="snail")
    subparsers = parser.add_subparsers(help="additional help")

    parser_split = subparsers.add_parser(
        "split", help="Split vector features on a regular grid"
    )
    parser_split.add_argument(
        "-v",
        "--vector",
        type=str,
        required=True,
        help="File with vector features to split",
    )
    parser_split.add_argument(
        "-l",
        "--layer",
        type=str,
        required=False,
        help="Layer in file with vector features to split",
    )
    parser_split.add_argument(
        "-r",
        "--raster",
        type=str,
        required=False,
        nargs="+",
        help="Raster file/s to use as definition of splitting grid",
    )
    parser_split.add_argument(
        "-t",
        "--transform",
        type=str,
        required=False,
        nargs=6,
        help="Transform definition of splitting grid",
    )
    parser_split.add_argument(
        "-a",
        "--attribute",
        action="store_true",
        help="Attribute raster values to split output",
    )
    parser_split.set_defaults(func=split)

    parser_process = subparsers.add_parser(
        "process", help="Split vectors and attribute raster values"
    )
    parser_process.add_argument(
        "-vs", "--vectors", type=str, help="CSV file with vector layers"
    )
    parser_process.add_argument(
        "-rs", "--rasters", type=str, help="CSV file with raster layers"
    )
    parser_process.set_defaults(func=process)


def split(args):
    raster_data = rasterio.open(args.raster)
    vector_data = geopandas.read_file(args.vector)

    geom_type = vector_data.iloc[0].geometry.geom_type

    if geom_type == "LineString":
        new_gdf = split_linestrings(vector_data, raster_data)
    elif geom_type == "Polygon":
        new_gdf = split_polygons(vector_data, raster_data)
    else:
        raise ValueError(
            f"Could not process vector data of type {geom_type}, expected Polygon or LineString"
        )

    new_gdf.to_file(args.output)


def snail_raster2split(args):
    if isinstance(args.raster, str):
        args.raster = [
            args.raster,
        ]

    with rasterio.open(args.raster[0]) as dataset:
        raster_width = dataset.width
        raster_height = dataset.height
        raster_transform = list(dataset.transform)
    # Make key: filename dict with filename (without ext) as key
    rasters = {splitext(k)[0]: v for (k, v) in zip(args.raster, args.raster)}
    vector_data = geopandas.read_file(args.vector)

    new_gdf = raster2split(
        vector_data,
        rasters,
        width=raster_width,
        height=raster_height,
        transform=raster_transform,
        band_number=args.band,
        inplace=True,
    )
    new_gdf.to_file(args.output)


def process(args):
    networks_csv = args.vectors
    hazards_csv = args.rasters

    # Ignore writing-to-parquet warnings
    warnings.filterwarnings(
        "ignore", message=".*initial implementation of Parquet.*"
    )
    # Ignore reading-geopackage warnings
    warnings.filterwarnings(
        "ignore", message=".*Sequential read of iterator was interrupted.*"
    )

    # Enable info logging
    logging.basicConfig(format="%(asctime)s %(message)s", level=logging.INFO)
    logging.info("Start.")
    main(".", networks_csv, hazards_csv)
    logging.info("Done.")


def main(data_path, networks_csv, hazards_csv):
    # read transforms, record with hazards
    hazards = pandas.read_csv(hazards_csv)
    hazard_slug = os.path.basename(hazards_csv).replace(".csv", "")
    hazard_transforms, transforms = read_transforms(hazards, data_path)
    hazard_transforms.to_csv(
        hazards_csv.replace(".csv", "__with_transforms.csv"), index=False
    )

    # read networks
    networks = pandas.read_csv(networks_csv)

    for network_path in networks.path:
        fname = os.path.join(data_path, network_path)
        out_fname = os.path.join(
            data_path,
            "..",
            "results",
            "hazard_asset_intersection",
            os.path.basename(network_path).replace(
                ".gpkg", f"_splits__{hazard_slug}.gpkg"
            ),
        )
        pq_fname_nodes = out_fname.replace(".gpkg", "__nodes.geoparquet")
        pq_fname_edges = out_fname.replace(".gpkg", "__edges.geoparquet")
        pq_fname_areas = out_fname.replace(".gpkg", "__areas.geoparquet")

        # skip if output is there already (not using gpkg currently)
        # if os.path.exists(out_fname):
        #     logging.info("Skipping %s. Already exists: %s", os.path.basename(fname), out_fname)
        #     continue

        logging.info("Processing %s", os.path.basename(fname))
        layers = fiona.listlayers(fname)
        logging.info("Layers: %s", layers)

        if "nodes" in layers:
            # skip if output is there already
            if os.path.exists(pq_fname_nodes):
                logging.info(
                    "Skipping %s. Already exists: %s",
                    os.path.basename(fname),
                    pq_fname_nodes,
                )
            else:
                # look up nodes cell index
                nodes = geopandas.read_file(fname, layer="nodes")
                logging.info("Node CRS %s", nodes.crs)
                nodes = process_nodes(
                    nodes, transforms, hazard_transforms, data_path
                )
                # nodes.to_file(out_fname, driver="GPKG", layer="nodes")
                nodes.to_parquet(pq_fname_nodes)

        if "edges" in layers:
            # skip if output is there already
            if os.path.exists(pq_fname_edges):
                logging.info(
                    "Skipping %s. Already exists: %s",
                    os.path.basename(fname),
                    pq_fname_edges,
                )
            else:
                # split lines
                edges = geopandas.read_file(fname, layer="edges")
                logging.info("Edge CRS %s", edges.crs)
                edges = process_edges(
                    edges, transforms, hazard_transforms, data_path
                )
                # edges.to_file(out_fname, driver="GPKG", layer="edges")
                edges.to_parquet(pq_fname_edges)

        if "areas" in layers:
            # skip if output is there already
            if os.path.exists(pq_fname_areas):
                logging.info(
                    "Skipping %s. Already exists: %s",
                    os.path.basename(fname),
                    pq_fname_areas,
                )
            else:
                # split polygons
                areas = geopandas.read_file(fname, layer="areas")
                logging.info("Area CRS %s", areas.crs)
                areas = explode_multi(areas)
                areas = process_areas(
                    areas, transforms, hazard_transforms, data_path
                )
                # areas.to_file(out_fname, driver="GPKG", layer="areas")
                areas.to_parquet(pq_fname_areas)


# Helper class to store a raster transform and CRS
Transform = namedtuple("Transform", ["crs", "width", "height", "transform"])


def associate_raster(
    df, key, fname, cell_index_col="cell_index", band_number=1
):
    with rasterio.open(fname) as dataset:
        band_data = dataset.read(band_number)
        df[key] = df[cell_index_col].apply(lambda i: band_data[i[1], i[0]])


def read_transforms(hazards, data_path):
    transforms = []
    transform_id = 0
    hazard_transforms = []
    for hazard in hazards.itertuples():
        hazard_path = hazard.path
        with rasterio.open(
            os.path.join(data_path, "hazards", hazard_path)
        ) as dataset:
            crs = dataset.crs
            width = dataset.width
            height = dataset.height
            transform = Transform(crs, width, height, tuple(dataset.transform))
        # add transform to list if not present
        if transform not in transforms:
            transforms.append(transform)
            transform_id = transform_id + 1

        # record hazard/transform details
        hazard_transform_id = transforms.index(transform)
        hazard_transform = hazard._asdict()
        del hazard_transform["Index"]
        hazard_transform["transform_id"] = hazard_transform_id
        hazard_transform["width"] = transform.width
        hazard_transform["height"] = transform.height
        hazard_transform["crs"] = str(transform.crs)
        hazard_transform["transform_0"] = transform.transform[0]
        hazard_transform["transform_1"] = transform.transform[1]
        hazard_transform["transform_2"] = transform.transform[2]
        hazard_transform["transform_3"] = transform.transform[3]
        hazard_transform["transform_4"] = transform.transform[4]
        hazard_transform["transform_5"] = transform.transform[5]
        hazard_transforms.append(hazard_transform)
    hazard_transforms = pandas.DataFrame(hazard_transforms)

    return hazard_transforms, transforms


def process_nodes(nodes, transforms, hazard_transforms, data_path):
    # lookup per transform
    for i, t in enumerate(transforms):
        # transform to grid
        crs_df = nodes.to_crs(t.crs)
        # save cell index for fast lookup of raster values
        crs_df[f"cell_index_{i}"] = crs_df.geometry.progress_apply(
            lambda geom: get_indices(geom, t)
        )
        # transform back
        nodes = crs_df.to_crs(nodes.crs)

    # associate hazard values
    for hazard in hazard_transforms.itertuples():
        logging.info("Hazard %s transform %s", hazard.key, hazard.transform_id)
        fname = os.path.join(data_path, "hazards", hazard.path)
        cell_index_col = f"cell_index_{hazard.transform_id}"
        associate_raster(nodes, hazard.key, fname, cell_index_col)

    # split and drop tuple columns so GPKG can save
    for i, t in enumerate(transforms):
        nodes = split_index_column(nodes, f"cell_index_{i}")
        nodes.drop(columns=f"cell_index_{i}", inplace=True)
    return nodes


def try_merge(geom):
    if geom.geom_type == "MultiLineString":
        geom = linemerge(geom)
    return geom


def process_edges(edges, transforms, hazard_transforms, data_path):
    # handle multilinestrings
    edges.geometry = edges.geometry.apply(try_merge)
    geom_types = edges.geometry.apply(lambda g: g.geom_type)
    logging.info(geom_types.value_counts())
    edges = explode_multi(edges)

    # split edges per transform
    for i, t in enumerate(transforms):
        # transform to grid
        crs_df = edges.to_crs(t.crs)
        crs_df = split_df(crs_df, t)
        # save cell index for fast lookup of raster values
        crs_df[f"cell_index_{i}"] = crs_df.geometry.progress_apply(
            lambda geom: get_indices(geom, t)
        )
        # transform back
        edges = crs_df.to_crs(edges.crs)

    # associate hazard values
    for hazard in hazard_transforms.itertuples():
        logging.info("Hazard %s transform %s", hazard.key, hazard.transform_id)
        fname = os.path.join(data_path, "hazards", hazard.path)
        cell_index_col = f"cell_index_{hazard.transform_id}"
        associate_raster(edges, hazard.key, fname, cell_index_col)

    # split and drop tuple columns so GPKG can save
    for i, t in enumerate(transforms):
        edges = split_index_column(edges, f"cell_index_{i}")
        edges.drop(columns=f"cell_index_{i}", inplace=True)

    return edges


def split_df(df, t):
    # split
    core_splits = []
    for edge in df.itertuples():
        # split edge
        splits = split_linestring(
            edge.geometry, t.width, t.height, t.transform
        )
        # add to collection
        for s in splits:
            s_dict = edge._asdict()
            del s_dict["Index"]
            s_dict["geometry"] = s
            core_splits.append(s_dict)
    logging.info(f"Split {len(df)} edges into {len(core_splits)} pieces")
    sdf = geopandas.GeoDataFrame(core_splits, crs=t.crs, geometry="geometry")
    return sdf


def process_areas(areas, transforms, hazard_transforms, data_path):
    # split areas per transform
    for i, t in enumerate(transforms):
        # transform to grid
        crs_df = areas.to_crs(t.crs)
        crs_df = split_area_df(crs_df, t)
        # save cell index for fast lookup of raster values
        crs_df[f"cell_index_{i}"] = crs_df.geometry.progress_apply(
            lambda geom: get_indices(geom, t)
        )
        # transform back
        areas = crs_df.to_crs(areas.crs)

    # associate hazard values
    for hazard in hazard_transforms.itertuples():
        logging.info("Hazard %s transform %s", hazard.key, hazard.transform_id)
        fname = os.path.join(data_path, "hazards", hazard.path)
        cell_index_col = f"cell_index_{hazard.transform_id}"
        associate_raster(areas, hazard.key, fname, cell_index_col)

    # split and drop tuple columns so GPKG can save
    for i, t in enumerate(transforms):
        areas = split_index_column(areas, f"cell_index_{i}")
        areas.drop(columns=f"cell_index_{i}", inplace=True)

    return areas


def explode_multi(df):
    items = []
    geoms = []
    for item in df.itertuples(index=False):
        if item.geometry.geom_type in (
            "MultiPoint",
            "MultiLineString",
            "MultiPolygon",
        ):
            for part in item.geometry:
                items.append(item._asdict())
                geoms.append(part)
        else:
            items.append(item._asdict())
            geoms.append(item.geometry)

    df = geopandas.GeoDataFrame(items, crs=df.crs, geometry=geoms)
    return df


def set_precision(geom, precision):
    """Set geometry precision"""
    geom_mapping = mapping(geom)
    geom_mapping["coordinates"] = numpy.round(
        numpy.array(geom_mapping["coordinates"]), precision
    )
    return shape(geom_mapping)


def split_area_df(df, t):
    # split
    core_splits = []
    for area in df.itertuples():
        # split area
        splits = split_polygon(area.geometry, t.width, t.height, t.transform)
        # round to high precision (avoid floating point errors)
        splits = [set_precision(s, 9) for s in splits]
        # to polygons
        splits = list(polygonize(splits))
        # add to collection
        for s in splits:
            s_dict = area._asdict()
            del s_dict["Index"]
            s_dict["geometry"] = s
            core_splits.append(s_dict)
    logging.info(f"  Split {len(df)} areas into {len(core_splits)} pieces")
    sdf = geopandas.GeoDataFrame(core_splits)
    sdf.crs = t.crs
    return sdf


def get_indices(geom, t):
    x, y = get_cell_indices(geom, t.width, t.height, t.transform)

    # Raise error if cell index would be out of bounds
    if x > t.width or x < 0:
        raise ValueError
    if y > t.height or y < 0:
        raise ValueError

    # Or - set out-of-bounds value (-1,-1) if cell index would be out of bounds
    # if x > t.width or x < 0 or y > t.height or y < 0:
    #     x = -1
    #     y = -1
    return (x, y)


def split_index_column(df, prefix):
    df[f"{prefix}_x"] = df[prefix].apply(lambda i: i[0])
    df[f"{prefix}_y"] = df[prefix].apply(lambda i: i[1])
    return df
