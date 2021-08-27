"""Intersect networks with hazards

Purpose
-------

Intersect hazards and network line and point geometries with hazatd polygons

Write final results to Shapefiles

Input data requirements
-----------------------

1. Correct paths to all files and correct input parameters

2. Shapefiles of network edges or nodes with attributes:
    - edge_id or node_id - String/Integer/Float Edge ID or Node ID of network
    - geometry - Shapely geometry of edges as LineStrings or nodes as Points

3. Shapefile of hazards with attributes:
    - geometry - Shapely geometry of hazard Polygon

Results
-------

1. Edge shapefiles with attributes:
    - edge_id - String name of intersecting edge ID
    - length - Float length of intersection of edge LineString and hazard Polygon
    - geometry - Shapely LineString geometry of intersection of edge LineString and hazard Polygon

2. Node Shapefile with attributes:
    - node_id - String name of intersecting node ID
    - geometry - Shapely Point geometry of intersecting node ID

"""
import itertools
import os
import sys

import geopandas as gpd
import pandas as pd
from boltons.iterutils import pairwise
from geopy.distance import vincenty
from shapely.geometry import Polygon


def line_length(line, ellipsoid="WGS-84"):
    """Length of a line in meters, given in geographic coordinates.

    Adapted from https://gis.stackexchange.com/questions/4022/looking-for-a-pythonic-way-to-calculate-the-length-of-a-wkt-linestring#answer-115285

    Args:
        line: a shapely LineString object with WGS-84 coordinates.

        ellipsoid: string name of an ellipsoid that `geopy` understands (see http://geopy.readthedocs.io/en/latest/#module-geopy.distance).

    Returns:
        Length of line in kilometers.
    """
    if line.geometryType() == "MultiLineString":
        return sum(line_length(segment) for segment in line)

    return sum(
        vincenty(
            tuple(reversed(a)), tuple(reversed(b)), ellipsoid=ellipsoid
        ).kilometers
        for a, b in pairwise(line.coords)
    )


def extract_value_from_gdf(x, gdf_sindex, gdf, column_name):
    """Access value

    Parameters
    ----------
    x
        row of dataframe
    gdf_sindex
        spatial index of dataframe of which we want to extract the value
    gdf
        GeoDataFrame of which we want to extract the value
    column_name
        column that contains the value we want to extract

    Returns
    -------
    extracted value from other gdf
    """
    return gdf.loc[list(gdf_sindex.intersection(x.bounds[:2]))][
        column_name
    ].values[0]


def networkedge_polygon_intersection(
    edge_shapefile,
    hazard_shapefile,
    output_shapefile,
    edge_id_column,
    polygon_id_column,
    edge_length_column,
    crs={"init": "epsg:4326"},
):
    """Intersect network edges and hazards and write results to shapefiles

    Parameters
    ----------
    edge_shapefile
        Shapefile of network LineStrings
    hazard_shapefile
        Shapefile of hazard Polygons
    output_shapefile
        String name of edge-hazard shapefile for storing results


    Outputs
    -------
    output_shapefile
        - edge_id - String name of intersecting edge ID
        - length - Float length of intersection of edge LineString and hazard Polygon
        - geometry - Shapely LineString geometry of intersection of edge LineString and hazard Polygon
    """
    print(
        "* Starting {} and {} intersections".format(
            edge_shapefile, hazard_shapefile
        )
    )
    line_gpd = gpd.read_file(edge_shapefile)
    line_gpd.to_crs(crs)
    poly_gpd = gpd.read_file(hazard_shapefile)
    poly_gpd.to_crs(crs)
    if polygon_id_column is None:
        polygon_id_column = "ID"
        poly_gpd["ID"] = poly_gpd.index.values.tolist()

    if len(line_gpd.index) > 0 and len(poly_gpd.index) > 0:
        line_gpd.columns = map(str.lower, line_gpd.columns)
        poly_gpd.columns = map(str.lower, poly_gpd.columns)

        line_bounding_box = line_gpd.total_bounds
        line_bounding_box_coord = list(
            itertools.product(
                [line_bounding_box[0], line_bounding_box[2]],
                [line_bounding_box[1], line_bounding_box[3]],
            )
        )
        line_bounding_box_geom = Polygon(line_bounding_box_coord)
        line_bounding_box_gpd = gpd.GeoDataFrame(
            pd.DataFrame([[1], [line_bounding_box_geom]]).T, crs=crs
        )
        line_bounding_box_gpd.columns = ["ID", "geometry"]

        poly_bounding_box = poly_gpd.total_bounds
        poly_bounding_box_coord = list(
            itertools.product(
                [poly_bounding_box[0], poly_bounding_box[2]],
                [poly_bounding_box[1], poly_bounding_box[3]],
            )
        )
        poly_bounding_box_geom = Polygon(poly_bounding_box_coord)
        poly_bounding_box_gpd = gpd.GeoDataFrame(
            pd.DataFrame([[1], [poly_bounding_box_geom]]).T, crs=crs
        )
        poly_bounding_box_gpd.columns = ["ID", "geometry"]

        poly_sindex = poly_bounding_box_gpd.sindex

        selected_polys = poly_bounding_box_gpd.iloc[
            list(
                poly_sindex.intersection(
                    line_bounding_box_gpd.geometry.iloc[0].bounds
                )
            )
        ]
        if len(selected_polys.index) > 0:
            data = []
            poly_sindex = poly_gpd.sindex
            for lines in line_gpd.itertuples():
                intersected_polys = poly_gpd.iloc[
                    list(poly_sindex.intersection(lines.geometry.bounds))
                ]
                for poly in intersected_polys.itertuples():
                    if (
                        (lines.geometry.intersects(poly.geometry) is True)
                        and (poly.geometry.is_valid is True)
                        and (lines.geometry.is_valid is True)
                    ):
                        if line_length(lines.geometry) > 1e-3:
                            geom = lines.geometry.intersection(poly.geometry)
                            if crs == {"init": "epsg:4326"}:
                                data.append(
                                    {
                                        edge_id_column: getattr(
                                            lines, edge_id_column
                                        ),
                                        polygon_id_column: getattr(
                                            poly, polygon_id_column
                                        ),
                                        edge_length_column: 1000.0
                                        * line_length(geom),
                                        "geometry": geom,
                                    }
                                )
                            else:
                                data.append(
                                    {
                                        edge_id_column: getattr(
                                            lines, edge_id_column
                                        ),
                                        polygon_id_column: getattr(
                                            poly, polygon_id_column
                                        ),
                                        edge_length_column: 1000.0
                                        * geom.length,
                                        "geometry": geom,
                                    }
                                )
                        else:
                            data.append(
                                {
                                    edge_id_column: getattr(
                                        lines, edge_id_column
                                    ),
                                    polygon_id_column: getattr(
                                        poly, polygon_id_column
                                    ),
                                    edge_length_column: 0,
                                    "geometry": lines.geometry,
                                }
                            )
            if data:
                intersections_data = gpd.GeoDataFrame(
                    data,
                    columns=[edge_id_column, edge_length_column, "geometry"],
                    crs=crs,
                )
                intersections_data.to_file(output_shapefile, driver="GPKG")

                del intersections_data

    del line_gpd, poly_gpd


def networknode_polygon_intersection(
    node_shapefile,
    hazard_shapefile,
    output_shapefile,
    node_id_column,
    polygon_id_column,
    crs={"init": "epsg:4326"},
):
    """Intersect network nodes and hazards and write results to shapefiles

    Parameters
    ----------
    node_shapefile
        Shapefile of network Points
    hazard_shapefile
        Shapefile of hazard Polygons
    output_shapefile
        String name of node-hazard shapefile for storing results


    Outputs
    -------
    output_shapefile
        - node_id - String name of intersecting node ID
        - geometry - Shapely Point geometry of intersecting node ID
    """
    print(
        "* Starting {} and {} intersections".format(
            node_shapefile, hazard_shapefile
        )
    )
    point_gpd = gpd.read_file(node_shapefile)
    point_gpd.to_crs(crs)
    # if 'id' in point_gpd.columns.values.tolist():
    #     point_gpd.rename(columns={'id':node_id_column},inplace=True)
    poly_gpd = gpd.read_file(hazard_shapefile)
    poly_gpd.to_crs(crs)
    if polygon_id_column is None:
        polygon_id_column = "ID"
        poly_gpd["ID"] = poly_gpd.index.values.tolist()

    if len(point_gpd.index) > 0 and len(poly_gpd.index) > 0:
        point_gpd.columns = map(str.lower, point_gpd.columns)
        poly_gpd.columns = map(str.lower, poly_gpd.columns)
        data = []
        # create spatial index
        poly_sindex = poly_gpd.sindex
        for points in point_gpd.itertuples():
            intersected_polys = poly_gpd.iloc[
                list(poly_sindex.intersection(points.geometry.bounds))
            ]
            if len(intersected_polys.index) > 0:
                for poly in intersected_polys.itertuples():
                    if (
                        (points.geometry.intersects(poly.geometry) is True)
                        and (poly.geometry.is_valid is True)
                        and (points.geometry.is_valid is True)
                    ):
                        data.append(
                            {
                                node_id_column: getattr(
                                    points, node_id_column
                                ),
                                polygon_id_column: getattr(
                                    poly, polygon_id_column
                                ),
                                "geometry": points.geometry,
                            }
                        )
        if data:
            intersections_data = gpd.GeoDataFrame(
                data,
                columns=[node_id_column, polygon_id_column, "geometry"],
                crs=crs,
            )
            intersections_data.to_file(output_shapefile, driver="GPKG")

            del intersections_data

    del point_gpd, poly_gpd


def intersect_networks_and_polygons(
    hazard_dir,
    network_file_path,
    network_file_name,
    output_file_path,
    network_id_column,
    polygon_id_column,
    network_length_column=None,
    network_type="",
):
    """Walk through all hazard files and select network-hazard intersection criteria

    Parameters
    ----------
    hazard_dir : str
        name of directory where all hazard shapefiles are stored
    network_file_path : str
        name of directory where network shapefile is stored
    network_file_name : str
        name network shapefile
    output_file_path : str
        name of directory where network-hazard instersection result shapefiles will be stored
    network_type : str
        values of 'edges' or 'nodes'


    Outputs
    -------
    Edge or Node shapefiles

    """
    for root, dirs, files in os.walk(hazard_dir):
        for file in files:
            if not file.startswith("._"):
                if file.endswith(".shp") or file.endswith(".gpkg"):
                    hazard_file = os.path.join(root, file)
                    out_shp_name = (
                        network_file_name.replace(".shp", "")
                        + "_"
                        + file.replace(".shp", ".gpkg")
                    )
                    output_file = os.path.join(output_file_path, out_shp_name)
                    if network_type == "edges":
                        networkedge_polygon_intersection(
                            network_file_path,
                            hazard_file,
                            output_file,
                            network_id_column,
                            polygon_id_column,
                            network_length_column,
                        )
                    elif network_type == "nodes":
                        networknode_polygon_intersection(
                            network_file_path,
                            hazard_file,
                            output_file,
                            network_id_column,
                            polygon_id_column,
                        )


def edgenode_boundary_intersections(
    network_shapefile,
    polygon_shapefile,
    hazard_dictionary,
    data_dictionary,
    network_id_column,
    polygon_id_column,
    network_type="nodes",
):
    """Intersect network edges/nodes and boundary Polygons to collect boundary and hazard attributes

    Parameters
        - network_shapefile - Shapefile of edge LineStrings or node Points
        - polygon_shapefile - Shapefile of boundary Polygons
        - hazard_dictionary - Dictionary of hazard attributes
        - data_dictionary - Dictionary of network-hazard-boundary intersection attributes
        - network_type - String value -'edges' or 'nodes' - Default = 'nodes'
        - name_province - String name of province if needed - Default = ''

    Outputs
        data_dictionary - Dictionary of network-hazard-boundary intersection attributes:
            - edge_id/node_id - String name of intersecting edge ID or node ID
            - length - Float length of intersection of edge LineString and hazard Polygon: Only for edges
            - province_id - String/Integer ID of Province
            - province_name - String name of Province in English
            - district_id - String/Integer ID of District
            - district_name - String name of District in English
            - commune_id - String/Integer ID of Commune
            - commune_name - String name of Commune in English
            - hazard_attributes - Dictionary of all attributes from hazard dictionary
    """
    line_gpd = gpd.read_file(network_shapefile)
    poly_gpd = gpd.read_file(polygon_shapefile)

    if len(line_gpd.index) > 0 and len(poly_gpd.index) > 0:
        print(network_shapefile, len(line_gpd.index), len(poly_gpd.index))
        line_gpd.columns = map(str.lower, line_gpd.columns)
        poly_gpd.columns = map(str.lower, poly_gpd.columns)

        # create spatial index
        poly_sindex = poly_gpd.sindex

        poly_sindex = poly_gpd.sindex
        for l_index, lines in line_gpd.iterrows():
            intersected_polys = poly_gpd.iloc[
                list(poly_sindex.intersection(lines.geometry.bounds))
            ]
            for p_index, poly in intersected_polys.iterrows():
                if (
                    (lines["geometry"].intersects(poly["geometry"]) is True)
                    and (poly.geometry.is_valid is True)
                    and (lines.geometry.is_valid is True)
                ):
                    if network_type == "edges":
                        value_dictionary = {
                            network_id_column: lines[network_id_column],
                            "length": 1000.0
                            * line_length(
                                lines["geometry"].intersection(
                                    poly["geometry"]
                                )
                            ),
                            polygon_id_column: poly[polygon_id_column],
                        }
                    elif network_type == "nodes":
                        value_dictionary = {
                            network_id_column: lines[network_id_column],
                            polygon_id_column: poly[polygon_id_column],
                        }

                    data_dictionary.append(
                        {**value_dictionary, **hazard_dictionary}
                    )

    del line_gpd, poly_gpd
    return data_dictionary


def create_hazard_attributes_for_network(
    intersection_dir,
    climate_scenario,
    year,
    sector,
    hazard_files,
    hazard_df,
    thresholds,
    commune_shape,
    network_id_column,
    network_length_column,
    network_type="",
):
    """Extract results of network edges/nodes and hazard intersections to collect
    network-hazard intersection attributes

        - Combine with boundary Polygons to collect network-hazard-boundary intersection attributes
        - Write final results to an Excel sheet

    Parameters
    ----------
    intersection_dir : str
        Path to Directory where the network-hazard shapefile results are stored
    sector : str
        name of transport mode
    hazard_files : list[str]
        names of all hazard files
    hazard_df : pandas.DataFrame
        hazard attributes
    bands : list[int]
        integer values of hazard bands
    thresholds : list[int]
        integer values of hazard thresholds
    commune_shape
        Shapefile of commune boundaries and attributes
    network_type : str, optional
        value -'edges' or 'nodes': Default = 'nodes'
    name_province : str, optional
        name of province if needed: Default = ''

    Returns
    -------
    data_df : pandas.DataFrame
        network-hazard-boundary intersection attributes:
            - edge_id/node_id - String name of intersecting edge ID or node ID
            - length - Float length of intersection of edge LineString and hazard Polygon: Only for edges
            - province_id - String/Integer ID of Province
            - province_name - String name of Province in English
            - district_id - String/Integer ID of District
            - district_name - String name of District in English
            - commune_id - String/Integer ID of Commune
            - commune_name - String name of Commune in English
            - sector - String name of transport mode
            - hazard_type - String name of hazard type
            - model - String name of hazard model
            - year - String name of hazard year
            - climate_scenario - String name of hazard scenario
            - probability - Float/String value of hazard probability
            - band_num - Integer value of hazard band
            - min_val - Integer value of minimum value of hazard threshold
            - max_val - Integer value of maximum value of hazard threshold
            - length - Float length of intersection of edge LineString and hazard Polygon: Only for edges

    """
    data_dict = []
    for root, dirs, files in os.walk(intersection_dir):
        for file in files:
            if not file.startswith("._"):
                if file.endswith(".shp") or file.endswith(".gpkg"):
                    hazard_dict = {}
                    hazard_dict["sector"] = sector
                    hazard_shp = os.path.join(root, file)
                    hz_file = file.split("_")
                    hz_file = [
                        hz_file[h - 1] + "_" + hz_file[h]
                        for h in range(len(hz_file))
                        if "1in" in hz_file[h]
                    ][0]
                    hazard_dict["hazard_type"] = hazard_df.loc[
                        hazard_df.file_name == hz_file
                    ].hazard_type.values[0]
                    hazard_dict["model"] = hazard_df.loc[
                        hazard_df.file_name == hz_file
                    ].model.values[0]
                    hazard_dict["year"] = hazard_df.loc[
                        hazard_df.file_name == hz_file
                    ].year.values[0]
                    hazard_dict["climate_scenario"] = hazard_df.loc[
                        hazard_df.file_name == hz_file
                    ].climate_scenario.values[0]
                    hazard_dict["probability"] = hazard_df.loc[
                        hazard_df.file_name == hz_file
                    ].probability.values[0]

                    hazard_thrs = [
                        (thresholds[t], thresholds[t + 1])
                        for t in range(len(thresholds) - 1)
                        if "{0}-{1}".format(thresholds[t], thresholds[t + 1])
                        in file
                    ][0]
                    hazard_dict["min_hazard"] = hazard_thrs[0]
                    hazard_dict["max_hazard"] = hazard_thrs[1]

                    data_dict = spatial_scenario_selection(
                        hazard_shp,
                        commune_shape,
                        hazard_dict,
                        data_dict,
                        network_id_column,
                        network_type=network_type,
                    )

                    print("Done with file", file)

    data_df = pd.DataFrame(data_dict)
    data_df_cols = data_df.columns.values.tolist()
    if network_length_column in data_df_cols:
        selected_cols = [
            cols for cols in data_df_cols if cols != network_length_column
        ]
        data_df = (
            data_df.groupby(selected_cols)[network_length_column]
            .sum()
            .reset_index()
        )

    return data_df


def match_admin_layer_ids(
    admin_1_file,
    admin_2_file,
    admin_1_id,
    admin_2_id,
    encoding="utf-8",
    crs={"init": "epsg:4326"},
):
    # admin_1 = gpd.read_file(admin_1_file,encoding='utf-8')
    # admin_1 = admin_1.to_crs(crs)

    # admin_2 = gpd.read_file(admin_2_file,encoding='utf-8')
    # admin_2 = admin_2.to_crs(crs)

    # if admin_1_id == admin_2_id:
    # 	admin_1.rename(columns={admin_1_id:'admin1_id'},inplace=True)
    # 	admin_1_id = 'admin1_id'
    # 	admin_2.rename(columns={admin_2_id:'admin2_id'},inplace=True)
    # 	admin_2_id = 'admin2_id'
    # if 'Geometry' in admin_1.columns.values.tolist():
    # 	admin_1.rename(columns={'Geometry':'geom_type'},inplace=True)
    # if 'Geometry' in admin_2.columns.values.tolist():
    # 	admin_2.rename(columns={'Geometry':'geom_type'},inplace=True)

    sindex_admin_1 = admin_1.sindex

    admin_2["geometry_centroid"] = admin_2.geometry.centroid
    admin_2_centroids = admin_2[[admin_2_id, "geometry_centroid"]]
    admin_2_centroids.rename(
        columns={"geometry_centroid": "geometry"}, inplace=True
    )
    admin_2_matches = gpd.sjoin(
        admin_2_centroids,
        admin_1[[admin_1_id, "geometry"]],
        how="inner",
        op="within",
    ).reset_index()
    no_admin_2 = [
        x
        for x in admin_2[admin_2_id].tolist()
        if x not in admin_2_matches[admin_2_id].tolist()
    ]

    admin_2.drop("geometry_centroid", axis=1, inplace=True)
    if no_admin_2:
        remain_admin_2 = admin_2[admin_2[admin_2_id].isin(no_admin_2)]
        remain_admin_2[admin_1_id] = remain_admin_2.progress_apply(
            lambda x: extract_value_from_gdf(
                x, sindex_admin_1, admin_1_id, admin_1_id
            ),
            axis=1,
        )

        admin_2_matches = pd.concat(
            [admin_2_matches, remain_admin_2],
            axis=0,
            sort="False",
            ignore_index=True,
        )

    admin_2 = pd.merge(
        admin_2,
        admin_2_matches[[admin_2_id, admin_1_id]],
        how="left",
        on=[admin_2_id],
    )

    return admin_2
