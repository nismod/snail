"""Summarise network-hazard intersections

Purpose
-------

Collect network-hazard intersection attributes
    - Combine with boundary Polygons to collect network-hazard-boundary intersection attributes
    - Write final results to an Excel sheet

Input data requirements
-----------------------

1. Correct paths to all files and correct input parameters

2. Shapefiles of network-hazard intersections results with attributes:
    - edge_id or node_id - String/Integer/Float Edge ID or Node ID of network
    - length - Float length of edge intersecting with hazards
    - geometry - Shapely geometry of edges as LineString or nodes as Points

3. Shapefile of administrative boundaries of Argentina with attributes:
    - province_i - String/Integer ID of Province
    - pro_name_e - String name of Province in English
    - district_i - String/Integer ID of District
    - dis_name_e - String name of District in English
    - commune_id - String/Integer ID of Commune
    - name_eng - String name of Commune in English
    - geometry - Shapely geometry of boundary Polygon

4. Excel sheet of hazard attributes with attributes:
    - hazard_type - String name of hazard type
    - model - String name of hazard model
    - year - String name of hazard year
    - climate_scenario - String name of hazard scenario
    - probability - Float/String value of hazard probability
    - band_num - Integer value of hazard band
    - min_val - Integer value of minimum value of hazard threshold
    - max_val - Integer value of maximum value of hazard threshold

Results
-------

1. Excel sheet of network-hazard-boundary intersection with attributes:
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

"""
import itertools
import os
import sys

import geopandas as gpd
import pandas as pd
from shapely.geometry import Polygon
from tqdm import tqdm


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


def spatial_scenario_selection(
    network_shapefile,
    polygon_dataframe,
    hazard_dictionary,
    data_dictionary,
    network_id_column,
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
    poly_gpd = polygon_dataframe

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
                            "province_id": poly["province_id"],
                            "province_name": poly["province_name"],
                            "department_id": poly["department_id"],
                            "department_name": poly["department_name"],
                        }
                    elif network_type == "nodes":
                        value_dictionary = {
                            network_id_column: lines[network_id_column],
                            "province_id": poly["province_id"],
                            "province_name": poly["province_name"],
                            "department_id": poly["department_id"],
                            "department_name": poly["department_name"],
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
            if file.endswith(".shp"):
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
                hazard_dict["min_depth"] = hazard_thrs[0]
                hazard_dict["max_depth"] = hazard_thrs[1]

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
    if "length" in data_df_cols:
        selected_cols = [cols for cols in data_df_cols if cols != "length"]
        data_df = data_df.groupby(selected_cols)["length"].sum().reset_index()

    return data_df


def main():
    """Collect results

    1. Specify the paths from where you to read and write:
        - Input data
        - Intermediate calcuations data
        - Output results

    2. Supply input data and parameters
        - Names of the three Provinces - List of string types
        - Names of modes - List of strings
        - Names of output modes - List of strings
        - Names of hazard bands - List of integers
        - Names of hazard thresholds - List of integers
        - Condition 'Yes' or 'No' is the users wants to process results

    3. Give the paths to the input data files:
        - Commune boundary and stats data shapefile
        - Hazard datasets description Excel file
        - String name of sheet in hazard datasets description Excel file
    """
    tqdm.pandas()
    incoming_data_path = "path/to/project/inputdata/"
    data_path = "path/to/project/inputdata/"
    output_path = "path/to/project/outputdata/"

    # Supply input data and parameters
    modes = [
        "road",
        "rail",
        "bridge",
        "air",
        "port",
    ]  # change this to your network datasets
    modes_shapefile_names = [
        "road_edges",
        "rail_edges",
        "bridge_edges",
        "air_nodes",
        "port_nodes",
    ]  # change this to your network datasets
    point_asset_modes = [
        "air",
        "port",
    ]  # change this to your network point asset datasets
    line_asset_modes = [
        "road",
        "rail",
        "bridge",
    ]  # change this to your network line asset datasets
    modes_id_cols = [
        "edge_id",
        "edge_id",
        "bridge_id",
        "node_id",
        "node_id",
    ]  # This should match the ID column name in your network datasets
    climate_scenarios = [
        "historical",
        "rcp4p5",
        "rcp8p5",
    ]  # change this to GLOFRIS scenarios

    # We assume here that we will extract only flood outlines with depth:
    #   50cm-1m
    #   1m-2m
    #   2m-3m
    #   3m-4m
    #   4m-999m (>4m)
    # You can change this if you want to get different flood depth bands and ranges
    # Change the values in the thresholds and thresholds_label accordingly
    thresholds = ["50cm", "1m", "2m", "3m", "4m", "999m"]

    # Give the paths to the input data files
    # load provinces and get geometry of the right province
    print("* Reading provinces dataframe")
    # change this to the name of the province shapefile for China
    province_path = os.path.join(
        incoming_data_path,
        "admin_boundaries_and_census",
        "province",
        "province.shp",
    )
    provinces = gpd.read_file(province_path, encoding="utf-8")
    provinces = provinces.to_crs({"init": "epsg:4326"})
    # The file should have a column named 'province_id' and a column named 'province_name'
    # If these columns are given some other name then rename them as per the next line below
    # provinces.rename(columns={'OBJECTID':'province_id','nombre':'province_name'},inplace=True)
    sindex_provinces = provinces.sindex

    """Assign provinces to zones
    """
    print("* Reading department dataframe")
    # change this to the name of the department shapefile for China
    # If you do not have a department level admin then skip this
    zones_path = os.path.join(
        incoming_data_path,
        "admin_boundaries_and_census",
        "departaments",
        "Departaments.shp",
    )
    zones = gpd.read_file(zones_path, encoding="utf-8")
    zones = zones.to_crs({"init": "epsg:4326"})
    # The file should have a column named 'department_id' and a column named 'department_name'
    # If these columns are given some other name then rename them as per the next line below
    # zones.rename(columns={'OBJECTID':'department_id','Name':'department_name'},inplace=True)

    zones["geometry_centroid"] = zones.geometry.centroid
    zones_centriods = zones[
        ["department_id", "department_name", "geometry_centroid"]
    ]
    zones_centriods.rename(
        columns={"geometry_centroid": "geometry"}, inplace=True
    )
    zone_matches = gpd.sjoin(
        zones_centriods,
        provinces[["province_id", "province_name", "geometry"]],
        how="inner",
        op="within",
    ).reset_index()
    no_zones = [
        x
        for x in zones["department_id"].tolist()
        if x not in zone_matches["department_id"].tolist()
    ]

    zones.drop("geometry_centroid", axis=1, inplace=True)
    if no_zones:
        remain_zones = zones[zones["department_id"].isin(no_zones)]
        remain_zones["province_name"] = remain_zones.progress_apply(
            lambda x: extract_value_from_gdf(
                x, sindex_provinces, provinces, "province_name"
            ),
            axis=1,
        )
        remain_zones["province_id"] = remain_zones.progress_apply(
            lambda x: extract_value_from_gdf(
                x, sindex_provinces, provinces, "province_id"
            ),
            axis=1,
        )

        zone_matches = pd.concat(
            [zone_matches, remain_zones],
            axis=0,
            sort="False",
            ignore_index=True,
        )

    zones = pd.merge(
        zones,
        zone_matches[["department_id", "province_id", "province_name"]],
        how="left",
        on=["department_id"],
    )

    hazard_description_file = os.path.join(
        data_path, "flood_data", "GLOFRIS", "glofris_files.csv"
    )  # change this to the GLOFRIS data path and the csv file describing the data

    # Specify the output files and paths to be created
    output_dir = os.path.join(output_path, "hazard_scenarios")
    if os.path.exists(output_dir) == False:
        os.mkdir(output_dir)

    # Read hazard datasets desciptions
    print("* Reading hazard datasets desciptions")
    hazard_df = pd.read_csv(hazard_description_file)
    hazard_files = hazard_df["file_name"].values.tolist()

    # Process national scale results
    print("* Processing national scale results")
    for m in range(len(modes)):
        mode_data_df = []
        for cl_sc in range(len(climate_scenarios)):
            intersection_dir = os.path.join(
                output_path,
                "networks_hazards_intersection_shapefiles",
                "{}_hazard_intersections".format(modes[m]),
                climate_scenarios[cl_sc],
            )

            if modes[m] in ["road", "rail", "bridge"]:
                ntype = "edges"
            else:
                ntype = "nodes"
            data_df = create_hazard_attributes_for_network(
                intersection_dir,
                climate_scenarios[cl_sc],
                years[cl_sc],
                modes[m],
                hazard_files,
                hazard_df,
                thresholds,
                zones,
                modes_id_cols[m],
                network_type=ntype,
            )

            mode_data_df.append(data_df)
            del data_df

        mode_data_df = pd.concat(
            mode_data_df, axis=0, sort="False", ignore_index=True
        )
        data_path = os.path.join(
            output_dir, "{}_hazard_intersections.csv".format(modes[m])
        )
        mode_data_df.to_csv(data_path, index=False, encoding="utf-8")
        del mode_data_df


if __name__ == "__main__":
    main()
