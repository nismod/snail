"""Plotting flooding intersection results for transport networks
The results show different outcomes with respect to:
1. Total exposed lengths of networks
2. Expected exposures for different flood depths
3. Variations across climate models and emission scenarios
4. Spatial adminstrative level results of locations and extents of exposures   
"""

import os
import sys
import pandas as pd
import geopandas as gpd
import numpy as np
import convert_hazard_data as chd
import network_intersections as ni
import exposure_stats as es
import fragility_damage_analysis as fda
from tqdm import tqdm

tqdm.pandas()


def main():
    base_path = "/Users/raghavpant/Desktop/OIA_projects"
    country = "Vietnam"
    data_path = os.path.join(base_path, country, "data")
    results_path = os.path.join(base_path, country, "results")
    figures_path = os.path.join(base_path, country, "figures")
    flood_results_folder = os.path.join(results_path, "flood_scenarios")
    admin_intersection_folder = os.path.join(results_path, "network_stats")
    summarised_results = os.path.join(results_path, "flood_exposure_stats")
    if os.path.exists(summarised_results) == False:
        os.mkdir(summarised_results)

    network_description = [
        {
            "sector": "road",
            "sector_folder": "post_processed_networks",
            "edge_shapefile": "road_edges.shp",
            "edge_csvfile": "road_edges.csv",
            "edge_id": "edge_id",
            "from_node": "from_node",
            "to_node": "to_node",
            "node_shapefile": "road_nodes.shp",
            "node_id": "node_id",
            "flood_length_column": "length",  # The actual length is in meters!
            "length_factor": 0.001,
            "network_type": "edges",
        },
    ]
    flood_description = {
        "min_depth_column": "min_val",
        "max_depth_column": "max_val",
        "probability_column": "probability",
        "climate_model_factors": [
            "hazard_type",
            "model",
            "year",
            "climate_scenario",
            "subsistence",
            "percentile",
        ],
        "years": [2018, 2030, 2050],
    }

    network_admin_intersections_description = {
        "admin_file": "national_scale_stats.xlsx",
        "admin_id": "commune_id",
        "network_sheets": ["road"],
        "length_columns": ["length", "length"],
    }
    length_factor = 0.001
    hazard_type = "fluvial flooding"
    # es.network_flood_exposure_stats(data_path,flood_results_folder,admin_intersection_folder,summarised_results,
    #     network_description,flood_description,network_admin_intersections_description,save_files=True)

    # calculate the EADs
    road_edges = gpd.read_file(
        os.path.join(data_path, "post_processed_networks", "road_edges.shp")
    )
    print(road_edges)

    # load cost file
    adapt_path = os.path.join(data_path, "Adaptation_options")
    # load rehab costs
    rehab_costs = pd.read_excel(
        os.path.join(adapt_path, "adaptation_costs_road_types.xlsx"),
        sheet_name="rehabilitation_costs",
        index_col=[0, 1],
    ).fillna(0)
    print(rehab_costs)

    road_edges["cost_persqm"] = road_edges.progress_apply(
        lambda x: fda.damage_costs_per_area_vietnam(
            x, rehab_costs, length_factor, national=True
        ),
        axis=1,
    )

    road_edges = road_edges[
        [
            "edge_id",
            "road_class",
            "road_cond",
            "terrain",
            "width",
            "cost_persqm",
        ]
    ]

    exposure_results = pd.read_csv(
        os.path.join(flood_results_folder, "road_hazard_intersections.csv")
    )
    # Converting the flooded lengths from meters to kilometers
    # exposure_results[length_column] = length_factor*exposure_results[length_column]
    # exposure_results['return_period'] = 1.0/exposure_results[probability_column]
    min_depth_column = "min_val"
    max_depth_column = "max_val"
    flood_length_column = "length"
    exposure_results["min_flood_depth"] = exposure_results.progress_apply(
        lambda x: es.process_flood_depths(x, min_depth_column), axis=1
    )
    exposure_results["max_flood_depth"] = exposure_results.progress_apply(
        lambda x: es.process_flood_depths(x, max_depth_column), axis=1
    )

    exposure_results = pd.merge(
        exposure_results, road_edges, how="left", on=["edge_id"]
    )
    # print (exposure_results)

    exposure_results["min_damage_percent"] = exposure_results.progress_apply(
        lambda x: fda.damage_function_roads_v2(
            x.min_flood_depth, x.road_cond, 1
        ),
        axis=1,
    )
    exposure_results["max_damage_percent"] = exposure_results.progress_apply(
        lambda x: fda.damage_function_roads_v2(
            x.max_flood_depth, x.road_cond, 1
        ),
        axis=1,
    )
    exposure_results["min_damage_cost"] = (
        0.01
        * exposure_results["min_damage_percent"]
        * exposure_results["width"]
        * exposure_results[flood_length_column]
        * exposure_results["cost_persqm"]
    )
    exposure_results["max_damage_cost"] = (
        0.01
        * exposure_results["max_damage_percent"]
        * exposure_results["width"]
        * exposure_results[flood_length_column]
        * exposure_results["cost_persqm"]
    )
    print(exposure_results)

    exposure_results = exposure_results[
        exposure_results.hazard_type == "fluvial flooding"
    ]
    climate_model_factors = [
        "edge_id",
        "hazard_type",
        "model",
        "year",
        "climate_scenario",
    ]
    probability_column = "probability"
    exposure_min = (
        exposure_results.groupby(climate_model_factors)[
            ["min_flood_depth", "min_damage_percent", "length", "probability"]
        ]
        .min()
        .reset_index()
    )
    exposure_min.rename(
        columns={
            "length": "min_flood_length",
            "probability": "min_probability",
        },
        inplace=True,
    )
    exposure_max = (
        exposure_results.groupby(climate_model_factors)[
            ["max_flood_depth", "max_damage_percent", "length", "probability"]
        ]
        .max()
        .reset_index()
    )
    exposure_max.rename(
        columns={
            "length": "max_flood_length",
            "probability": "max_probability",
        },
        inplace=True,
    )
    exposure_minmax = pd.merge(
        exposure_min, exposure_max, how="left", on=climate_model_factors
    )
    del exposure_min, exposure_max

    df = (
        exposure_results.groupby(climate_model_factors + [probability_column])[
            "min_damage_cost"
        ]
        .sum()
        .reset_index()
    )
    min_ead = fda.expected_risks(
        df,
        climate_model_factors,
        probability_column,
        "min_damage_cost",
        "min_ead",
        probability_threshold=0,
    )

    df = (
        exposure_results.groupby(climate_model_factors + [probability_column])[
            "max_damage_cost"
        ]
        .sum()
        .reset_index()
    )
    max_ead = fda.expected_risks(
        df,
        climate_model_factors,
        probability_column,
        "max_damage_cost",
        "max_ead",
        probability_threshold=0,
    )
    del df

    ead = pd.merge(min_ead, max_ead, how="left", on=climate_model_factors)
    ead = pd.merge(ead, road_edges, how="left", on=["edge_id"])
    ead = pd.merge(ead, exposure_minmax, how="left", on=climate_model_factors)

    road_cols = [
        "edge_id",
        "road_class",
        "road_cond",
        "terrain",
        "width",
        "cost_persqm",
    ]
    exposure_cols = [
        "hazard_type",
        "model",
        "year",
        "climate_scenario",
        "min_flood_depth",
        "max_flood_depth",
        "min_flood_length",
        "max_flood_length",
        "min_probability",
        "max_probability",
        "min_damage_percent",
        "max_damage_percent",
        "min_ead",
        "max_ead",
    ]
    ead[road_cols + exposure_cols].to_csv(
        os.path.join(summarised_results, "roads_expected_annual_damages.csv"),
        index=False,
    )


if __name__ == "__main__":
    main()
