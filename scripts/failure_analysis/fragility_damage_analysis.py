"""Estimating the fragility and damages to roads 
"""

import os
import sys
import pandas as pd
import geopandas as gpd
import numpy as np
import convert_hazard_data as chd
import network_intersections as ni
from scipy import integrate
from tqdm import tqdm

tqdm.pandas()


def damage_function_roads_v1(flood_depth, multiplication_factor):
    """
    Damage curve adapted from:
        Tariq, M.A.U.R., Hoes, O. and Ashraf, M., 2014.
        Risk-based design of dike elevation employing alternative enumeration.
        Journal of Water Resources Planning and Management, 140(8), p.05014002.
    Inputs:
        flood_depth: String name of flood depth column
        multiplication_factor: A factor to upscale or downscale the damage percentage
    Returns:
        damage_percentage: The percentage of damage corresponding to a given flood depth
    """
    if flood_depth <= 0.5:
        damage_percent = 40.0 * flood_depth
    elif 0.5 < flood_depth <= 1.0:
        damage_percent = 30.0 * flood_depth + 5.0
    else:
        damage_percent = 10.0 * flood_depth + 25.0

    damage_percent = multiplication_factor * damage_percent
    if damage_percent > 100:
        damage_percent = 100

    return damage_percent


def damage_function_roads_v2(flood_depth, road_type, multiplication_factor):
    """
    Damage curve adapted from:
        Tariq, M.A.U.R., Hoes, O. and Ashraf, M., 2014.
        Risk-based design of dike elevation employing alternative enumeration.
        Journal of Water Resources Planning and Management, 140(8), p.05014002.
    Inputs:
        flood_depth: String name of flood depth column
        road_type: String - paved or unpaved
        multiplication_factor: A factor to upscale or downscale the damage percentage
    Returns:
        damage_percentage: The percentage of damage corresponding to a given flood depth
    """
    if road_type == "paved":
        if flood_depth <= 0.1:
            damage_percent = 0.0 * flood_depth
        elif 0.1 < flood_depth <= 0.25:
            damage_percent = 100.0 / 15.0 * (flood_depth - 0.1)
        elif 0.25 < flood_depth <= 0.5:
            damage_percent = 4.0 * flood_depth
        elif 0.5 < flood_depth <= 0.75:
            damage_percent = 12.0 * flood_depth - 4.0
        else:
            damage_percent = 20.0 * flood_depth - 10.0
    elif road_type == "unpaved":
        if flood_depth <= 0.1:
            damage_percent = 10.0 * flood_depth
        elif 0.1 < flood_depth <= 0.25:
            damage_percent = 10.0 / 15.0 * (100.0 * flood_depth + 5.0)
        elif 0.25 < flood_depth <= 0.5:
            damage_percent = 120.0 * flood_depth - 10.0
        elif 0.5 < flood_depth <= 1.00:
            damage_percent = 80.0 * flood_depth + 10.0
        else:
            damage_percent = 20.0 * flood_depth + 70.0

    damage_percent = multiplication_factor * damage_percent
    if damage_percent > 100:
        damage_percent = 100

    return damage_percent


def expected_risks(
    dataframe,
    index_columns,
    probability_column,
    risk_column,
    expected_risk_column,
    probability_threshold=0,
):
    dataframe = dataframe.set_index(index_columns)
    index_values = list(set(dataframe.index.values.tolist()))
    expected_risks = []
    for cl in index_values:
        prob_risk = sorted(
            list(
                zip(
                    dataframe.loc[cl, probability_column].values.tolist(),
                    dataframe.loc[cl, risk_column].values.tolist(),
                )
            ),
            key=lambda x: x[0],
        )
        if probability_threshold > 0:
            prob_risk = [
                pr for pr in prob_risk if pr[0] < probability_threshold
            ]
        if len(prob_risk) > 1:
            risks = integrate.trapz(
                np.array([x[1] for x in prob_risk]),
                np.array([x[0] for x in prob_risk]),
            )
        elif len(prob_risk) == 1:
            risks = 0.5 * prob_risk[0][0] * prob_risk[0][1]
        else:
            risks = 0
        expected_risks.append(tuple(list(cl) + [risks]))

    expected_risks = pd.DataFrame(
        expected_risks, columns=index_columns + [expected_risk_column]
    )

    return expected_risks


def damage_costs_per_area_argentina(x, rehab_costs, design_width, asset_type):
    if asset_type == "road":
        return 1.0e3 * cost_rehab / design_width
    else:
        return 1.0e6 * cost_rehab / design_width


def damage_costs_per_area_vietnam(
    x, rehab_costs, length_factor, national=False
):
    """Estimate the total cost and benefits for a road segment. This function is used within a
    pandas apply

    Parameters
    ----------
    x
        a row from the road segment dataframe that we are considering
    rehab_costs
        rehabilitation costs after a disaster

    Returns
    -------
    uncer_output : list
        outcomes for the initial adaptation costs of this road segment
    tot_uncer_output : list
        outcomes for the total adaptation costs of this road segment
    rel_share : list
        relative share of each factor in the initial adaptation cost of this road segment
    tot_rel_share : list
        relative share of each factor in the total adaptation cost of this road segment
    bc_ratio : list
        benefit cost ratios for this road segment

    """
    # Identify terrain type of the road
    if x.terrain.lower().strip() == "mountain" or x.asset_type == "Bridge":
        ter_type = "mountain"
    elif x.terrain.lower().strip() == "flat":
        ter_type = "flat"

    rehab_costs["rate_m"] = length_factor * rehab_costs.basic_cost
    # Identify asset type, which is the main driver of the costs
    if (x.asset_type == "Expressway") | (
        (national == True) & (x.road_class == 1)
    ):
        rehab_cost = rehab_costs.loc[("Expressway", ter_type), "rate_m"]
        rehab_corr = rehab_costs.loc[("Expressway", ter_type), "design_width"]
    elif (x.asset_type == "National roads") | (
        (national == True) & (x.road_class == 2)
    ):
        rehab_cost = rehab_costs.loc[
            ("National  2x Carriageway", ter_type), "rate_m"
        ]
        rehab_corr = rehab_costs.loc[
            ("National  2x Carriageway", ter_type), "design_width"
        ]
    elif (x.asset_type == "National roads") | (
        (national == True) & (x.road_class == 3)
    ):
        rehab_cost = rehab_costs.loc[
            ("National  1x Carriageway", ter_type), "rate_m"
        ]
        rehab_corr = rehab_costs.loc[
            ("National  1x Carriageway", ter_type), "design_width"
        ]
    elif (x.asset_type == "Provincial roads") | (
        (national == True) & (x.road_class == 4)
    ):
        rehab_cost = rehab_costs.loc[("Provincial", ter_type), "rate_m"]
        rehab_corr = rehab_costs.loc[("Provincial", ter_type), "design_width"]
    elif (
        (x.asset_type == "Urban roads/Named roads")
        | (x.asset_type == "Boulevard")
    ) | ((national == True) & (x.road_class == 5)):
        rehab_cost = rehab_costs.loc[("District", ter_type), "rate_m"]
        rehab_corr = rehab_costs.loc[("District", ter_type), "design_width"]
    elif (x.asset_type == "Other roads") | (
        (national == True) & (x.road_class == 6)
    ):
        rehab_cost = rehab_costs.loc[("Commune", ter_type), "rate_m"]
        rehab_corr = rehab_costs.loc[("Commune", ter_type), "design_width"]
    elif x.asset_type == "Bridge":
        rehab_cost = rehab_costs.rate_m.max()
        rehab_corr = rehab_costs.design_width.max()
    else:
        rehab_cost = rehab_costs.rate_m.min()
        rehab_corr = rehab_costs.design_width.min()

    rehab_cost = rehab_cost / rehab_corr

    return rehab_cost


def damage_costs_per_area_tanzania(
    x, rehab_costs, length_factor, national=False
):
    if x.road_cond == "paved":
        return 1
    else:
        return 2
