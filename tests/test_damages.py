"""Test damage assessment"""

import os
from itertools import pairwise
from unittest.mock import Mock

import numpy
import pandas
import pytest
from numpy.testing import assert_allclose
from scipy.integrate import trapezoid

from snail.damages import (
    DamageCurve,
    PiecewiseLinearDamageCurve,
    _integrate_ead_samples,
    calculate_direct_damages,
)


@pytest.fixture
def curve():
    curve_data = pandas.DataFrame(
        {"intensity": [0.0, 10, 20, 30], "damage": [0, 0.1, 0.2, 1.0]}
    )
    return PiecewiseLinearDamageCurve(curve_data)


def test_linear_curve(curve):
    # check inheritance
    assert isinstance(curve, DamageCurve)


def test_equality(curve):
    curve_copy = PiecewiseLinearDamageCurve(
        pandas.DataFrame(
            {
                "intensity": [0.0, 10, 20, 30],
                "damage": [0, 0.1, 0.2, 1.0],
            }
        )
    )
    assert curve == curve_copy


def test_read_csv(curve):
    fname = os.path.join(
        os.path.dirname(__file__),
        "integration",
        "piecewise-linear-damage-curve.csv",
    )
    curve_from_file = PiecewiseLinearDamageCurve.from_csv(fname)
    assert curve == curve_from_file


def test_read_csv_arguments():
    fname = os.path.join(
        os.path.dirname(__file__),
        "integration",
        "paved-road-flood-depth-damage.csv",
    )
    actual = PiecewiseLinearDamageCurve.from_csv(
        fname,
        intensity_col="inundation_depth_(m)",
        damage_col="road_unpaved",
        sep="\t",
    )
    expected = PiecewiseLinearDamageCurve(
        pandas.DataFrame(
            {
                "intensity": [0, 1, 2, 3, 4, 5],
                "damage": [0, 0.28, 0.46, 0.64, 0.82, 1],
            }
        )
    )
    assert actual == expected


def test_read_excel(curve):
    fname = os.path.join(
        os.path.dirname(__file__),
        "integration",
        "piecewise-linear-damage-curve.xlsx",
    )
    curve_from_file = PiecewiseLinearDamageCurve.from_excel(fname)
    assert curve == curve_from_file


def test_read_excel_arguments():
    fname = os.path.join(
        os.path.dirname(__file__),
        "integration",
        "paved-road-flood-depth-damage.xlsx",
    )
    actual = PiecewiseLinearDamageCurve.from_excel(
        fname,
        sheet_name="flood",
        intensity_col="inundation_depth_(m)",
        damage_col="road_unpaved",
    )
    expected = PiecewiseLinearDamageCurve(
        pandas.DataFrame(
            {
                "intensity": [0, 1, 2, 3, 4, 5],
                "damage": [0, 0.28, 0.46, 0.64, 0.82, 1],
            }
        )
    )
    assert actual == expected


def test_linear_curve_pass_through(curve):
    # check specified intensities give specified damages
    assert_allclose(curve.damage_fraction(curve.intensity), curve.damage)


def test_linear_curve_interpolation(curve):
    # sense-check interpolation
    intensities = numpy.array([5, 15, 25])
    expected = numpy.array([0.05, 0.15, 0.6])
    actual = curve.damage_fraction(intensities)
    assert_allclose(actual, expected)


def test_linear_curve_out_of_bounds(curve):
    # sense-check out-of-bounds
    intensities = numpy.array(
        [-numpy.inf, -999, -0.0, 0, 30, 999, numpy.inf, numpy.nan]
    )
    expected = numpy.array([0, 0, 0, 0, 1, 1, 1, numpy.nan])
    actual = curve.damage_fraction(intensities)
    assert_allclose(actual, expected)


def test_linear_curve_translate_y(curve):
    increased = curve.translate_y(0.1)
    expected = numpy.array([0.1, 0.2, 0.3, 1.0, 1.0])
    assert_allclose(increased.damage, expected)

    expected = numpy.array([0, 10, 20, 28.75, 30])
    assert_allclose(increased.intensity, expected)


def test_linear_curve_translate_y_down(curve):
    decreased = curve.translate_y(-0.1)
    expected = numpy.array([0, 0, 0.1, 0.9])
    assert_allclose(decreased.damage, expected)

    expected = numpy.array([0, 10, 20, 30])
    assert_allclose(decreased.intensity, expected)


def test_linear_curve_scale_y(curve):
    increased = curve.scale_y(2)
    expected = numpy.array([0, 0.2, 0.4, 1.0, 1.0])
    assert_allclose(increased.damage, expected)

    expected = numpy.array([0, 10, 20, 23.75, 30])
    assert_allclose(increased.intensity, expected)


def test_linear_curve_scale_y_down(curve):
    decreased = curve.scale_y(0.1)
    expected = numpy.array([0, 0.01, 0.02, 0.1])
    assert_allclose(decreased.damage, expected)

    expected = numpy.array([0, 10, 20, 30])
    assert_allclose(decreased.intensity, expected)


def test_linear_curve_translate_x(curve):
    increased = curve.translate_x(5)
    expected = numpy.array([0, 0.1, 0.2, 1.0])
    assert_allclose(increased.damage, expected)

    expected = numpy.array([5, 15, 25, 35])
    assert_allclose(increased.intensity, expected)


def test_linear_curve_translate_x_down(curve):
    decreased = curve.translate_x(-5)
    expected = numpy.array([0, 0.1, 0.2, 1.0])
    assert_allclose(decreased.damage, expected)

    expected = numpy.array([-5, 5, 15, 25])
    assert_allclose(decreased.intensity, expected)


def test_linear_curve_scale_x(curve):
    increased = curve.scale_x(2)
    expected = numpy.array([0, 0.1, 0.2, 1.0])
    assert_allclose(increased.damage, expected)

    expected = numpy.array([0, 20, 40, 60])
    assert_allclose(increased.intensity, expected)


def test_linear_curve_scale_x_down(curve):
    decreased = curve.scale_x(0.1)
    expected = numpy.array([0, 0.1, 0.2, 1.0])
    assert_allclose(decreased.damage, expected)

    expected = numpy.array([0, 1, 2, 3])
    assert_allclose(decreased.intensity, expected)


def test_interpolation(curve):
    curve_050 = curve.scale_y(0.5)
    curve_100 = curve
    expected = curve.scale_y(0.75)
    actual = PiecewiseLinearDamageCurve.interpolate(curve_050, curve_100, 0.5)
    assert actual == expected


def test_interpolation_unaligned():
    a = PiecewiseLinearDamageCurve(
        pandas.DataFrame(
            {
                "intensity": [0.0, 10.0],
                "damage": [0.0, 1.0],
            }
        )
    )
    b = PiecewiseLinearDamageCurve(
        pandas.DataFrame(
            {
                "intensity": [0.0, 5.0, 10.0],
                "damage": [0.0, 0.1, 0.8],
            }
        )
    )
    expected = PiecewiseLinearDamageCurve(
        pandas.DataFrame(
            {
                "intensity": [0.0, 5.0, 10.0],
                "damage": [0.0, 0.3, 0.9],
            }
        )
    )
    actual = PiecewiseLinearDamageCurve.interpolate(a, b, 0.5)
    assert actual == expected


def _constant_curve(value):
    return PiecewiseLinearDamageCurve(
        pandas.DataFrame({"intensity": [0.0, 1.0], "damage": [value, value]})
    )


def test_direct_damages_deterministic_single_curve():
    splits = pandas.DataFrame(
        {
            "asset": ["a", "a", "b"],
            "length": [2.0, 3.0, 1.0],
            "rp_10": [0.5, 0.5, 0.0],
            "rp_100": [1.0, 1.0, 0.5],
        }
    )
    curve = PiecewiseLinearDamageCurve(
        pandas.DataFrame({"intensity": [0.0, 1.0], "damage": [0.0, 0.25]})
    )

    result = calculate_direct_damages(
        splits,
        asset_id="asset",
        quantity="length",
        intensity_columns={10: "rp_10", 100: "rp_100"},
        curves={"central": curve},
        curve_weights={"central": 1.0},
        cost_mean=100.0,
        cost_asset_cv=0.0,
        n_samples=20,
        quantiles=(0.05, 0.25, 0.5, 0.75, 0.95),
        seed=42,
    )

    actual = result.quantiles.set_index(["asset", "intensity_case"])["q_50"]
    # a, RP10: 0.25 * 0.5 * (2 + 3) * 100 = 62.5
    assert actual.loc[("a", 10)] == pytest.approx(62.5)
    # a, RP100: 0.25 * 1.0 * (2 + 3) * 100 = 125
    assert actual.loc[("a", 100)] == pytest.approx(125.0)
    # b, RP10: 0.25 * 0.0 * 1 * 100 = 0
    assert actual.loc[("b", 10)] == pytest.approx(0.0)
    # b, RP100: 0.25 * 0.5 * 1 * 100 = 12.5
    assert actual.loc[("b", 100)] == pytest.approx(12.5)
    assert result.metadata["seed"] == 42


def test_direct_damages_calculates_ead_quantiles():
    splits = pandas.DataFrame(
        {"asset": ["a"], "length": [1.0], "frequent": [1.0], "rare": [3.0]}
    )
    curve = PiecewiseLinearDamageCurve(
        pandas.DataFrame({"intensity": [0.0, 10.0], "damage": [0.0, 1.0]})
    )
    result = calculate_direct_damages(
        splits,
        asset_id="asset",
        quantity="length",
        intensity_columns={"rp2": "frequent", "rp10": "rare"},
        return_periods={"rp10": 10.0, "rp2": 2.0},
        curves={"linear": curve},
        curve_weights={"linear": 1.0},
        cost_mean=10.0,
        cost_asset_cv=0.0,
        quantiles=(0.5,),
        ead_quantiles=(0.05, 0.5, 0.95),
        n_samples=10,
        seed=3,
    )

    assert result.quantiles is not None
    assert len(result.quantiles) == 2
    assert result.ead_quantiles is not None
    assert result.ead_quantiles.loc[0, "n_splits"] == 1
    # d(h, c) = h/10 * c, with c=10
    # Supplied (probability, damage) pairs are (0.1, 3) and (0.5, 1).
    # Integrating 1 - 2*ln(p/0.5)/ln(5) gives 0.2 + 0.8/ln(5).
    # Add the rare tail rectangle 0.1*3; omit probabilities above 0.5.
    expected_ead = 0.5 + 0.8 / numpy.log(5.0)
    assert result.ead_quantiles.loc[0, "q_05"] == pytest.approx(expected_ead)
    assert result.ead_quantiles.loc[0, "q_50"] == pytest.approx(expected_ead)
    assert result.ead_quantiles.loc[0, "q_95"] == pytest.approx(expected_ead)

    ead_only = calculate_direct_damages(
        splits,
        asset_id="asset",
        quantity="length",
        intensity_columns={"rp2": "frequent", "rp10": "rare"},
        return_periods={"rp10": 10.0, "rp2": 2.0},
        curves={"linear": curve},
        curve_weights={"linear": 1.0},
        cost_mean=10.0,
        cost_asset_cv=0.0,
        quantiles=None,
        ead_quantiles=(0.5,),
        n_samples=10,
        seed=3,
    )
    assert ead_only.quantiles is None
    assert ead_only.ead_quantiles.loc[0, "q_50"] == pytest.approx(expected_ead)


@pytest.fixture(params=[(0.0, 1.0), (4.0, 2.0), (7.0, 0.0), (0.0, 0.0)])
def log_linear_ead_case(request):
    a, b = request.param
    # D(p) = a - b*ln(p), with one simulation sample.
    return_periods = {"rp100": 100.0, "rp25": 25.0, "rp5": 5.0, "rp2": 2.0}
    sampled_damage = {
        case: numpy.array([a - b * numpy.log(1.0 / period)])
        for case, period in return_periods.items()
    }
    return a, b, return_periods, sampled_damage


def test_integrate_ead_samples_matches_log_linear_numerical_case(log_linear_ead_case):
    a, b, return_periods, sampled_damage = log_linear_ead_case
    actual = _integrate_ead_samples(sampled_damage, return_periods)[0]

    p_min = 1.0 / max(return_periods.values())
    p_max = 1.0 / min(return_periods.values())
    grid = numpy.geomspace(p_min, p_max, 100_001)
    grid_damage = a - b * numpy.log(grid)
    numerical = trapezoid(grid_damage, x=grid) + grid_damage[0] * p_min
    assert_allclose(actual, numerical, rtol=1e-8, atol=1e-12)


def test_integrate_ead_samples_uses_intermediate_points():
    probabilities = numpy.array([0.01, 0.04, 0.2, 0.5])
    damage = numpy.array([100.0, 80.0, 20.0, 0.0])
    cases = ["rare", "uncommon", "common", "frequent"]
    sampled_damage = {case: numpy.array([value]) for case, value in zip(cases, damage)}
    return_periods = dict(zip(cases, 1.0 / probabilities))
    actual = _integrate_ead_samples(sampled_damage, return_periods)[0]

    # Include every supplied probability, so numerical trapezoids do not cross corners.
    grid = numpy.unique(
        numpy.concatenate(
            [
                numpy.geomspace(p_lo, p_hi, 100_001)
                for p_lo, p_hi in pairwise(probabilities)
            ]
        )
    )
    grid_damage = numpy.interp(numpy.log(grid), numpy.log(probabilities), damage)
    numerical = trapezoid(grid_damage, x=grid) + damage[0] * probabilities[0]
    assert_allclose(actual, numerical, rtol=1e-8, atol=1e-12)


@pytest.mark.parametrize("return_period", [1.0, 20.0])
def test_integrate_ead_samples_single_event_contributes_only_tail(return_period):
    damage = numpy.array([0.0, 2.0, 100.0])
    actual = _integrate_ead_samples({"event": damage}, {"event": return_period})
    assert_allclose(actual, damage / return_period)


def test_ead_quantiles_are_calculated_from_integrated_samples():
    sampled_damage = {
        "rp10": numpy.array([0.0, 10.0, 100.0]),
        "rp2": numpy.array([100.0, 0.0, 10.0]),
    }
    ead_samples = _integrate_ead_samples(sampled_damage, {"rp10": 10.0, "rp2": 2.0})

    # Log-linear interval integrals plus each sample's rare tail rectangle.
    expected = numpy.array(
        [
            50.0 - 40.0 / numpy.log(5.0),
            4.0 / numpy.log(5.0),
            5.0 + 36.0 / numpy.log(5.0),
        ]
    )
    assert_allclose(ead_samples, expected)
    assert numpy.quantile(ead_samples, 0.5) == pytest.approx(expected[0])

    # Per-period medians are 10 at both RP10 (p=0.1) and RP2 (p=0.5):
    # Constant damage gives 10 * 0.5 = 5, including the rare tail rectangle.
    per_period_medians = {
        case: numpy.median(samples) for case, samples in sampled_damage.items()
    }
    integrated_medians = _integrate_ead_samples(
        {case: numpy.array([value]) for case, value in per_period_medians.items()},
        {"rp10": 10.0, "rp2": 2.0},
    )[0]
    assert integrated_medians == pytest.approx(5.0)
    assert numpy.quantile(ead_samples, 0.5) != pytest.approx(integrated_medians)


@pytest.mark.parametrize(
    "bad_input, message",
    [
        ("missing_return_periods", "must be supplied together"),
        ("missing_ead_quantiles", "must be supplied together"),
        ("mismatched_labels", "keys must match intensity_columns"),
        ("nonmapping_periods", "must map intensity-case labels"),
        ("duplicate_periods", "finite, unique, and at least one year"),
        ("period_below_one", "finite, unique, and at least one year"),
        ("nonfinite_period", "finite, unique, and at least one year"),
        ("nonnumeric_period", "return periods must be finite numbers"),
        ("no_requested_summaries", "request quantiles, ead_quantiles, or both"),
    ],
)
def test_direct_damages_validates_ead_inputs(bad_input, message):
    splits = pandas.DataFrame(
        {"asset": ["a"], "length": [1.0], "rp10": [1.0], "rp2": [2.0]}
    )
    arguments = {
        "asset_id": "asset",
        "quantity": "length",
        "intensity_columns": {"rp10": "rp10", "rp2": "rp2"},
        "curves": {"central": _constant_curve(0.5)},
        "curve_weights": {"central": 1.0},
        "cost_mean": 10.0,
        "cost_asset_cv": 0.0,
        "quantiles": (0.5,),
        "ead_quantiles": (0.5,),
        "return_periods": {"rp10": 10.0, "rp2": 2.0},
    }

    match bad_input:
        case "missing_return_periods":
            arguments["return_periods"] = None
        case "missing_ead_quantiles":
            arguments["ead_quantiles"] = None
        case "mismatched_labels":
            arguments["return_periods"] = {"other": 10.0}
        case "nonmapping_periods":
            arguments["return_periods"] = [("rp10", 10.0), ("rp2", 2.0)]
        case "duplicate_periods":
            arguments["return_periods"] = {"rp10": 10.0, "rp2": 10.0}
        case "period_below_one":
            arguments["return_periods"] = {"rp10": 0.5, "rp2": 2.0}
        case "nonfinite_period":
            arguments["return_periods"] = {"rp10": numpy.inf, "rp2": 2.0}
        case "nonnumeric_period":
            arguments["return_periods"] = {"rp10": "bad", "rp2": 2.0}
        case "no_requested_summaries":
            arguments["quantiles"] = None
            arguments["ead_quantiles"] = None
            arguments["return_periods"] = None
        case _:
            pytest.fail(f"Unhandled bad input: {bad_input}")

    with pytest.raises(ValueError, match=message):
        calculate_direct_damages(splits, **arguments)


def test_direct_damages_share_curve_draws_across_splits():
    splits = pandas.DataFrame(
        {
            "asset": ["a", "a"],
            "length": [1.0, 2.0],
            "hazard": [1.0, 1.0],
        }
    )
    result = calculate_direct_damages(
        splits,
        asset_id="asset",
        quantity="length",
        intensity_columns={10: "hazard"},
        curves={"low": _constant_curve(0.2), "high": _constant_curve(0.8)},
        curve_weights={"low": 0.6, "high": 0.4},
        cost_mean=10.0,
        cost_asset_cv=0.0,
        n_samples=1000,
        quantiles=(0.05, 0.25, 0.5, 0.75, 0.95),
        seed=9,
    )

    summary = result.quantiles.iloc[0]
    # q_05: the 60% low-curve draws total 0.2 * (1 + 2) * 10 = 6.
    assert summary["q_05"] == pytest.approx(6.0)
    # q_50: low-curve draws have 60% weight, so the median is 6.
    assert summary["q_50"] == pytest.approx(6.0)
    # q_95: the high-curve draws total 0.8 * (1 + 2) * 10 = 24.
    assert summary["q_95"] == pytest.approx(24.0)


@pytest.mark.parametrize(
    "uncertainty", ["hazard_alpha", "eta_sd", "cost_asset_cv", "cost_systemic_cv"]
)
def test_direct_damages_share_hazard_damage_and_cost_draws_across_splits(uncertainty):
    split_inputs = pandas.DataFrame(
        {"asset": ["a", "a"], "length": [1.0, 2.0], "hazard": [1.0, 1.0]}
    )
    combined_input = pandas.DataFrame(
        {"asset": ["a"], "length": [3.0], "hazard": [1.0]}
    )
    linear_curve = PiecewiseLinearDamageCurve(
        pandas.DataFrame({"intensity": [0.0, 10.0], "damage": [0.0, 1.0]})
    )
    kwargs = {
        "asset_id": "asset",
        "quantity": "length",
        "intensity_columns": {10: "hazard"},
        "curves": {"linear": linear_curve},
        "curve_weights": {"linear": 1.0},
        "cost_mean": 10.0,
        "cost_asset_cv": 0.0,
        "n_samples": 7,
        "quantiles": (0.05, 0.5, 0.95),
        "seed": 17,
    }
    kwargs[uncertainty] = 0.2

    split_result = calculate_direct_damages(split_inputs, **kwargs)
    combined_result = calculate_direct_damages(combined_input, **kwargs)
    # Shared per-asset draws make lengths 1 + 2 equivalent to one length-3 split.
    pandas.testing.assert_frame_equal(
        split_result.quantiles[["q_05", "q_50", "q_95"]],
        combined_result.quantiles[["q_05", "q_50", "q_95"]],
    )


def test_direct_damages_share_curve_draws_across_assets():
    splits = pandas.DataFrame(
        {"asset": ["a", "b"], "length": [1.0, 2.0], "hazard": [1.0, 1.0]}
    )
    result = calculate_direct_damages(
        splits,
        asset_id="asset",
        quantity="length",
        intensity_columns={10: "hazard"},
        curves={"low": _constant_curve(0.2), "high": _constant_curve(0.8)},
        curve_weights={"low": 0.5, "high": 0.5},
        cost_mean=10.0,
        cost_asset_cv=0.0,
        n_samples=1,  # N.B. single sample
        quantiles=(0.5,),
        seed=0,
    )

    summary = result.quantiles.set_index("asset")
    # Both assets share the high curve: 0.8 * 1 km * 10 = 8 for a.
    assert summary.loc["a", "q_50"] == pytest.approx(8.0)
    # Asset b has twice the length: 0.8 * 2 km * 10 = 16.
    assert summary.loc["b", "q_50"] == pytest.approx(16.0)


@pytest.mark.parametrize("uncertainty", ["hazard_alpha", "eta_sd", "cost_asset_cv"])
@pytest.mark.parametrize("scenario", [None, "scenario"])
def test_direct_damages_draw_local_uncertainty_independently(uncertainty, scenario):
    splits = pandas.DataFrame(
        {
            "asset": ["a", "b"] if scenario is None else ["a", "a"],
            "scenario": ["base", "future"],
            "length": [1.0, 1.0],
            "hazard": [1.0, 1.0],
        }
    )
    curve = PiecewiseLinearDamageCurve(
        pandas.DataFrame({"intensity": [0.0, 10.0], "damage": [0.0, 1.0]})
    )
    arguments = {
        "asset_id": "asset",
        "scenario": scenario,
        "quantity": "length",
        "intensity_columns": {10: "hazard"},
        "curves": {"linear": curve},
        "curve_weights": {"linear": 1.0},
        "cost_mean": 10.0,
        "cost_asset_cv": 0.0,
        "n_samples": 1,
        "quantiles": (0.5,),
        "seed": 17,
    }
    arguments[uncertainty] = 0.2
    result = calculate_direct_damages(splits, **arguments)
    # Identical inputs differ solely because the groups receive local draws.
    first, second = result.quantiles["q_50"]
    assert first != pytest.approx(second, rel=1e-12)


def test_direct_damages_share_draws_across_return_periods():
    splits = pandas.DataFrame(
        {
            "asset": ["a"],
            "length": [1.0],
            "rp_10": [1.0],
            "rp_100": [2.0],
        }
    )
    linear_curve = PiecewiseLinearDamageCurve(
        pandas.DataFrame({"intensity": [0.0, 10.0], "damage": [0.0, 1.0]})
    )
    result = calculate_direct_damages(
        splits,
        asset_id="asset",
        quantity="length",
        intensity_columns={10: "rp_10", 100: "rp_100"},
        curves={"linear": linear_curve},
        curve_weights={"linear": 1.0},
        cost_mean=10.0,
        cost_asset_cv=0.2,
        cost_systemic_cv=0.2,
        hazard_alpha=0.1,
        eta_sd=0.1,
        n_samples=10,  # Low samples
        quantiles=(0.05, 0.25, 0.5, 0.75, 0.95),
        seed=9,
    )

    summary = result.quantiles.set_index("intensity_case")
    for quantile in ("q_05", "q_50", "q_95"):
        # RP100 exactly doubles RP10's hazard and damage quantile, with few
        # samples and very tight relative tolerance.
        assert summary.loc[100, quantile] == pytest.approx(
            2.0 * summary.loc[10, quantile], rel=1e-12
        )


def test_direct_damages_totals_keep_scenarios_separate():
    splits = pandas.DataFrame(
        {
            "asset": ["a", "b", "a", "b"],
            "scenario": ["base", "base", "future", "future"],
            "quantity": [1.0] * 4,
            "hazard": [1.0, 2.0, 3.0, 4.0],
        }
    )
    curve = PiecewiseLinearDamageCurve(
        pandas.DataFrame({"intensity": [0.0, 10.0], "damage": [0.0, 1.0]})
    )
    result = calculate_direct_damages(
        splits,
        asset_id="asset",
        scenario="scenario",
        quantity="quantity",
        intensity_columns={10: "hazard"},
        curves={"linear": curve},
        curve_weights={"linear": 1.0},
        cost_mean=10.0,
        n_samples=1,
        quantiles=(0.5,),
        return_periods={10: 10.0},
        ead_quantiles=(0.5,),
        aggregate_assets=True,
    )

    # Damage = hazard/10 * quantity * cost, so each asset's damage equals its hazard.
    summary = result.total_quantiles.set_index("scenario")
    assert summary.loc["base", "q_50"] == pytest.approx(3.0)  # 1 + 2
    assert summary.loc["future", "q_50"] == pytest.approx(7.0)  # 3 + 4
    assert len(summary) == 2
    assert (summary["n_assets"] == 2).all()
    assert (summary["n_splits"] == 2).all()

    # With one RP10 event, EAD is just the tail rectangle: total damage * 1 / 10
    ead = result.total_ead_quantiles.set_index("scenario")
    assert ead.loc["base", "q_50"] == pytest.approx(0.3)
    assert ead.loc["future", "q_50"] == pytest.approx(0.7)
    assert len(ead) == 2
    assert (ead["n_assets"] == 2).all()
    assert (ead["n_splits"] == 2).all()


def test_direct_damages_scenario_groups_and_reproducible_quantiles():
    splits = pandas.DataFrame(
        {
            "asset": ["a", "a"],
            "scenario": ["base", "future"],
            "length": [1.0, 1.0],
            "hazard": [1.0, 2.0],
        }
    )
    curve = PiecewiseLinearDamageCurve(
        pandas.DataFrame({"intensity": [0.0, 10.0], "damage": [0.0, 1.0]})
    )
    kwargs = {
        "asset_id": "asset",
        "scenario": "scenario",
        "quantity": "length",
        "intensity_columns": {10: "hazard"},
        "curves": {"central": curve},
        "curve_weights": {"central": 1.0},
        "cost_mean": 10.0,
        "cost_asset_cv": 0.0,
        "n_samples": 100,
        "quantiles": (0.05, 0.25, 0.5, 0.75, 0.95),
        "seed": 12,
    }

    first = calculate_direct_damages(splits, **kwargs)
    second = calculate_direct_damages(splits, **kwargs)
    # The same seed repeats the same sampled quantiles.
    pandas.testing.assert_frame_equal(first.quantiles, second.quantiles)
    summary = first.quantiles.set_index("scenario")
    assert set(summary.index) == {"base", "future"}
    # Base median: 0.1 * 1 km * 10 = 1.
    assert summary.loc["base", "q_50"] == pytest.approx(1.0)
    # Future median: 0.2 * 1 km * 10 = 2.
    assert summary.loc["future", "q_50"] == pytest.approx(2.0)


def test_direct_damages_share_curve_draws_across_asset_scenario_groups():
    splits = pandas.DataFrame(
        {
            "asset": ["a", "a", "a", "a"],
            "scenario": ["base", "base", "future", "future"],
            "length": [1.0, 2.0, 1.0, 2.0],
            "hazard": [1.0, 1.0, 1.0, 1.0],
        }
    )
    result = calculate_direct_damages(
        splits,
        asset_id="asset",
        scenario="scenario",
        quantity="length",
        intensity_columns={10: "hazard"},
        curves={"low": _constant_curve(0.2), "high": _constant_curve(0.8)},
        curve_weights={"low": 0.5, "high": 0.5},
        cost_mean=10.0,
        cost_asset_cv=0.0,
        n_samples=1,  # N.B. Single sample
        quantiles=(0.5,),
        seed=0,
    )

    summary = result.quantiles.set_index("scenario")
    # Both scenarios share the high curve: 0.8 * (1 + 2) * 10 = 24.
    assert summary.loc["base", "q_50"] == pytest.approx(24.0)
    # The same systemic curve applies to future's splits: 0.8 * (1 + 2) * 10 = 24.
    assert summary.loc["future", "q_50"] == pytest.approx(24.0)


@pytest.mark.parametrize("cost_mean", [100.0, "cost"])
def test_direct_damages_share_systemic_cost_multipliers(cost_mean):
    splits = pandas.DataFrame(
        {
            "asset": ["a", "a", "b", "c"],
            "scenario": ["base", "future", "base", "future"],
            "length": [1.0, 2.0, 3.0, 1.0],
            "hazard": [1.0, 1.0, 1.0, 1.0],
            "cost": [100.0, 100.0, 150.0, 200.0],
        }
    )
    arguments = {
        "asset_id": "asset",
        "scenario": "scenario",
        "quantity": "length",
        "intensity_columns": {10: "hazard"},
        "curves": {"central": _constant_curve(0.5)},
        "curve_weights": {"central": 1.0},
        "cost_mean": cost_mean,
        "cost_asset_cv": 0.0,
        "n_samples": 9,
        "quantiles": (0.0, 0.5, 1.0),
        "seed": 42,
    }
    result = calculate_direct_damages(splits, cost_systemic_cv=0.05, **arguments)
    doubled = calculate_direct_damages(splits, cost_systemic_cv=0.10, **arguments)
    columns = ["q_00", "q_50", "q_100"]
    baselines = splits["cost"].to_numpy() if cost_mean == "cost" else cost_mean
    # Recover unit cost, then divide by the baseline to get relative changes.
    costs = result.quantiles[columns].to_numpy() / (
        0.5 * splits["length"].to_numpy()[:, None]
    )
    relative_changes = (costs.T / baselines).T - 1.0
    doubled_costs = doubled.quantiles[columns].to_numpy() / (
        0.5 * splits["length"].to_numpy()[:, None]
    )
    doubled_changes = (doubled_costs.T / baselines).T - 1.0
    for change in relative_changes:
        assert_allclose(change, relative_changes[0], rtol=1e-12)
    # Doubling the systemic CV doubles each relative change when neither draw is clamped.
    assert_allclose(doubled_changes, 2 * relative_changes, rtol=1e-12)
    assert (costs > 0).all()
    assert (doubled_costs > 0).all()
    assert relative_changes[0, 0] < relative_changes[0, 1] < relative_changes[0, 2]
    assert result.metadata["cost_systemic_cv"] == 0.05
    assert result.metadata["quantity_column"] == "length"
    assert result.metadata["draw_scopes"] == {
        "damage_curve": "call",
        "cost_system": "call",
        "hazard": "asset_scenario",
        "damage_noise": "asset_scenario",
        "cost": "asset_scenario",
    }


@pytest.mark.parametrize("systemic_cv", [0.0, 0.25])
def test_direct_damages_mixed_asset_quantities(systemic_cv):
    # Precomputed count, line lengths and polygon areas; one geometry type per asset.
    splits = pandas.DataFrame(
        {
            "asset": ["point", "line", "line", "polygon", "polygon"],
            "quantity": [1.0, 2.0, 3.0, 10.0, 20.0],
            "unit_cost": [100.0, 10.0, 10.0, 4.0, 4.0],
            "hazard": [1.0] * 5,
        }
    )
    result = calculate_direct_damages(
        splits,
        asset_id="asset",
        quantity="quantity",
        intensity_columns={10: "hazard"},
        curves={"central": _constant_curve(0.5)},
        curve_weights={"central": 1.0},
        cost_mean="unit_cost",
        cost_systemic_cv=systemic_cv,
        quantiles=(0.0, 0.5, 1.0),
        n_samples=100,
        seed=42,
    )
    summary = result.quantiles.set_index("asset")
    columns = ["q_00", "q_50", "q_100"]
    # Baseline damages: half of 1*100, (2+3)*10 and (10+20)*4.
    if systemic_cv == 0.0:
        for asset, expected in [("point", 50.0), ("line", 25.0), ("polygon", 60.0)]:
            assert_allclose(summary.loc[asset, columns].to_numpy(dtype=float), expected)
    else:
        multipliers = summary.loc["point", columns].to_numpy(dtype=float) / 50.0
        assert multipliers[0] < multipliers[1] < multipliers[2]
        for asset, baseline in [("line", 25.0), ("polygon", 60.0)]:
            assert_allclose(
                summary.loc[asset, columns].to_numpy(dtype=float) / baseline,
                multipliers,
            )
    assert summary["n_splits"].to_dict() == {"point": 1, "line": 2, "polygon": 2}
    assert result.metadata["quantity_column"] == "quantity"


def test_direct_damages_clamp_both_cost_stages(monkeypatch):
    rng = Mock()
    rng.choice.return_value = numpy.zeros(4, dtype=int)
    rng.standard_normal.side_effect = [
        numpy.array([-2.0, -2.0, 1.0, 1.0]),  # Systemic cost draws.
        numpy.zeros(4),  # Hazard draws.
        numpy.zeros(4),  # Damage draws.
        numpy.array([0.0, -2.0, -2.0, 1.0]),  # Local cost draws.
    ]
    monkeypatch.setattr("snail.damages.numpy.random.default_rng", lambda seed: rng)
    result = calculate_direct_damages(
        pandas.DataFrame({"asset": ["a"], "length": [2.0], "hazard": [1.0]}),
        asset_id="asset",
        quantity="length",
        intensity_columns={10: "hazard"},
        curves={"central": _constant_curve(0.5)},
        curve_weights={"central": 1.0},
        cost_mean=1.0,
        cost_systemic_cv=1.0,
        cost_asset_cv=1.0,
        n_samples=4,
        quantiles=(0.0, 0.5, 1.0),
        seed=0,
    )
    # Systemic costs [-1, -1, 2, 2] become [0, 0, 2, 2] before local multipliers.
    # Multipliers [1, -1, -1, 2] and the second clamp give costs [0, 0, 0, 4].
    # Damage = 0.5 * 2 * cost, so minimum, median and maximum are 0, 0 and 4.
    assert_allclose(
        result.quantiles.loc[0, ["q_00", "q_50", "q_100"]].to_numpy(dtype=float),
        [0.0, 0.0, 4.0],
    )


@pytest.mark.parametrize(
    "bad_input, error, message",
    [
        ("missing_asset_column", KeyError, "Required columns"),
        ("missing_length_column", KeyError, "Required columns"),
        ("missing_hazard_column", KeyError, "Required columns"),
        ("missing_scenario_column", KeyError, "Required columns"),
        ("missing_cost_column", KeyError, "Required columns"),
        ("empty_intensity_columns", ValueError, "at least one intensity case"),
        ("zero_samples", ValueError, "n_samples"),
        ("noninteger_samples", ValueError, "n_samples"),
        ("empty_quantiles", ValueError, "quantiles must be a non-empty sequence"),
        ("out_of_range_quantile", ValueError, "quantiles must be a non-empty sequence"),
        ("nonfinite_quantile", ValueError, "quantiles must be a non-empty sequence"),
        ("duplicate_quantiles", ValueError, "quantiles must not contain duplicates"),
        ("duplicate_quantile_columns", ValueError, "distinct percentile column names"),
        ("negative_eta_sd", ValueError, "non-negative"),
        ("nonfinite_eta_sd", ValueError, "non-negative"),
        ("negative_cost_asset_cv", ValueError, "non-negative"),
        ("nonfinite_cost_asset_cv", ValueError, "non-negative"),
        (
            "negative_cost_systemic_cv",
            ValueError,
            "cost_systemic_cv must be finite and non-negative",
        ),
        (
            "nan_cost_systemic_cv",
            ValueError,
            "cost_systemic_cv must be finite and non-negative",
        ),
        (
            "infinite_cost_systemic_cv",
            ValueError,
            "cost_systemic_cv must be finite and non-negative",
        ),
        (
            "negative_infinite_cost_systemic_cv",
            ValueError,
            "cost_systemic_cv must be finite and non-negative",
        ),
        ("missing_hazard_alpha", ValueError, "hazard_alpha missing intensity cases"),
        (
            "negative_hazard_alpha",
            ValueError,
            "hazard_alpha values must be non-negative",
        ),
        (
            "nonfinite_hazard_alpha",
            ValueError,
            "hazard_alpha values must be non-negative",
        ),
        (
            "negative_scalar_hazard_alpha",
            ValueError,
            "hazard_alpha values must be non-negative",
        ),
        ("empty_curves", ValueError, "same non-empty keys"),
        ("mismatched_curve_keys", ValueError, "same non-empty keys"),
        ("negative_curve_weight", ValueError, "curve_weights must be finite"),
        ("nonfinite_curve_weight", ValueError, "curve_weights must be finite"),
        ("zero_curve_weights", ValueError, "curve_weights must be finite"),
        ("negative_cost_mean", ValueError, "cost_mean must be finite and non-negative"),
        (
            "nonfinite_cost_mean",
            ValueError,
            "cost_mean must be finite and non-negative",
        ),
        (
            "nonnumeric_cost_column",
            ValueError,
            "Column 'cost' must contain finite numeric values",
        ),
        (
            "nonfinite_cost_column",
            ValueError,
            "Column 'cost' must contain finite numeric values",
        ),
        ("negative_cost_column", ValueError, "Column 'cost' must be non-negative"),
        (
            "nonnumeric_length",
            ValueError,
            "Column 'length' must contain finite numeric values",
        ),
        (
            "nonfinite_length",
            ValueError,
            "Column 'length' must contain finite numeric values",
        ),
        ("negative_length", ValueError, "Column 'length' must be non-negative"),
        (
            "nonnumeric_hazard",
            ValueError,
            "Column 'hazard' must contain finite numeric values",
        ),
        (
            "nonfinite_hazard",
            ValueError,
            "Column 'hazard' must contain finite numeric values",
        ),
        ("negative_hazard", ValueError, "Column 'hazard' must be non-negative"),
        ("missing_asset_id", ValueError, "identifiers must not be missing"),
        ("missing_scenario_id", ValueError, "identifiers must not be missing"),
        ("inconsistent_asset_cost", ValueError, "constant within group"),
    ],
)
def test_direct_damages_validates_inputs(bad_input, error, message):
    splits = pandas.DataFrame(
        {
            "asset": ["a"],
            "scenario": ["base"],
            "length": [1.0],
            "hazard": [1.0],
            "cost": [10.0],
        }
    )
    arguments = {
        "asset_id": "asset",
        "scenario": None,
        "quantity": "length",
        "intensity_columns": {10: "hazard"},
        "curves": {"central": _constant_curve(0.5)},
        "curve_weights": {"central": 1.0},
        "cost_mean": 10.0,
        "cost_asset_cv": 0.0,
        "n_samples": 10,
        "quantiles": (0.05, 0.25, 0.5, 0.75, 0.95),
    }

    match bad_input:
        case "missing_asset_column":
            splits = splits.drop(columns="asset")
        case "missing_length_column":
            splits = splits.drop(columns="length")
        case "missing_hazard_column":
            arguments["intensity_columns"] = {10: "missing_hazard"}
        case "missing_scenario_column":
            arguments["scenario"] = "missing_scenario"
        case "missing_cost_column":
            arguments["cost_mean"] = "missing_cost"
        case "empty_intensity_columns":
            arguments["intensity_columns"] = {}
        case "zero_samples":
            arguments["n_samples"] = 0
        case "noninteger_samples":
            arguments["n_samples"] = 1.5
        case "empty_quantiles":
            arguments["quantiles"] = ()
        case "out_of_range_quantile":
            arguments["quantiles"] = (1.1,)
        case "nonfinite_quantile":
            arguments["quantiles"] = (numpy.nan,)
        case "duplicate_quantiles":
            arguments["quantiles"] = (0.5, 0.5)
        case "duplicate_quantile_columns":
            arguments["quantiles"] = (0.501, 0.504)
        case "negative_eta_sd":
            arguments["eta_sd"] = -0.1
        case "nonfinite_eta_sd":
            arguments["eta_sd"] = numpy.inf
        case "negative_cost_asset_cv":
            arguments["cost_asset_cv"] = -0.1
        case "nonfinite_cost_asset_cv":
            arguments["cost_asset_cv"] = numpy.nan
        case "negative_cost_systemic_cv":
            arguments["cost_systemic_cv"] = -0.1
        case "nan_cost_systemic_cv":
            arguments["cost_systemic_cv"] = numpy.nan
        case "infinite_cost_systemic_cv":
            arguments["cost_systemic_cv"] = numpy.inf
        case "negative_infinite_cost_systemic_cv":
            arguments["cost_systemic_cv"] = -numpy.inf
        case "missing_hazard_alpha":
            arguments["hazard_alpha"] = {20: 0.1}
        case "negative_hazard_alpha":
            arguments["hazard_alpha"] = {10: -0.1}
        case "nonfinite_hazard_alpha":
            arguments["hazard_alpha"] = {10: numpy.inf}
        case "negative_scalar_hazard_alpha":
            arguments["hazard_alpha"] = -0.1
        case "empty_curves":
            arguments["curves"] = arguments["curve_weights"] = {}
        case "mismatched_curve_keys":
            arguments["curve_weights"] = {"other": 1.0}
        case "negative_curve_weight":
            arguments["curve_weights"] = {"central": -1.0}
        case "nonfinite_curve_weight":
            arguments["curve_weights"] = {"central": numpy.nan}
        case "zero_curve_weights":
            arguments["curve_weights"] = {"central": 0.0}
        case "negative_cost_mean":
            arguments["cost_mean"] = -1.0
        case "nonfinite_cost_mean":
            arguments["cost_mean"] = numpy.inf
        case "nonnumeric_cost_column":
            arguments["cost_mean"] = "cost"
            splits["cost"] = ["bad"]
        case "nonfinite_cost_column":
            arguments["cost_mean"] = "cost"
            splits["cost"] = [numpy.inf]
        case "negative_cost_column":
            arguments["cost_mean"] = "cost"
            splits["cost"] = [-1.0]
        case "nonnumeric_length":
            splits["length"] = ["bad"]
        case "nonfinite_length":
            splits.loc[0, "length"] = numpy.inf
        case "negative_length":
            splits.loc[0, "length"] = -1.0
        case "nonnumeric_hazard":
            splits["hazard"] = ["bad"]
        case "nonfinite_hazard":
            splits.loc[0, "hazard"] = numpy.nan
        case "negative_hazard":
            splits.loc[0, "hazard"] = -1.0
        case "missing_asset_id":
            splits.loc[0, "asset"] = None
        case "missing_scenario_id":
            arguments["scenario"] = "scenario"
            splits.loc[0, "scenario"] = None
        case "inconsistent_asset_cost":
            splits = pandas.concat([splits, splits], ignore_index=True)
            splits.loc[1, "cost"] = 20.0
            arguments["cost_mean"] = "cost"
        case _:
            pytest.fail(f"Unhandled bad input: {bad_input}")

    with pytest.raises(error, match=message):
        calculate_direct_damages(splits, **arguments)
