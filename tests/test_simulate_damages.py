"""Test array-based Monte Carlo direct damages"""

from functools import partial

import numpy
import pandas
import pytest
from numpy.testing import assert_allclose, assert_array_equal
from scipy.integrate import trapezoid

from snail import damages
from snail.damages import (
    DRAW_SCHEME,
    DirectDamageArrays,
    PiecewiseLinearDamageCurve,
    _integrate_ead_samples,
    _uniform_draws,
    calculate_direct_damages,
    expected_annual_damage,
    simulate_direct_damages,
    summarise_samples,
)

FULL_UNCERTAINTY = {
    "hazard_cv": 0.2,
    "eta_sd": 0.3,
    "cost_asset_cv": 0.3,
    "cost_systemic_cv": 0.2,
}


def _curve(intensity, damage):
    return PiecewiseLinearDamageCurve(
        pandas.DataFrame({"intensity": intensity, "damage": damage})
    )


@pytest.fixture
def curves():
    return [
        _curve([0.0, 1.0, 2.0], [0.0, 0.3, 1.0]),
        _curve([0.0, 1.0, 2.0], [0.0, 0.5, 0.8]),
    ]


@pytest.fixture
def inputs():
    """Twelve splits of seven assets; asset 3 is dry, asset 6 has no splits."""
    rng = numpy.random.default_rng(1)
    asset_index = numpy.array([0, 0, 1, 2, 2, 2, 3, 3, 4, 4, 5, 5])
    intensity = rng.random((len(asset_index), 3)) * 2
    intensity[asset_index == 3] = 0.0
    intensity[1] = 0.0  # a dry split of a wet asset
    intensity[4, 0] = 0.0
    quantity = rng.random(len(asset_index)) + 0.5
    return intensity, asset_index, quantity


def _run(inputs, curves, **kwargs):
    intensity, asset_index, quantity = inputs
    options = {
        "curves": curves,
        "curve_weights": [1.0, 2.0],
        "cost_mean": 10.0,
        "n_samples": 500,
        "return_periods": [10.0, 100.0, 1000.0],
        "n_assets": 7,
        "seed": 42,
    }
    options.update(kwargs)
    return simulate_direct_damages(intensity, asset_index, quantity, **options)


def _capture_blocks(monkeypatch):
    blocks = []
    monkeypatch.setattr(
        damages,
        "_run_simulation",
        partial(damages._run_simulation, on_block=blocks.append),
    )
    return blocks


def test_returns_arrays_with_documented_shapes(inputs, curves):
    result = _run(inputs, curves, keep_total_samples=True, **FULL_UNCERTAINTY)
    assert isinstance(result, DirectDamageArrays)
    assert result.split_fraction.shape == (12, 3, 3)
    assert result.split_cost.shape == (12, 3, 3)
    assert result.split_ead.shape == (12, 3)
    assert result.asset_cost.shape == (7, 3, 3)
    assert result.asset_ead.shape == (7, 3)
    assert result.total_cost.shape == (3, 3)
    assert result.total_ead.shape == (3,)
    assert result.total_samples.shape == (500, 3)
    assert_array_equal(result.quantiles, [0.05, 0.5, 0.95])
    assert_array_equal(result.return_periods, [10.0, 100.0, 1000.0])
    assert result.metadata["draw_scheme"] == DRAW_SCHEME
    assert result.metadata["curve_weights"] == pytest.approx([1 / 3, 2 / 3])
    assert result.metadata["seed"] == 42
    assert result.metadata["systemic_key"] is None
    # quantiles are ordered
    assert (numpy.diff(result.asset_cost, axis=-1) >= 0).all()


def test_split_samples_sum_to_asset_and_total_samples(monkeypatch, inputs, curves):
    blocks = _capture_blocks(monkeypatch)
    result = _run(
        inputs, curves, keep_total_samples=True, block_size=3, **FULL_UNCERTAINTY
    )
    assert len(blocks) > 1
    asset_index = inputs[1]
    total = numpy.zeros((3, 500))
    for block in blocks:
        assets = asset_index[block["split_index"]]
        for position, asset in enumerate(block["asset_index"]):
            assert_allclose(
                block["split_cost"][assets == asset].sum(axis=0),
                block["asset_cost"][position],
                rtol=1e-12,
            )
        total += block["asset_cost"].sum(axis=0)
        order = block["sample_order"]
    total_samples = numpy.empty((500, 3))
    total_samples[order] = total.T
    assert_allclose(result.total_samples, total_samples, rtol=1e-12)
    # and the summaries come from those samples
    assert_allclose(
        result.total_cost, summarise_samples(result.total_samples.T, (0.05, 0.5, 0.95))
    )


def test_summaries_equal_numpy_quantile_of_samples(monkeypatch, inputs, curves):
    blocks = _capture_blocks(monkeypatch)
    quantiles = (0.0, 0.01, 0.3, 0.5, 0.77, 1.0)
    for options in ({}, FULL_UNCERTAINTY):
        blocks.clear()
        result = _run(inputs, curves, quantiles=quantiles, n_samples=101, **options)
        for block in blocks:
            splits = block["split_index"]
            for name, actual in [
                ("fraction", result.split_fraction[splits]),
                ("split_cost", result.split_cost[splits]),
                ("asset_cost", result.asset_cost[block["asset_index"]]),
            ]:
                expected = numpy.moveaxis(
                    numpy.quantile(block[name], quantiles, axis=-1), 0, -1
                )
                assert_allclose(actual, expected, rtol=1e-12, atol=0)
            for name, actual in [
                ("split_cost", result.split_ead[splits]),
                ("asset_cost", result.asset_ead[block["asset_index"]]),
            ]:
                ead = expected_annual_damage(
                    block[name].transpose(0, 2, 1), result.return_periods
                )
                expected = numpy.moveaxis(
                    numpy.quantile(ead, quantiles, axis=-1), 0, -1
                )
                assert_allclose(actual, expected, rtol=1e-12, atol=0)


@pytest.mark.parametrize("n", [1, 2, 7, 1000])
def test_summarise_samples_matches_numpy_quantile(n):
    samples = numpy.random.default_rng(n).lognormal(size=(4, 3, n))
    quantiles = [0.0, 0.05, 0.123, 0.5, 0.95, 1.0]
    expected = numpy.moveaxis(numpy.quantile(samples, quantiles, axis=-1), 0, -1)
    assert_allclose(summarise_samples(samples, quantiles), expected, rtol=1e-12)


def test_summarise_samples_propagates_nan():
    samples = numpy.array([[1.0, numpy.nan, 2.0], [1.0, 2.0, 3.0]])
    actual = summarise_samples(samples, [0.5])
    assert numpy.isnan(actual[0, 0])
    assert actual[1, 0] == 2.0


def test_expected_annual_damage_matches_integrate_ead_samples():
    samples = numpy.random.default_rng(3).random((50, 4))
    return_periods = [100.0, 2.0, 1000.0, 10.0]
    expected = _integrate_ead_samples(
        {case: samples[:, case] for case in range(4)},
        dict(enumerate(return_periods)),
    )
    assert_array_equal(expected_annual_damage(samples, return_periods), expected)


def test_draws_depend_only_on_seed_stream_and_key():
    keys = numpy.array([5, 0, 2**63 + 7, 3], dtype=numpy.uint64)
    all_draws = _uniform_draws(11, "cost", keys, 0, 1000)
    assert ((all_draws > 0) & (all_draws < 1)).all()
    subset = _uniform_draws(11, "cost", keys[[3, 0]], 0, 1000)
    assert_array_equal(subset, all_draws[[3, 0]])
    prefix = _uniform_draws(11, "cost", keys, 0, 10)
    assert_array_equal(prefix, all_draws[:, :10])
    assert not (_uniform_draws(12, "cost", keys, 0, 10) == prefix).any()
    assert not (_uniform_draws(11, "hazard", keys, 0, 10) == prefix).any()


def test_assets_get_identical_results_in_subsets_and_any_order(inputs, curves):
    intensity, asset_index, quantity = inputs
    keys = numpy.array([-3, 10, 2**40, -(2**62), 7, 99, 1], dtype=numpy.int64)
    full = _run(inputs, curves, asset_keys=keys, **FULL_UNCERTAINTY)

    # Assets 5, 0 and 2 only, in that order, splits shuffled within the call
    chosen = numpy.array([5, 0, 2])
    new_index = {old: new for new, old in enumerate(chosen)}
    rows = numpy.concatenate([numpy.flatnonzero(asset_index == a) for a in chosen])
    rows = rows[[5, 0, 3, 1, 4, 2, 6]]
    part = simulate_direct_damages(
        intensity[rows],
        numpy.array([new_index[a] for a in asset_index[rows]]),
        quantity[rows],
        curves=curves,
        curve_weights=[1.0, 2.0],
        cost_mean=10.0,
        n_samples=500,
        return_periods=[10.0, 100.0, 1000.0],
        seed=42,
        asset_keys=keys[chosen],
        block_size=1,
        **FULL_UNCERTAINTY,
    )
    for name in ("split_fraction", "split_cost", "split_ead"):
        assert_array_equal(getattr(part, name), getattr(full, name)[rows])
    for name in ("asset_cost", "asset_ead"):
        # asset 2 has three splits, summed in a different order
        assert_array_equal(getattr(part, name)[:2], getattr(full, name)[[5, 0]])
        assert_allclose(getattr(part, name)[2], getattr(full, name)[2], rtol=1e-12)


def test_unsigned_and_signed_keys_with_the_same_bits_agree(inputs, curves):
    signed = numpy.array([-1, 0, 1, 2, 3, 4, -(2**63)], dtype=numpy.int64)
    unsigned = signed.view(numpy.uint64)
    a = _run(inputs, curves, asset_keys=signed, cost_asset_cv=0.5)
    b = _run(inputs, curves, asset_keys=unsigned, cost_asset_cv=0.5)
    assert_array_equal(a.asset_cost, b.asset_cost)


@pytest.mark.parametrize("block_size", [1, 7, 100_000])
def test_results_do_not_depend_on_block_size(inputs, curves, block_size):
    options = dict(FULL_UNCERTAINTY, keep_total_samples=True)
    reference = _run(inputs, curves, block_size=4, **options)
    result = _run(inputs, curves, block_size=block_size, **options)
    for name in ("split_fraction", "split_cost", "split_ead", "asset_cost"):
        assert_array_equal(getattr(result, name), getattr(reference, name))
    assert_array_equal(result.asset_ead, reference.asset_ead)
    for name in ("total_cost", "total_ead", "total_samples"):
        assert_allclose(getattr(result, name), getattr(reference, name), rtol=1e-12)


def test_total_samples_add_up_across_calls(inputs, curves):
    intensity, asset_index, quantity = inputs
    options = dict(
        FULL_UNCERTAINTY,
        curves=curves,
        curve_weights=[1.0, 2.0],
        cost_mean=10.0,
        n_samples=500,
        seed=42,
        keep_total_samples=True,
        systemic_key=3,
    )
    whole = simulate_direct_damages(intensity, asset_index, quantity, **options)
    first = asset_index < 3
    parts = [
        simulate_direct_damages(
            intensity[rows],
            asset_index[rows],
            quantity[rows],
            asset_keys=numpy.arange(7),
            n_assets=7,
            **options,
        )
        for rows in (first, ~first)
    ]
    summed = parts[0].total_samples + parts[1].total_samples
    assert_allclose(summed, whole.total_samples, rtol=1e-12)
    assert_allclose(
        summarise_samples(summed.T, whole.quantiles), whole.total_cost, rtol=1e-12
    )


def test_systemic_key_makes_systemic_draws_independent(inputs, curves):
    options = {"cost_systemic_cv": 0.3, "keep_total_samples": True}
    unkeyed = _run(inputs, curves, **options)
    again = _run(inputs, curves, **options)
    keyed = _run(inputs, curves, systemic_key=1, **options)
    other = _run(inputs, curves, systemic_key=2, **options)
    assert_array_equal(unkeyed.total_samples, again.total_samples)
    assert keyed.metadata["systemic_key"] == 1
    for a, b in [(unkeyed, keyed), (keyed, other)]:
        assert not numpy.isclose(a.total_samples, b.total_samples).all(axis=1).any()
    # with no systemic uncertainty, the key does not affect local draws
    local = {"cost_asset_cv": 0.3, "hazard_cv": 0.2, "eta_sd": 0.2}
    one_curve = curves[:1]
    a = _run(inputs, one_curve, curve_weights=[1.0], systemic_key=1, **local)
    b = _run(inputs, one_curve, curve_weights=[1.0], systemic_key=2, **local)
    assert_array_equal(a.asset_cost, b.asset_cost)


def test_scenarios_use_common_random_numbers(curves):
    linear = [_curve([0.0, 10.0], [0.0, 1.0])]
    options = {
        "curves": linear,
        "curve_weights": [1.0],
        "cost_mean": 5.0,
        "cost_asset_cv": 0.4,
        "n_samples": 300,
        "quantiles": (0.1, 0.5, 0.9),
        "asset_keys": [17, 4],
    }
    baseline = numpy.array([[1.0, 2.0], [3.0, 1.5]])
    future = baseline * numpy.array([[2.0, 3.0], [1.5, 2.0]])
    a = simulate_direct_damages(baseline, [0, 1], [1.0, 1.0], **options)
    b = simulate_direct_damages(future, [0, 1], [1.0, 1.0], **options)
    # same cost draws, so each cost quantile scales with the damage fraction
    ratio = (future / baseline)[:, :, None]
    assert_allclose(b.asset_cost, a.asset_cost * ratio, rtol=1e-12)


def test_zero_uncertainty_gives_deterministic_values():
    curve = _curve([0.0, 2.0], [0.0, 1.0])
    intensity = numpy.array([[0.5, 1.0, 1.5], [1.0, 1.0, 3.0], [0.0, 0.2, 0.4]])
    quantity = numpy.array([2.0, 3.0, 4.0])
    return_periods = numpy.array([2.0, 10.0, 100.0])
    result = simulate_direct_damages(
        intensity,
        [0, 0, 1],
        quantity,
        curves=[curve],
        curve_weights=[1.0],
        cost_mean=[10.0, 20.0],
        n_samples=50,
        return_periods=return_periods,
        quantiles=(0.0, 0.5, 1.0),
    )
    fraction = numpy.clip(intensity / 2.0, 0, 1)
    split_cost = fraction * quantity[:, None] * numpy.array([10.0, 10.0, 20.0])[:, None]
    asset_cost = numpy.stack([split_cost[:2].sum(axis=0), split_cost[2]])
    assert_allclose(result.split_fraction, fraction[..., None].repeat(3, axis=-1))
    assert_allclose(result.split_cost, split_cost[..., None].repeat(3, axis=-1))
    assert_allclose(result.asset_cost, asset_cost[..., None].repeat(3, axis=-1))
    assert_allclose(result.total_cost[:, 0], asset_cost.sum(axis=0))

    # Independent EAD: damage linear in log(p) between events, constant to p = 0
    def hand_ead(damage):
        p = 1.0 / return_periods
        grid = numpy.geomspace(p.min(), p.max(), 200_001)
        values = numpy.interp(numpy.log(grid), numpy.log(p[::-1]), damage[::-1])
        return trapezoid(values, grid) + p.min() * damage[numpy.argmin(p)]

    for actual, damage in [
        (result.split_ead, split_cost),
        (result.asset_ead, asset_cost),
        (result.total_ead[None], asset_cost.sum(axis=0, keepdims=True)),
    ]:
        expected = numpy.array([hand_ead(row) for row in damage])
        assert_allclose(actual, expected[:, None].repeat(3, axis=-1), rtol=1e-8)


def test_per_event_hazard_cv(curves):
    result = simulate_direct_damages(
        [[1.0, 1.0]],
        [0],
        [1.0],
        curves=curves[:1],
        curve_weights=[1.0],
        cost_mean=1.0,
        hazard_cv=[0.0, 0.5],
        n_samples=200,
    )
    assert_allclose(result.split_fraction[0, 0], 0.3)
    assert result.split_fraction[0, 1, 0] < 0.3 < result.split_fraction[0, 1, 2]


def test_dry_assets_are_skipped_with_exact_zeros(inputs, curves):
    intensity, asset_index, quantity = inputs
    result = _run(inputs, curves, **FULL_UNCERTAINTY)
    assert result.metadata["skip_dry"]
    assert result.metadata["n_computed_assets"] == 5
    assert result.metadata["n_computed_splits"] == 9
    dry_assets = [3, 6]
    for name in ("asset_cost", "asset_ead"):
        assert (getattr(result, name)[dry_assets] == 0).all()
    dry_splits = (intensity == 0).all(axis=1)
    for name in ("split_fraction", "split_cost", "split_ead"):
        assert (getattr(result, name)[dry_splits] == 0).all()
    assert (result.split_cost[4, 0] == 0).all()

    # Without the dry asset, every other result is unchanged
    wet = asset_index != 3
    reduced = simulate_direct_damages(
        intensity[wet],
        asset_index[wet],
        quantity[wet],
        curves=curves,
        curve_weights=[1.0, 2.0],
        cost_mean=10.0,
        n_samples=500,
        return_periods=[10.0, 100.0, 1000.0],
        n_assets=7,
        seed=42,
        **FULL_UNCERTAINTY,
    )
    assert_array_equal(reduced.split_cost, result.split_cost[wet])
    assert_array_equal(reduced.asset_cost, result.asset_cost)


def test_curves_damaging_at_zero_intensity_disable_skipping(inputs, curves):
    offset = [curves[0], _curve([0.0, 2.0], [0.1, 0.5])]
    result = _run(inputs, offset)
    assert not result.metadata["skip_dry"]
    assert result.metadata["n_computed_splits"] == 12
    assert (result.asset_cost[3, :, -1] > 0).all()  # 95% chooses the offset curve
    assert (result.asset_cost[6] == 0).all()  # no splits


def test_levels_select_outputs(inputs, curves):
    result = _run(inputs, curves, levels=("asset",), keep_total_samples=True)
    assert result.asset_cost is not None
    assert result.split_fraction is None
    assert result.split_cost is None
    assert result.split_ead is None
    assert result.total_cost is None
    assert result.total_ead is None
    assert result.total_samples.shape == (500, 3)
    full = _run(inputs, curves)
    assert_array_equal(result.asset_cost, full.asset_cost)
    assert full.total_samples is None
    no_ead = _run(inputs, curves, return_periods=None)
    assert no_ead.split_ead is None and no_ead.total_ead is None
    assert no_ead.asset_cost is not None


def test_first_samples_do_not_depend_on_n_samples(inputs, curves):
    a = _run(inputs, curves, n_samples=100, keep_total_samples=True, cost_asset_cv=1)
    b = _run(inputs, curves, n_samples=300, keep_total_samples=True, cost_asset_cv=1)
    assert_allclose(a.total_samples, b.total_samples[:100], rtol=1e-12)


def test_agrees_in_distribution_with_calculate_direct_damages(curves):
    rng = numpy.random.default_rng(5)
    asset_index = numpy.repeat(numpy.arange(3), [1, 2, 3])
    intensity = rng.random((6, 3)) * 1.5 + 0.2
    quantity = rng.random(6) + 1.0
    return_periods = [10.0, 100.0, 1000.0]
    options = dict(
        FULL_UNCERTAINTY, curve_weights=[0.4, 0.6], cost_mean=10.0, n_samples=20_000
    )
    new = simulate_direct_damages(
        intensity,
        asset_index,
        quantity,
        curves=curves,
        return_periods=return_periods,
        quantiles=(0.25, 0.5, 0.75),
        **options,
    )
    splits = pandas.DataFrame(
        {"asset": asset_index, "quantity": quantity}
        | {f"rp{rp:g}": intensity[:, i] for i, rp in enumerate(return_periods)}
    )
    old = calculate_direct_damages(
        splits,
        asset_id="asset",
        quantity="quantity",
        intensity_columns={rp: f"rp{rp:g}" for rp in return_periods},
        curves={"a": curves[0], "b": curves[1]},
        return_periods={rp: rp for rp in return_periods},
        quantiles=(0.25, 0.5, 0.75),
        ead_quantiles=(0.25, 0.5, 0.75),
        aggregate_assets=True,
        seed=1,
        **(options | {"curve_weights": {"a": 0.4, "b": 0.6}}),
    )
    old_cost = old.quantiles[["q_25", "q_50", "q_75"]].to_numpy().reshape(3, 3, 3)
    old_ead = old.ead_quantiles[["q_25", "q_50", "q_75"]].to_numpy()
    assert_allclose(new.asset_cost, old_cost, rtol=0.04)
    assert_allclose(new.asset_ead, old_ead, rtol=0.04)
    old_total = old.total_ead_quantiles[["q_25", "q_50", "q_75"]].to_numpy()[0]
    assert_allclose(new.total_ead, old_total, rtol=0.04)


def test_empty_input_gives_empty_and_zero_results(curves):
    result = simulate_direct_damages(
        numpy.zeros((0, 2)),
        numpy.zeros(0, dtype=int),
        numpy.zeros(0),
        curves=curves,
        curve_weights=[1, 1],
        cost_mean=1.0,
        n_assets=2,
        keep_total_samples=True,
    )
    assert result.split_cost.shape == (0, 2, 3)
    assert (result.asset_cost == 0).all()
    assert (result.total_samples == 0).all()


@pytest.mark.parametrize(
    "change, message",
    [
        ({"intensity": [1.0, 2.0]}, "intensity must have shape"),
        ({"intensity": [[1.0, numpy.nan], [0, 0]]}, "finite and non-negative"),
        ({"intensity": [[1.0, -1.0], [0, 0]]}, "finite and non-negative"),
        ({"intensity": [[1.0, numpy.inf], [0, 0]]}, "finite and non-negative"),
        ({"asset_index": [0]}, "asset_index must have shape"),
        ({"asset_index": [0.0, 1.0]}, "asset_index must contain integers"),
        ({"asset_index": [0, -1]}, "asset_index must be non-negative"),
        ({"asset_index": [0, 2], "n_assets": 2}, "less than n_assets"),
        ({"quantity": [1.0]}, "quantity must have shape"),
        ({"quantity": [1.0, -1.0]}, "quantity must be finite"),
        ({"curves": []}, "curves must be a non-empty"),
        ({"curves": [object()], "curve_weights": [1]}, "curves must be a non-empty"),
        ({"curve_weights": [1.0]}, "one weight per curve"),
        ({"curve_weights": [1.0, -1.0]}, "curve_weights must be finite"),
        ({"curve_weights": [0.0, 0.0]}, "curve_weights must be finite"),
        ({"cost_mean": -1.0}, "cost_mean must be finite"),
        ({"cost_mean": numpy.nan}, "cost_mean must be finite"),
        ({"cost_mean": [1.0, 2.0, 3.0]}, "cost_mean must be a scalar"),
        ({"cost_systemic_cv": -0.1}, "cost_systemic_cv"),
        ({"cost_asset_cv": numpy.nan}, "cost_asset_cv"),
        ({"eta_sd": None}, "eta_sd"),
        ({"hazard_cv": -0.1}, "hazard_cv must be finite"),
        ({"hazard_cv": [0.1, 0.2, 0.3]}, "hazard_cv must be a scalar"),
        ({"n_samples": 0}, "n_samples"),
        ({"n_samples": 1.5}, "n_samples"),
        ({"n_samples": True}, "n_samples"),
        ({"quantiles": []}, "quantiles must be a non-empty"),
        ({"quantiles": [0.5, 1.5]}, "quantiles must be a non-empty"),
        ({"quantiles": [0.5, 0.5]}, "duplicates"),
        ({"return_periods": [10.0]}, "one value per event"),
        ({"return_periods": [10.0, 10.0]}, "unique"),
        ({"return_periods": [0.5, 10.0]}, "at least one year"),
        ({"seed": -1}, "seed"),
        ({"seed": 2**64}, "seed"),
        ({"seed": 1.0}, "seed"),
        ({"asset_keys": [1]}, "asset_keys must have shape"),
        ({"asset_keys": [1, 1]}, "asset_keys must be unique"),
        ({"asset_keys": [1.0, 2.0]}, "asset_keys must contain integers"),
        ({"systemic_key": 2**64}, "systemic_key"),
        ({"systemic_key": "a"}, "systemic_key"),
        ({"systemic_key": True}, "systemic_key"),
        ({"levels": ("asset", "group")}, "levels must be a subset"),
        ({"levels": ()}, "at least one level"),
        ({"block_size": 0}, "block_size"),
    ],
)
def test_validates_inputs(curves, change, message):
    arguments = {
        "intensity": [[1.0, 2.0], [0.0, 0.5]],
        "asset_index": [0, 1],
        "quantity": [1.0, 2.0],
        "curves": curves,
        "curve_weights": [1.0, 1.0],
        "cost_mean": 1.0,
        "return_periods": [10.0, 100.0],
    }
    arguments.update(change)
    with pytest.raises(ValueError, match=message):
        simulate_direct_damages(
            arguments.pop("intensity"),
            arguments.pop("asset_index"),
            arguments.pop("quantity"),
            **arguments,
        )


@pytest.mark.parametrize(
    "call, message",
    [
        (lambda: summarise_samples(numpy.zeros((2, 0)), [0.5]), "at least one"),
        (lambda: summarise_samples([1.0, 2.0], [2.0]), "quantiles"),
        (lambda: expected_annual_damage(numpy.zeros((2, 3)), [1, 2]), "per event"),
        (lambda: expected_annual_damage(numpy.zeros((2, 2)), [1, 1]), "unique"),
    ],
)
def test_helpers_validate_inputs(call, message):
    with pytest.raises(ValueError, match=message):
        call()
