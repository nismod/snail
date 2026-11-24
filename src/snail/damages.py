"""Damage assessment"""

import operator
from abc import ABC, abstractmethod
from collections.abc import Callable, Collection, Mapping, Sequence
from dataclasses import dataclass
from itertools import pairwise
from numbers import Real
from typing import Any

import numpy
import pandas
import pandera.pandas as pandera
from numpy.typing import ArrayLike
from pandera.typing import DataFrame, Series
from scipy.interpolate import interp1d
from scipy.special import ndtri


@dataclass(frozen=True)
class DirectDamageResult:
    """Summarised sampled direct damages and their reproducibility metadata.

    ``metadata`` records the seed and model inputs. ``quantiles`` is a long-form
    frame with one row per asset, scenario and intensity case, when requested.
    ``ead_quantiles`` contains one row per asset and scenario when requested.
    Quantile columns are named ``q_05``, ``q_50``, etc.

    When asset aggregation is requested, ``total_quantiles`` and
    ``total_ead_quantiles`` summarise sums across assets within each scenario.
    All raw samples are discarded after summarising.
    """

    metadata: dict[str, Any]
    quantiles: pandas.DataFrame | None
    ead_quantiles: pandas.DataFrame | None = None
    total_quantiles: pandas.DataFrame | None = None
    total_ead_quantiles: pandas.DataFrame | None = None


def _is_finite_non_negative(value: Any) -> bool:
    return (
        isinstance(value, Real)
        and not isinstance(value, bool)
        and bool(numpy.isfinite(value))
        and value >= 0
    )


def _validate_quantiles(
    quantiles: Sequence[float] | None, name: str
) -> tuple[tuple[float, ...] | None, list[str] | None]:
    if quantiles is None:
        return None, None
    quantiles = tuple(quantiles)
    if len(quantiles) == 0 or any(
        not isinstance(q, Real) or not numpy.isfinite(q) or not 0 <= q <= 1
        for q in quantiles
    ):
        raise ValueError(f"{name} must be a non-empty sequence in [0, 1]")
    if len(set(quantiles)) != len(quantiles):
        raise ValueError(f"{name} must not contain duplicates")
    # avoid rounding - keep precision in label in case of non-whole-percentile quantiles
    columns = [f"q_{(q * 100)}".removesuffix(".0") for q in quantiles]
    if len(set(columns)) != len(columns):
        raise ValueError(f"{name} must map to distinct percentile column names")
    return quantiles, columns


def _validate_return_periods(
    return_periods: Mapping[Any, float] | None,
    intensity_columns: Mapping[Any, str],
) -> dict[Any, float] | None:
    """Check case labels and normalise finite, unique return periods in years."""
    if return_periods is None:
        return None
    if not isinstance(return_periods, Mapping):
        # Preserve the public API's ValueError for invalid return-period inputs.
        raise ValueError("return_periods must map intensity-case labels to years")  # noqa: TRY004
    if set(return_periods) != set(intensity_columns):
        raise ValueError("return_periods keys must match intensity_columns")
    try:
        values = {case: float(period) for case, period in return_periods.items()}
    except (TypeError, ValueError) as error:
        raise ValueError(
            "return periods must be finite numbers of at least one year"
        ) from error
    periods = numpy.asarray(list(values.values()), dtype=float)
    if (
        not numpy.isfinite(periods).all()
        or (periods < 1.0).any()
        or len(set(periods)) != len(periods)
    ):
        raise ValueError("return periods must be finite, unique, and at least one year")
    return values


def _quantile_summary(
    samples: numpy.ndarray, quantiles: Sequence[float], columns: Sequence[str]
) -> dict[str, float]:
    """Calculate requested quantiles together and map them to output columns."""
    values = numpy.quantile(samples, quantiles)
    return {column: float(value) for column, value in zip(columns, values)}


def _integrate_ead_samples(
    sampled_damage_by_case: Mapping[Any, numpy.ndarray],
    return_periods: Mapping[Any, float],
) -> numpy.ndarray:
    """
    Integrate arrays of damage-probability pairs to find Expected Annual
    Damages (EAD).

    Damage is assumed to vary linearly with natural log of annual exceedance
    probability between supplied events. Each interval's contribution to EAD is
    integrated analytically with respect to annual exceedance probability.

    We peform constant extrapolation in damage from the lowest supplied
    probability to zero probability. There is no extrapolation beyond the
    highest supplied probability.

    Parameters
    ----------
    sampled_damage_by_case : Mapping[Any, numpy.ndarray]
        Mapping from ``k`` intensity-case labels to one-dimensional damage
        arrays of shape ``(n,)``, where ``n`` is the number of simulation
        samples. All arrays must have the same length.
    return_periods : Mapping[Any, float]
        Mapping from the same ``k`` case labels to finite, unique return periods
        in years, each at least one year. At least one case is required. Annual
        exceedance probability is ``1 / return_period``. Cases are sorted by
        increasing probability before integration.

    Returns
    -------
    numpy.ndarray
        Expected annual damage samples of shape ``(n,)``, including the
        low-probability tail rectangle. Each output corresponds to the same
        sample index in the input damage arrays.
    """
    cases = sorted(return_periods, key=lambda case: 1.0 / return_periods[case])
    p = numpy.asarray([1.0 / return_periods[case] for case in cases])
    D = numpy.column_stack([sampled_damage_by_case[case] for case in cases])
    return _integrate_sorted_ead(D, p)


def _integrate_sorted_ead(
    D: numpy.ndarray, p: numpy.ndarray, axis: int = -1
) -> numpy.ndarray:
    """Integrate damages ``D`` over increasing probabilities ``p`` of shape
    ``(k,)``, on ``D``'s ``axis`` of length ``k``, removing that axis."""
    axis = axis % D.ndim
    trailing = (1,) * (D.ndim - axis - 1)

    def along(index):
        return D[(slice(None),) * axis + (index,)]

    p = p.reshape(p.shape + trailing)
    p_0 = p[:-1]  # lower probability
    p_1 = p[1:]  # higher probability
    w = p_1 - p_0  # width
    L = numpy.log(p_1 / p_0)  # log ratio
    lambd = w / L  # log mean, also known as lambda

    # Assume:
    # p_0 < p_1
    # w = p_1 - p_0
    # L = ln(p_1 / p_0)
    # delta_D = D_1 - D_0

    # D(p) = D_0 + s * ln(p / p_0)
    # s = delta_D / L
    # J = integral from p_0 to p_1 of ln(p / p_0) dp
    #   = [p * ln(p / p_0) - p] evaluated from p_0 to p_1
    #   = p_1 * L - w
    # I = D_0 * w + s * J
    #   = D_0 * w + delta_D * (p_1 - w/L)
    # If lambda = w/L:
    # I = D_0 * w + (D_1 - D_0) * (p_1 - lambda)
    #   = D_0 * (lambda - p_0) + D_1 * (p_1 - lambda)

    areas = along(slice(None, -1)) * (lambd - p_0) + along(slice(1, None)) * (
        p_1 - lambd
    )
    integral = areas.sum(axis=axis)
    rare_tail = p[0] * along(0)

    return integral + rare_tail


def calculate_direct_damages(
    splits: pandas.DataFrame,
    *,
    asset_id: str,
    quantity: str,
    intensity_columns: Mapping[Any, str],
    hazard_cv: float | Mapping[Any, float] = 0.0,
    curves: Mapping[str, "DamageCurve"],
    curve_weights: Mapping[str, float],
    eta_sd: float = 0.0,
    cost_mean: float | str,
    cost_asset_cv: float = 0.0,
    cost_systemic_cv: float = 0.0,
    n_samples: int = 1000,
    quantiles: Sequence[float] | None = None,
    return_periods: Mapping[Any, float] | None = None,
    ead_quantiles: Sequence[float] | None = None,
    seed: int | None = None,
    scenario: str | None = None,
    aggregate_assets: bool = False,
) -> DirectDamageResult:
    """Sample and summarise direct damages for pre-split assets.

    Each row in ``splits`` is one asset split. ``intensity_columns`` maps
    intensity-case labels (such as named events or return-periods) to hazard
    intensity columns. Each simulation selects a weighted damage curve and
    draws a systemic cost multiplier shared across all assets and scenarios.
    Each asset/scenario group then receives independent hazard, damage-noise
    and local cost draws, shared across its splits and intensity cases.
    Hazard and cost values are clamped at zero, damage fractions are clipped to [0, 1],
    and asset splits damages are summed for each intensity case. The result
    returns requested quantiles and run metadata. Samples can optionally be
    accumulated across assets to calculate total quantiles.

    In the equations below, ``Z`` denotes a draw from a standard normal
    distribution with mean 0 and standard deviation 1.

    Parameters
    ----------
    splits : pandas.DataFrame
        Pre-split asset data, with one row per split. Each split's quantity
        and hazard intensities are used to calculate and aggregate its damage.
    asset_id : str
        Name of the column identifying the parent asset for each split.
    quantity : str
        Name of the column giving each split's non-negative quantity: count
        (normally 1) for points, length for lines, or area for polygons.
        Units must match the per-unit rehabilitation cost. Each asset has one
        geometry type, but different assets may use different quantity units.
        Geometry measurement and unit consistency are the caller's responsibility.
    intensity_columns : Mapping[Any, str]
        Mapping from an intensity-case label to the column containing its
        central hazard intensity for each segment. Labels identify the cases
        in the returned summaries and may represent any kind of event or
        return-period maps, for example.
    hazard_cv : float or Mapping[Any, float], default 0.0
        Coefficient of variation of hazard intensity before clamping. Supply
        one value for all cases or a mapping from each intensity-case label
        to its value; the mapping keys must exactly match ``intensity_columns``.
        A case's sampled intensity is
        ``max(0, mean * (1 + hazard_cv * Z))``.
    curves : Mapping[str, DamageCurve]
        Damage curves available for sampling, keyed by identifiers that match
        the keys in ``curve_weights``.
    curve_weights : Mapping[str, float]
        Non-negative relative selection weights for the curves. Weights are
        normalised before sampling and must have a positive total.
    eta_sd : float, default 0.0
        Standard deviation of the normal damage-noise term. The sampled
        damage fraction is multiplied by
        ``exp(eta_sd * Z - eta_sd**2 / 2)`` before clipping to [0, 1].
    cost_mean : float or str
        Baseline rehabilitation cost per unit of quantity. Give a non-negative number
        to use the same mean for all assets, or a column name to read an
        asset-specific mean. Column values must be constant across that
        asset's splits within each scenario. For a point with quantity 1,
        this is the whole-asset cost; for lines and polygons, value is assumed
        uniformly distributed per unit length or area.
    cost_asset_cv : float, default 0.0
        Coefficient of variation of the asset normal cost factor before clamping.
        Sampled unit cost is ``max(0, systemic_cost * (1 + cost_asset_cv * Z))``,
        with an independent draw for each asset/scenario group, shared across
        its splits and intensity cases. Zero disables asset-specific cost variation.
    cost_systemic_cv : float, default 0.0
        Finite, non-negative coefficient of variation of the systemic normal
        cost factor before clamping. Each simulation uses
        ``systemic_cost = cost_mean * max(0, 1 + cost_systemic_cv * Z_system)``.
        The same dimensionless multiplier applies to every asset/scenario
        baseline, including assets with different quantity units.
        Zero disables systemic cost variation.
    n_samples : int, default 1000
        Number of uncertainty realisations to draw for each asset or
        asset/scenario group.
    quantiles : Sequence[float] or None, default None
        Quantiles to report for each asset and intensity case. Values must be
        in [0, 1]; output columns use percentile names such as ``q_05``. Set
        to None to omit per-intensity quantiles.
    return_periods : Mapping[Any, float] or None, default None
        Mapping from intensity-case labels to return periods in years. Supply
        this together with ``ead_quantiles`` to request expected annual damage
        (EAD) quantiles. Keys must exactly match ``intensity_columns``.
        Note that return periods RP are interpreted as the reciprocal of annual
        exceedance probability (AEP), so ``AEP = 1 / RP``. If the hazard data
        instead defines return period R as the mean recurrence interval of
        events exceeding a theshold, it should be converted to ``1 / AEP``.
        For example, assuming a Poisson process for events, R = 2 years
        corresponds to AEP ~0.393 or RP ~2.54. This distinction would affect
        expected annual damage (EAD) calculations.
    ead_quantiles : Sequence[float] or None, default None
        Quantiles to report for each asset/scenario EAD distribution. Requires
        ``return_periods``. Values use the same validation and column naming as
        ``quantiles``.
    seed : int or None, default None
        Seed for random number generator. Set a seed to make the sampled
        summaries reproducible for the same inputs and row order.
    scenario : str or None, default None
        Optional column identifying scenarios. If supplied, aggregation and
        local random draws are grouped by both asset and scenario; otherwise
        they are grouped by asset alone. Systemic draws span all scenarios.
    aggregate_assets : bool, default False
        Sum matching simulation samples across assets, separately within each
        scenario. Return ``total_quantiles`` at ``quantiles`` when supplied.
        When EAD is requested, integrate the summed event samples and return
        ``total_ead_quantiles`` at ``ead_quantiles``. Total summaries include
        ``n_assets`` and ``n_splits``.

    Returns
    -------
    DirectDamageResult
        ``quantiles`` contains one row per asset, scenario (if supplied), and
        intensity case when requested. ``ead_quantiles`` contains one row per
        asset/scenario group when requested. Both have ``n_splits`` and one
        column per requested quantile. ``metadata`` records the seed and model
        configuration. When ``aggregate_assets`` is True, the ``total_*``
        fields contain corresponding total quantile summaries; otherwise
        these fields are None. Total event summaries use one row per scenario
        and intensity case; total EAD summaries use one row per scenario.

    Raises
    ------
    KeyError
        If a required input column is missing.
    ValueError
        If parameters are invalid, numeric inputs are missing or non-finite,
        or a column-based cost mean varies within an asset/scenario group.

    Notes
    -----
    Curve choices and systemic cost multipliers are shared across the entire call
    at each simulation index. Hazard, damage noise and local cost draws are
    independent between asset/scenario groups and shared across all splits
    and intensity cases within each group. Both cost stages are clamped at
    zero; the final cost mean may therefore differ from the input baseline.

    Global damage curve choices are made first, then systemic cost draws, then
    local draws for each group. Metadata records inputs and draw_scopes ("call"
    or "asset_scenario").

    Asset aggregation sums corresponding simulation indices before calculating
    total quantiles. This preserves shared systemic uncertainty and local
    variation. Total EAD integrates the summed event damages; the fixed
    integration weights make this equivalent to summing asset EAD samples.

    If EAD is requested, each sample's damage is assumed linear in the natural
    logarithm of annual exceedance probability between supplied events. These
    intervals are integrated analytically with respect to probability. We peform
    constant extrapolation in damage from the lowest supplied probability to
    zero probability. There is no extrapolation beyond the highest supplied
    probability. Quantiles are calculated from the integrated samples.
    """
    required = {asset_id, quantity, *intensity_columns.values()}
    if scenario is not None:
        required.add(scenario)
    if isinstance(cost_mean, str):
        required.add(cost_mean)
    missing = required - set(splits.columns)
    if missing:
        raise KeyError(f"Required columns not present in splits: {sorted(missing)}")
    if not intensity_columns:
        raise ValueError("intensity_columns must contain at least one intensity case")
    if (
        not isinstance(n_samples, (int, numpy.integer))
        or isinstance(n_samples, bool)
        or n_samples <= 0
    ):
        raise ValueError("n_samples must be greater than zero")

    if quantiles is None and ead_quantiles is None:
        raise ValueError("request quantiles, ead_quantiles, or both")
    quantiles, quantile_columns = _validate_quantiles(quantiles, "quantiles")
    ead_quantiles, ead_quantile_columns = _validate_quantiles(
        ead_quantiles, "ead_quantiles"
    )
    if (ead_quantiles is None) != (return_periods is None):
        raise ValueError("return_periods and ead_quantiles must be supplied together")

    return_period_values = _validate_return_periods(return_periods, intensity_columns)

    if not _is_finite_non_negative(eta_sd) or not _is_finite_non_negative(
        cost_asset_cv
    ):
        raise ValueError("eta_sd and cost_asset_cv must be non-negative")
    if not _is_finite_non_negative(cost_systemic_cv):
        raise ValueError("cost_systemic_cv must be finite and non-negative")
    if isinstance(hazard_cv, Mapping):
        missing_cv = set(intensity_columns) - set(hazard_cv)
        if missing_cv:
            raise ValueError(f"hazard_cv missing intensity cases: {sorted(missing_cv)}")
        unknown_cv = set(hazard_cv) - set(intensity_columns)
        if unknown_cv:
            raise ValueError(
                f"hazard_cv has unknown intensity cases: {sorted(unknown_cv, key=str)}"
            )
        hazard_cv_by_case = {case: hazard_cv[case] for case in intensity_columns}
    else:
        hazard_cv_by_case = dict.fromkeys(intensity_columns, hazard_cv)
    if not all(_is_finite_non_negative(cv) for cv in hazard_cv_by_case.values()):
        raise ValueError("hazard_cv values must be non-negative")
    hazard_cv_by_case = {case: float(cv) for case, cv in hazard_cv_by_case.items()}

    if not curves or set(curves) != set(curve_weights):
        raise ValueError("curves and curve_weights must have the same non-empty keys")
    weights = numpy.asarray([curve_weights[key] for key in curves], dtype=float)
    if not numpy.isfinite(weights).all() or (weights < 0).any() or weights.sum() <= 0:
        raise ValueError("curve_weights must be finite, non-negative, and sum > 0")
    weights /= weights.sum()
    normalised_curve_weights = {key: weight for key, weight in zip(curves, weights)}
    curve_ids = list(curves)
    curve_values = [curves[key] for key in curve_ids]

    if not isinstance(cost_mean, str) and not _is_finite_non_negative(cost_mean):
        raise ValueError("cost_mean must be finite and non-negative")
    numeric_columns = list(dict.fromkeys([quantity, *intensity_columns.values()]))
    if isinstance(cost_mean, str) and cost_mean not in numeric_columns:
        numeric_columns.append(cost_mean)
    working = splits.copy()
    for column in numeric_columns:
        values = pandas.to_numeric(splits[column], errors="coerce")
        if values.isna().any() or not numpy.isfinite(values.to_numpy()).all():
            raise ValueError(f"Column '{column}' must contain finite numeric values")
        if (values < 0).any():
            raise ValueError(f"Column '{column}' must be non-negative")
        working[column] = values

    group_cols = [asset_id] + ([scenario] if scenario is not None else [])
    if working[group_cols].isna().any().any():
        raise ValueError("Asset and scenario identifiers must not be missing")

    rng = numpy.random.default_rng(seed)

    # s = the number of splits; n = n_samples
    # Make systemic draws, shared across the whole call
    selected_curve = rng.choice(len(curve_values), size=n_samples, p=weights)  # (n,)
    z_system = rng.standard_normal(n_samples)  # (n,)
    systemic_multiplier = numpy.maximum(0.0, 1.0 + cost_systemic_cv * z_system)  # (n,)

    rows: list[dict[str, Any]] = []
    ead_rows: list[dict[str, Any]] = []
    total_samples: dict[Any, dict[Any, numpy.ndarray]] = {}
    total_counts: dict[Any, dict[str, int]] = {}
    grouper: str | list[str] = group_cols[0] if len(group_cols) == 1 else group_cols

    # Asset-scenario loop
    for key, group in working.groupby(grouper, sort=False, dropna=False):
        key_values = (key,) if len(group_cols) == 1 else key
        # e.g. {"asset_id": "road_2718", "scenario": "future"}
        group_identifiers = dict(zip(group_cols, key_values))

        if aggregate_assets:
            scenario_key = group_identifiers[scenario] if scenario is not None else None
            if scenario_key not in total_samples:
                total_samples[scenario_key] = {
                    case: numpy.zeros(n_samples) for case in intensity_columns
                }
                total_counts[scenario_key] = {"n_assets": 0, "n_splits": 0}
            total_counts[scenario_key]["n_assets"] += 1
            total_counts[scenario_key]["n_splits"] += len(group)

        if isinstance(cost_mean, str):
            costs = group[cost_mean].to_numpy(dtype=float)
            if not numpy.allclose(costs, costs[0]):
                raise ValueError(
                    f"Cost mean must be constant within group {group_identifiers!r}"
                )
            group_cost_mean = float(costs[0])
        else:
            group_cost_mean = float(cost_mean)

        # Make local draws, shared across asset-scenario
        size = len(group)  # s
        z_hazard = rng.standard_normal(n_samples)  # (n,)
        z_damage = rng.standard_normal(n_samples)  # (n,)
        z_cost = rng.standard_normal(n_samples)  # (n,)
        systemic_cost = group_cost_mean * systemic_multiplier  # (n,)
        cost_draw = numpy.maximum(
            0.0, systemic_cost * (1.0 + cost_asset_cv * z_cost)
        )  # (n,)
        damage_multiplier = numpy.exp(eta_sd * z_damage - 0.5 * eta_sd**2)  # (n,)
        quantities = group[quantity].to_numpy(dtype=float)  # (s,)
        sampled_totals: dict[Any, numpy.ndarray] = {}  # each value: (n,)

        # Intensity case (e.g. return-period) iterator
        for case, column in intensity_columns.items():
            means = group[column].to_numpy(dtype=float)  # (s,)
            hazard = numpy.maximum(
                0.0,
                means[None, :] * (1.0 + hazard_cv_by_case[case] * z_hazard[:, None]),
            )  # (n, s)
            damage_fraction = numpy.empty((n_samples, size), dtype=float)  # (n, s)
            for curve_index, curve in enumerate(curve_values):
                mask = selected_curve == curve_index  # (n,)
                if mask.any():
                    damage_fraction[mask] = curve.damage_fraction(hazard[mask])
            damage_fraction = numpy.clip(
                damage_fraction * damage_multiplier[:, None], 0.0, 1.0
            )  # (n, s)
            total = (damage_fraction * quantities[None, :] * cost_draw[:, None]).sum(
                axis=1
            )  # (n,)
            if aggregate_assets:
                total_samples[scenario_key][case] += total

            if quantiles is not None:
                summary = dict(group_identifiers)
                summary["intensity_case"] = case
                summary["n_splits"] = size
                summary.update(_quantile_summary(total, quantiles, quantile_columns))
                rows.append(summary)
            if ead_quantiles is not None:
                sampled_totals[case] = total

        if ead_quantiles is not None:
            ead_samples = _integrate_ead_samples(
                sampled_totals, return_period_values
            )  # (n,)
            summary = dict(group_identifiers)
            summary["n_splits"] = size
            summary.update(
                _quantile_summary(ead_samples, ead_quantiles, ead_quantile_columns)
            )
            ead_rows.append(summary)

    total_rows: list[dict[str, Any]] = []
    total_ead_rows: list[dict[str, Any]] = []
    for scenario_key, totals in total_samples.items():
        total_identifiers = {scenario: scenario_key} if scenario is not None else {}
        total_identifiers.update(total_counts[scenario_key])
        if quantiles is not None:
            for case, samples in totals.items():
                summary = dict(total_identifiers, intensity_case=case)
                summary.update(_quantile_summary(samples, quantiles, quantile_columns))
                total_rows.append(summary)
        if ead_quantiles is not None:
            samples = _integrate_ead_samples(totals, return_period_values)
            summary = dict(total_identifiers)
            summary.update(
                _quantile_summary(samples, ead_quantiles, ead_quantile_columns)
            )
            total_ead_rows.append(summary)

    metadata = {
        "seed": seed,
        "n_samples": n_samples,
        "quantiles": list(quantiles) if quantiles is not None else None,
        "return_periods": return_period_values,
        "ead_quantiles": (list(ead_quantiles) if ead_quantiles is not None else None),
        "asset_id_column": asset_id,
        "scenario_column": scenario,
        "aggregate_assets": aggregate_assets,
        "quantity_column": quantity,
        "intensity_columns": dict(intensity_columns),
        "curve_ids": curve_ids,
        "curve_weights": dict(normalised_curve_weights),
        "cost_mean": cost_mean,
        "cost_asset_cv": cost_asset_cv,
        "cost_systemic_cv": cost_systemic_cv,
        "hazard_cv": hazard_cv_by_case,
        "eta_sd": eta_sd,
        "draw_scopes": {
            "damage_curve": "call",
            "cost_system": "call",
            "hazard": "asset_scenario",
            "damage_noise": "asset_scenario",
            "cost": "asset_scenario",
        },
    }
    quantile_frame = None
    if quantiles is not None:
        output_columns = [*group_cols, "intensity_case", "n_splits", *quantile_columns]
        quantile_frame = pandas.DataFrame(rows, columns=output_columns)
    ead_quantile_frame = None
    if ead_quantiles is not None:
        output_columns = [*group_cols, "n_splits", *ead_quantile_columns]
        ead_quantile_frame = pandas.DataFrame(ead_rows, columns=output_columns)

    total_quantile_frame = None
    total_ead_quantile_frame = None
    scenario_columns = [scenario] if scenario is not None else []
    if aggregate_assets and quantiles is not None:
        output_columns = [
            *scenario_columns,
            "intensity_case",
            "n_assets",
            "n_splits",
            *quantile_columns,
        ]
        total_quantile_frame = pandas.DataFrame(total_rows, columns=output_columns)
    if aggregate_assets and ead_quantiles is not None:
        output_columns = [
            *scenario_columns,
            "n_assets",
            "n_splits",
            *ead_quantile_columns,
        ]
        total_ead_quantile_frame = pandas.DataFrame(
            total_ead_rows, columns=output_columns
        )

    return DirectDamageResult(
        quantiles=quantile_frame,
        metadata=metadata,
        ead_quantiles=ead_quantile_frame,
        total_quantiles=total_quantile_frame,
        total_ead_quantiles=total_ead_quantile_frame,
    )


#: Version of the random-draw scheme used by :func:`simulate_direct_damages`.
#: Change it whenever the draws for given inputs would change.
DRAW_SCHEME = "philox4x64-keyed-v1"

# Philox key word 1 for each stream; key word 0 is the seed.
_STREAM_IDS = {"curve": 1, "system": 2, "hazard": 3, "damage": 4, "cost": 5}
# Philox counter word 3 marks what counter word 2 holds.
_COUNTER_LOCAL = 0  # word 2 is the asset key
_COUNTER_SYSTEMIC = 1  # word 2 is zero: no systemic key
_COUNTER_SYSTEMIC_KEYED = 2  # word 2 is the systemic key
_LEVELS = ("split", "asset", "total")
# Target number of float64 values in one (split, event, sample) block array.
_BLOCK_VALUES = 2**22


@dataclass(frozen=True)
class DirectDamageArrays:
    """Quantiles of sampled direct damages at split, asset and total level.

    Returned by :func:`simulate_direct_damages`. Arrays are aligned by
    position with its inputs: ``s`` splits, ``a`` assets, ``e`` events,
    ``q`` quantiles and ``n`` samples. Arrays for levels that were not
    requested, and EAD arrays when no return periods were given, are None.

    Attributes
    ----------
    quantiles : numpy.ndarray
        Requested quantiles, shape ``(q,)``.
    return_periods : numpy.ndarray or None
        Return periods of the events in years, shape ``(e,)``.
    split_fraction : numpy.ndarray or None
        Quantiles of each split's damage fraction, shape ``(s, e, q)``.
    split_cost : numpy.ndarray or None
        Quantiles of each split's damage cost, shape ``(s, e, q)``.
    split_ead : numpy.ndarray or None
        Quantiles of each split's expected annual damage, shape ``(s, q)``.
    asset_cost : numpy.ndarray or None
        Quantiles of each asset's damage cost, shape ``(a, e, q)``.
    asset_ead : numpy.ndarray or None
        Quantiles of each asset's expected annual damage, shape ``(a, q)``.
    total_cost : numpy.ndarray or None
        Quantiles of the total damage cost over all assets, shape ``(e, q)``.
    total_ead : numpy.ndarray or None
        Quantiles of the total expected annual damage, shape ``(q,)``.
    total_samples : numpy.ndarray or None
        Total damage cost samples, shape ``(n, e)``, when
        ``keep_total_samples`` was set. Sample ``i`` of separate calls with
        the same seed, ``n_samples``, curves and ``systemic_key`` shares the
        same curve choice and systemic cost draw, so these can be summed
        across calls and summarised with :func:`summarise_samples` and
        :func:`expected_annual_damage`.
    metadata : dict
        Seed, model inputs, normalised curve weights and ``draw_scheme``.
    """

    quantiles: numpy.ndarray
    return_periods: numpy.ndarray | None
    split_fraction: numpy.ndarray | None
    split_cost: numpy.ndarray | None
    split_ead: numpy.ndarray | None
    asset_cost: numpy.ndarray | None
    asset_ead: numpy.ndarray | None
    total_cost: numpy.ndarray | None
    total_ead: numpy.ndarray | None
    total_samples: numpy.ndarray | None
    metadata: dict[str, Any]


def _lerp(a, b, t: float):
    """Linear interpolation exactly as ``numpy.quantile`` does it."""
    diff_b_a = b - a
    if t >= 0.5:
        return b - diff_b_a * (1 - t)
    return a + diff_b_a * t


def _quantile_positions(
    n: int, quantiles: numpy.ndarray
) -> list[tuple[int, int, float]]:
    """Sorted-sample indices and weights of numpy's default (linear) method."""
    virtual = (n - 1) * quantiles
    positions = []
    for v in virtual:
        lo = int(numpy.floor(v))
        positions.append((lo, min(lo + 1, n - 1), float(v - lo)))
    return positions


def _check_quantiles(quantiles: Sequence[float], name: str = "quantiles"):
    try:
        values = numpy.asarray(quantiles, dtype=float)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be a non-empty sequence in [0, 1]") from error
    if (
        values.ndim != 1
        or len(values) == 0
        or not numpy.isfinite(values).all()
        or (values < 0).any()
        or (values > 1).any()
    ):
        raise ValueError(f"{name} must be a non-empty sequence in [0, 1]")
    if len(numpy.unique(values)) != len(values):
        raise ValueError(f"{name} must not contain duplicates")
    return values


def _check_return_periods(return_periods: ArrayLike, n_event: int | None = None):
    try:
        values = numpy.asarray(return_periods, dtype=float)
    except (TypeError, ValueError) as error:
        raise ValueError(
            "return periods must be finite numbers of at least one year"
        ) from error
    if values.ndim != 1 or len(values) == 0:
        raise ValueError("return_periods must be a non-empty one-dimensional array")
    if n_event is not None and len(values) != n_event:
        raise ValueError("return_periods must have one value per event")
    if (
        not numpy.isfinite(values).all()
        or (values < 1.0).any()
        or len(numpy.unique(values)) != len(values)
    ):
        raise ValueError("return periods must be finite, unique, and at least one year")
    return values


def _sorted_quantiles(sorted_samples: numpy.ndarray, quantiles: numpy.ndarray):
    """Quantiles over the last axis of already sorted samples: (..., n) -> (..., q)."""
    n = sorted_samples.shape[-1]
    out = numpy.empty(sorted_samples.shape[:-1] + (len(quantiles),))
    for j, (lo, hi, t) in enumerate(_quantile_positions(n, quantiles)):
        out[..., j] = _lerp(sorted_samples[..., lo], sorted_samples[..., hi], t)
    # numpy.quantile returns NaN for any slice containing NaN, which sorts last
    nan_rows = numpy.isnan(sorted_samples[..., -1])
    if nan_rows.any():
        out[nan_rows] = numpy.nan
    return out


def _summarise(samples: numpy.ndarray, quantiles: numpy.ndarray, overwrite=False):
    if overwrite:
        samples.sort(axis=-1)
        return _sorted_quantiles(samples, quantiles)
    return _sorted_quantiles(numpy.sort(samples, axis=-1), quantiles)


def _summarise_rows(samples, quantiles, wet=None, overwrite=False):
    """Summarise ``samples`` (..., n), returning exact zeros where ``wet``
    (of shape ``samples.shape[:-1]``) is False."""
    if wet is None or wet.all():
        return _summarise(samples, quantiles, overwrite=overwrite)
    out = numpy.zeros(samples.shape[:-1] + (len(quantiles),))
    if wet.any():
        out[wet] = _summarise(samples[wet], quantiles, overwrite=True)
    return out


def _discrete_quantiles(values, counts, quantiles):
    """Quantiles of samples that take ``values[..., k]`` in ``counts[k]`` of the
    samples, without forming the samples: (..., K) -> (..., q). Equal to
    :func:`summarise_samples` of the expanded samples."""
    n = int(counts.sum())
    order = numpy.argsort(values, axis=-1, kind="stable")
    sorted_values = numpy.take_along_axis(values, order, axis=-1)
    cumulative = numpy.cumsum(counts[order], axis=-1)
    out = numpy.empty(values.shape[:-1] + (len(quantiles),))

    def value_at(position):
        index = (cumulative <= position).sum(axis=-1, keepdims=True)
        return numpy.take_along_axis(sorted_values, index, axis=-1)[..., 0]

    for j, (lo, hi, t) in enumerate(_quantile_positions(n, quantiles)):
        out[..., j] = _lerp(value_at(lo), value_at(hi), t)
    return out


def summarise_samples(samples: ArrayLike, quantiles: Sequence[float]) -> numpy.ndarray:
    """Calculate quantiles of samples over their last axis.

    Uses one sort and numpy's default (linear) interpolation, so the values
    equal ``numpy.moveaxis(numpy.quantile(samples, quantiles, axis=-1), 0, -1)``.

    Parameters
    ----------
    samples : ArrayLike
        Samples of shape ``(..., n)``, for example ``total_samples.T`` summed
        across calls to :func:`simulate_direct_damages`.
    quantiles : Sequence[float]
        Unique quantiles in [0, 1].

    Returns
    -------
    numpy.ndarray
        Quantiles of shape ``(..., q)``. Slices containing NaN give NaN.

    Raises
    ------
    ValueError
        If quantiles are invalid or there are no samples.
    """
    quantiles = _check_quantiles(quantiles)
    samples = numpy.asarray(samples, dtype=float)
    if samples.ndim == 0 or samples.shape[-1] == 0:
        raise ValueError("samples must have at least one sample on the last axis")
    return _summarise(samples, quantiles)


def expected_annual_damage(
    samples: ArrayLike, return_periods: ArrayLike
) -> numpy.ndarray:
    """Integrate damage samples over events to expected annual damage (EAD).

    Uses the same maths as :func:`calculate_direct_damages`: damage is assumed
    linear in the natural logarithm of annual exceedance probability
    (``1 / return_period``) between events, integrated analytically, with
    constant extrapolation from the rarest event to zero probability and no
    extrapolation beyond the most frequent event.

    Parameters
    ----------
    samples : ArrayLike
        Damage samples of shape ``(n, e)``, events on the last axis (any
        leading shape ``(..., e)`` is accepted).
    return_periods : ArrayLike
        Return periods of the ``e`` events in years, in any order: finite,
        unique and at least one year.

    Returns
    -------
    numpy.ndarray
        EAD samples of shape ``(n,)`` (generally ``samples.shape[:-1]``).

    Raises
    ------
    ValueError
        If return periods are invalid or do not match the events.
    """
    samples = numpy.asarray(samples, dtype=float)
    if samples.ndim == 0:
        raise ValueError("samples must have events on the last axis")
    return_periods = _check_return_periods(return_periods, samples.shape[-1])
    order = numpy.argsort(1.0 / return_periods, kind="stable")
    p = 1.0 / return_periods[order]
    return _integrate_sorted_ead(samples[..., order], p)


def _ead_weights(return_periods: numpy.ndarray) -> numpy.ndarray:
    """Weights ``w`` of shape ``(e,)``, in the given event order, such that
    ``(D * w).sum(-1)`` equals :func:`expected_annual_damage` up to rounding.

    EAD is linear in the event damages, so the integration in
    :func:`_integrate_sorted_ead` collapses to one weight per event.
    """
    order = numpy.argsort(1.0 / return_periods, kind="stable")
    weights = numpy.empty(len(return_periods))
    weights[order] = _integrate_sorted_ead(
        numpy.eye(len(return_periods)), 1.0 / return_periods[order]
    )
    return weights


def _event_axis_ead(samples: numpy.ndarray, weights: numpy.ndarray):
    """EAD of samples with events on axis 1: (b, e, n) -> (b, n)."""
    return numpy.matmul(weights, samples)


def _to_uint64_key(value: Any, name: str) -> int:
    """Map an integer in [-2**63, 2**64) to a 64-bit key word."""
    if isinstance(value, (bool, numpy.bool_)):
        # Keep ValueError for all invalid inputs, as elsewhere in this module.
        raise ValueError(f"{name} must be an integer")  # noqa: TRY004
    try:
        value = operator.index(value)
    except TypeError as error:
        raise ValueError(f"{name} must be an integer") from error
    if not -(2**63) <= value < 2**64:
        raise ValueError(f"{name} must be in [-2**63, 2**64)")
    return value % 2**64


def _uniform_draws(
    seed: int, stream: str, words: numpy.ndarray, flag: int, n: int
) -> numpy.ndarray:
    """Draw ``n`` uniforms in (0, 1) for each counter word: (len(words), n).

    Each row comes from Philox4x64 with key ``(seed, stream id)`` and counter
    ``(0, 0, word, flag)``, so it depends only on these and not on the other
    rows. Each uniform uses one 64-bit output: ``((x >> 12) + 0.5) / 2**52``.
    """
    key = numpy.array([seed, _STREAM_IDS[stream]], dtype=numpy.uint64)
    counter = numpy.array([0, 0, 0, flag], dtype=numpy.uint64)
    raw = numpy.empty((len(words), n), dtype=numpy.uint64)
    for i, word in enumerate(words):
        counter[2] = word
        raw[i] = numpy.random.Philox(key=key, counter=counter).random_raw(n)
    raw >>= numpy.uint64(12)
    uniforms = raw.astype(float)
    uniforms += 0.5
    uniforms *= 2.0**-52
    return uniforms


def _normal_draws(seed, stream, words, flag, n):
    return ndtri(_uniform_draws(seed, stream, words, flag, n))


def _check_cv(value: Any, name: str) -> float:
    if not _is_finite_non_negative(value):
        raise ValueError(f"{name} must be finite and non-negative")
    return float(value)


def _as_float_array(value: Any, name: str) -> numpy.ndarray:
    try:
        return numpy.asarray(value, dtype=float)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be numeric") from error


@dataclass
class _Simulation:
    """Validated inputs of :func:`simulate_direct_damages`."""

    intensity: numpy.ndarray  # (s, e)
    asset_index: numpy.ndarray  # (s,)
    quantity: numpy.ndarray  # (s,)
    curves: list
    weights: numpy.ndarray  # (K,)
    cost_mean: numpy.ndarray  # (a,)
    cost_mean_scalar: float | None
    cost_systemic_cv: float
    cost_asset_cv: float
    hazard_cv: numpy.ndarray  # (e,)
    eta_sd: float
    n_samples: int
    quantiles: numpy.ndarray
    return_periods: numpy.ndarray | None
    seed: int
    asset_keys: numpy.ndarray  # (a,) uint64
    systemic_key: int | None
    systemic_word: int
    systemic_flag: int
    n_assets: int
    levels: frozenset
    keep_total_samples: bool
    block_size: int


def _validate_simulation(
    intensity,
    asset_index,
    quantity,
    *,
    curves,
    curve_weights,
    cost_mean,
    cost_systemic_cv,
    cost_asset_cv,
    hazard_cv,
    eta_sd,
    n_samples,
    quantiles,
    return_periods,
    seed,
    asset_keys,
    systemic_key,
    n_assets,
    levels,
    keep_total_samples,
    block_size,
) -> _Simulation:
    intensity = _as_float_array(intensity, "intensity")
    if intensity.ndim != 2 or intensity.shape[1] == 0:
        raise ValueError("intensity must have shape (n_split, n_event), n_event > 0")
    if not numpy.isfinite(intensity).all() or (intensity < 0).any():
        raise ValueError("intensity must be finite and non-negative")
    n_split, n_event = intensity.shape

    asset_index = numpy.asarray(asset_index)
    if asset_index.shape != (n_split,):
        raise ValueError("asset_index must have shape (n_split,)")
    if n_split and asset_index.dtype.kind not in "iu":
        raise ValueError("asset_index must contain integers")
    asset_index = asset_index.astype(numpy.intp)
    if (asset_index < 0).any():
        raise ValueError("asset_index must be non-negative")
    min_assets = int(asset_index.max()) + 1 if n_split else 0
    if n_assets is None:
        n_assets = min_assets
    else:
        if isinstance(n_assets, bool) or not isinstance(n_assets, (int, numpy.integer)):
            raise ValueError("n_assets must be an integer")
        n_assets = int(n_assets)
        if n_assets < min_assets:
            raise ValueError("asset_index must be less than n_assets")

    quantity = _as_float_array(quantity, "quantity")
    if quantity.shape != (n_split,):
        raise ValueError("quantity must have shape (n_split,)")
    if not numpy.isfinite(quantity).all() or (quantity < 0).any():
        raise ValueError("quantity must be finite and non-negative")

    curves = list(curves)
    if not curves or any(
        not callable(getattr(curve, "damage_fraction", None)) for curve in curves
    ):
        raise ValueError("curves must be a non-empty sequence of damage curves")
    weights = _as_float_array(curve_weights, "curve_weights")
    if weights.shape != (len(curves),):
        raise ValueError("curve_weights must have one weight per curve")
    if not numpy.isfinite(weights).all() or (weights < 0).any() or weights.sum() <= 0:
        raise ValueError("curve_weights must be finite, non-negative, and sum > 0")
    weights = weights / weights.sum()

    cost_mean = _as_float_array(cost_mean, "cost_mean")
    cost_mean_scalar = float(cost_mean) if cost_mean.ndim == 0 else None
    if cost_mean.ndim == 0:
        cost_mean = numpy.full(n_assets, cost_mean_scalar)
    elif cost_mean.shape != (n_assets,):
        raise ValueError("cost_mean must be a scalar or have shape (n_assets,)")
    if not numpy.isfinite(cost_mean).all() or (cost_mean < 0).any():
        raise ValueError("cost_mean must be finite and non-negative")
    if cost_mean_scalar is not None and not cost_mean_scalar >= 0:
        raise ValueError("cost_mean must be finite and non-negative")

    cost_systemic_cv = _check_cv(cost_systemic_cv, "cost_systemic_cv")
    cost_asset_cv = _check_cv(cost_asset_cv, "cost_asset_cv")
    eta_sd = _check_cv(eta_sd, "eta_sd")
    hazard_cv = _as_float_array(hazard_cv, "hazard_cv")
    if hazard_cv.ndim == 0:
        hazard_cv = numpy.full(n_event, float(hazard_cv))
    elif hazard_cv.shape != (n_event,):
        raise ValueError("hazard_cv must be a scalar or have shape (n_event,)")
    if not numpy.isfinite(hazard_cv).all() or (hazard_cv < 0).any():
        raise ValueError("hazard_cv must be finite and non-negative")

    if (
        not isinstance(n_samples, (int, numpy.integer))
        or isinstance(n_samples, bool)
        or n_samples <= 0
    ):
        raise ValueError("n_samples must be a positive integer")
    quantiles = _check_quantiles(quantiles)
    if return_periods is not None:
        return_periods = _check_return_periods(return_periods, n_event)

    if (
        isinstance(seed, bool)
        or not isinstance(seed, (int, numpy.integer))
        or not 0 <= seed < 2**64
    ):
        raise ValueError("seed must be an integer in [0, 2**64)")

    if asset_keys is None:
        keys = numpy.arange(n_assets, dtype=numpy.uint64)
    else:
        keys = numpy.asarray(asset_keys)
        if keys.shape != (n_assets,):
            raise ValueError("asset_keys must have shape (n_assets,)")
        if n_assets and keys.dtype.kind not in "iu":
            raise ValueError("asset_keys must contain integers")
        # Signed keys (such as hashes) map to their two's complement bits
        keys = keys.astype(numpy.int64 if keys.dtype.kind == "i" else numpy.uint64)
        keys = keys.view(numpy.uint64)
        if len(numpy.unique(keys)) != n_assets:
            raise ValueError("asset_keys must be unique")

    if systemic_key is None:
        systemic_word, systemic_flag = 0, _COUNTER_SYSTEMIC
    else:
        systemic_word = _to_uint64_key(systemic_key, "systemic_key")
        systemic_flag = _COUNTER_SYSTEMIC_KEYED

    if isinstance(levels, str):
        levels = (levels,)
    levels = frozenset(levels)
    if not levels <= set(_LEVELS):
        raise ValueError(f"levels must be a subset of {_LEVELS}")
    if not levels and not keep_total_samples:
        raise ValueError("request at least one level, or keep_total_samples")

    if block_size is None:
        block_size = max(1, _BLOCK_VALUES // (n_samples * n_event))
    elif (
        isinstance(block_size, bool)
        or not isinstance(block_size, (int, numpy.integer))
        or block_size <= 0
    ):
        raise ValueError("block_size must be a positive integer or None")

    return _Simulation(
        intensity=intensity,
        asset_index=asset_index,
        quantity=quantity,
        curves=curves,
        weights=weights,
        cost_mean=cost_mean,
        cost_mean_scalar=cost_mean_scalar,
        cost_systemic_cv=cost_systemic_cv,
        cost_asset_cv=cost_asset_cv,
        hazard_cv=hazard_cv,
        eta_sd=eta_sd,
        n_samples=int(n_samples),
        quantiles=quantiles,
        return_periods=return_periods,
        seed=int(seed),
        asset_keys=keys,
        systemic_key=None if systemic_key is None else operator.index(systemic_key),
        systemic_word=systemic_word,
        systemic_flag=systemic_flag,
        n_assets=n_assets,
        levels=levels,
        keep_total_samples=bool(keep_total_samples),
        block_size=int(block_size),
    )


def simulate_direct_damages(
    intensity: ArrayLike,
    asset_index: ArrayLike,
    quantity: ArrayLike,
    *,
    curves: Sequence["DamageCurve"],
    curve_weights: Sequence[float],
    cost_mean: float | ArrayLike,
    cost_systemic_cv: float = 0.0,
    cost_asset_cv: float = 0.0,
    hazard_cv: float | ArrayLike = 0.0,
    eta_sd: float = 0.0,
    n_samples: int = 1000,
    quantiles: Sequence[float] = (0.05, 0.5, 0.95),
    return_periods: ArrayLike | None = None,
    seed: int = 0,
    asset_keys: ArrayLike | None = None,
    systemic_key: int | None = None,
    n_assets: int | None = None,
    levels: Collection[str] = ("split", "asset", "total"),
    keep_total_samples: bool = False,
    block_size: int | None = None,
) -> DirectDamageArrays:
    """Sample direct damages once and summarise them at split, asset and
    total level.

    An array form of :func:`calculate_direct_damages` for a single scenario
    combination, with the same uncertainty model. Inputs are aligned by
    position: ``s`` splits, each belonging to one of ``a`` assets, and ``e``
    events (for example the return periods of one climate scenario and
    epoch). Each simulation sample chooses a weighted damage curve and draws a
    systemic cost factor, shared by all assets. Each asset draws a hazard
    factor, a damage-noise factor and a local cost factor, shared by its
    splits and events. Split damages are summed to assets and assets to the
    total, sample by sample, and all levels are summarised from the same
    samples.

    In the equations below, ``Z`` denotes a standard normal draw. For split
    ``i`` of asset ``j`` and event ``k``:

    - hazard ``h = max(0, intensity[i, k] * (1 + hazard_cv[k] * Z_hazard[j]))``
    - damage fraction
      ``f = clip(curve(h) * exp(eta_sd * Z_damage[j] - eta_sd**2 / 2), 0, 1)``
    - unit cost ``u = max(0, cost_mean[j] * max(0, 1 + cost_systemic_cv *
      Z_system) * (1 + cost_asset_cv * Z_cost[j]))``
    - split cost ``f * quantity[i] * u``

    Parameters
    ----------
    intensity : ArrayLike
        Central hazard intensity of shape ``(s, e)``. Must be finite and
        non-negative: the caller decides what missing data means.
    asset_index : ArrayLike
        Integer parent asset of each split, shape ``(s,)``, in
        ``[0, n_assets)``.
    quantity : ArrayLike
        Non-negative quantity of each split, shape ``(s,)``, in the units of
        the cost (count, length or area).
    curves : Sequence[DamageCurve]
        Damage curves available for sampling.
    curve_weights : Sequence[float]
        Non-negative relative selection weights, one per curve, with a
        positive total. Normalised weights are recorded in the metadata.
    cost_mean : float or ArrayLike
        Baseline cost per unit of quantity: a non-negative scalar, or one
        value per asset, shape ``(a,)``.
    cost_systemic_cv : float, default 0.0
        Coefficient of variation of the systemic cost factor, before clamping.
    cost_asset_cv : float, default 0.0
        Coefficient of variation of each asset's local cost factor, before
        clamping.
    hazard_cv : float or ArrayLike, default 0.0
        Coefficient of variation of hazard intensity before clamping: a
        scalar, or one value per event, shape ``(e,)``.
    eta_sd : float, default 0.0
        Standard deviation of the normal damage-noise term.
    n_samples : int, default 1000
        Number of simulation samples.
    quantiles : Sequence[float], default (0.05, 0.5, 0.95)
        Unique quantiles in [0, 1] to report at every level. Calculated with
        numpy's default (linear) method.
    return_periods : ArrayLike or None, default None
        Return periods of the events in years, shape ``(e,)``: finite, unique
        and at least one year. Enables expected annual damage (EAD) arrays,
        integrated per sample as in :func:`expected_annual_damage`. Split and
        asset EAD apply that integration as one weight per event, equal up to
        rounding; total EAD is ``expected_annual_damage(total_samples)``.
    seed : int, default 0
        Seed in [0, 2**64) for all random streams.
    asset_keys : ArrayLike or None, default None
        Unique integer key of each asset, shape ``(a,)``, that keys its local
        random draws, for example a 64-bit hash of a stable asset id. Signed
        and unsigned 64-bit keys are accepted; signed keys map to their two's
        complement bits. Defaults to ``arange(a)``, which is only stable for a
        fixed asset order.
    systemic_key : int or None, default None
        Optional key, in [-2**63, 2**64), for the curve choice and systemic
        cost draws. With None these draws depend only on ``seed``, so they are
        shared by every call with the same seed (perfectly correlated across
        asset types, layers and hazards). With an integer they also depend on
        the key, so for example a different key per asset type gives
        independent systemic draws per type. ``total_samples`` from calls
        with different keys can still be summed: they are then independent
        rather than perfectly correlated, which is the caller's modelling
        choice.
    n_assets : int or None, default None
        Number of assets, ``a``. Defaults to ``asset_index.max() + 1``. Assets
        without splits have zero damage.
    levels : Collection[str], default ("split", "asset", "total")
        Levels to summarise, any of ``"split"``, ``"asset"`` and ``"total"``.
        Arrays of other levels are None and are not computed.
    keep_total_samples : bool, default False
        Return ``total_samples``, shape ``(n, e)``. This is independent of
        ``levels``: total samples are kept even if ``"total"`` is not in
        ``levels``, for example to sum across calls before summarising.
    block_size : int or None, default None
        Target number of splits per vectorised block. Blocks hold whole assets,
        so a block can exceed it by part of one asset's splits. Memory is about
        a few times ``8 * block_size * e * n`` bytes. Defaults to about
        ``2**22 / (n * e)`` splits (32 MB per block array). Results do not
        depend on it, beyond floating-point summation order of totals.

    Returns
    -------
    DirectDamageArrays
        Quantile arrays at the requested levels, optional total samples and
        metadata.

    Raises
    ------
    ValueError
        If any input is invalid, including non-finite or negative intensity.

    Notes
    -----
    Random draws are keyed rather than sequential. Each stream (``curve``,
    ``system``, ``hazard``, ``damage``, ``cost``) is a Philox4x64
    counter-based generator with key ``(seed, stream id)``. The per-asset
    streams (hazard, damage, cost) start at counter ``(0, 0, asset_key, 0)``.
    The systemic streams (curve choice, systemic cost) start at counter
    ``(0, 0, 0, 1)``, or ``(0, 0, systemic_key, 2)`` when a key is given.
    Each uniform uses one 64-bit output, ``((x >> 12) + 0.5) / 2**52``, in
    (0, 1). Normals are its inverse normal CDF. Curve choice is the inverse
    CDF of the normalised weights. So:

    - an asset's draws depend only on ``(seed, asset key, n_samples)``, not on
      which other assets are in the call, their order, blocks or skipped
      assets; and the first ``m`` samples are the same for any
      ``n_samples >= m``
    - scenarios use common random numbers: an asset gets the same local draws
      for every intensity array, so differences between scenarios reflect the
      hazard rather than sampling noise
    - total samples from separate calls (chunks of assets, asset types,
      layers) with the same seed and ``systemic_key`` share curve choice and
      systemic draws at each sample index, so they can be summed and then
      summarised

    The scheme is recorded as ``metadata["draw_scheme"]``. It differs from
    :func:`calculate_direct_damages`, so the two agree in distribution, not
    sample by sample.

    Damage fractions carry no cost uncertainty. A split's damage fraction
    samples are only formed when needed: with no hazard or damage-noise
    uncertainty, fraction quantiles are computed exactly from the curve
    choice counts.

    If every curve gives zero damage at zero intensity, splits with zero
    intensity in every event have zero damage in every sample. They are left
    out of the computation, and their results, and those of assets with no
    other splits, are exact zeros. Keyed draws mean this does not change any
    other result.
    """
    spec = _validate_simulation(
        intensity,
        asset_index,
        quantity,
        curves=curves,
        curve_weights=curve_weights,
        cost_mean=cost_mean,
        cost_systemic_cv=cost_systemic_cv,
        cost_asset_cv=cost_asset_cv,
        hazard_cv=hazard_cv,
        eta_sd=eta_sd,
        n_samples=n_samples,
        quantiles=quantiles,
        return_periods=return_periods,
        seed=seed,
        asset_keys=asset_keys,
        systemic_key=systemic_key,
        n_assets=n_assets,
        levels=levels,
        keep_total_samples=keep_total_samples,
        block_size=block_size,
    )
    return _run_simulation(spec)


def _run_simulation(
    spec: _Simulation, on_block: Callable[[dict], None] | None = None
) -> DirectDamageArrays:
    """Compute :func:`simulate_direct_damages` from validated inputs.

    ``on_block``, for testing, receives each block's samples, in the
    internal sample order ``sample_order`` (samples sorted by chosen curve).
    """
    n = spec.n_samples
    n_split, n_event = spec.intensity.shape
    n_q = len(spec.quantiles)
    n_curves = len(spec.curves)
    levels = spec.levels
    want_split = "split" in levels
    want_asset = "asset" in levels
    want_total = "total" in levels or spec.keep_total_samples
    want_ead = spec.return_periods is not None
    ead_weights = _ead_weights(spec.return_periods) if want_ead else None
    hazard_uncertain = bool((spec.hazard_cv > 0).any())
    noisy = spec.eta_sd > 0

    # Systemic draws, shared by every asset at each sample index
    if n_curves > 1:
        words = numpy.array([spec.systemic_word], dtype=numpy.uint64)
        u_curve = _uniform_draws(spec.seed, "curve", words, spec.systemic_flag, n)[0]
        cdf = numpy.cumsum(spec.weights)
        cdf[-1] = 1.0
        choice = numpy.minimum(
            numpy.searchsorted(cdf, u_curve, side="right"), n_curves - 1
        )
    else:
        choice = numpy.zeros(n, dtype=numpy.intp)
    # Work in samples sorted by chosen curve, so each curve applies to one slice
    sample_order = numpy.argsort(choice, kind="stable")
    curve_counts = numpy.bincount(choice, minlength=n_curves)
    curve_bounds = numpy.concatenate([[0], numpy.cumsum(curve_counts)])
    if spec.cost_systemic_cv > 0:
        words = numpy.array([spec.systemic_word], dtype=numpy.uint64)
        z_system = _normal_draws(spec.seed, "system", words, spec.systemic_flag, n)[0]
        systemic_multiplier = numpy.maximum(
            0.0, 1.0 + spec.cost_systemic_cv * z_system
        )[sample_order]
    else:
        systemic_multiplier = numpy.ones(n)

    # Splits that need computing: all, unless dry splits have zero damage
    zero_at_zero = all(
        (numpy.asarray(curve.damage_fraction(numpy.zeros(1)), dtype=float) <= 0).all()
        for curve in spec.curves
    )
    if zero_at_zero:
        computed = numpy.flatnonzero((spec.intensity > 0).any(axis=1))
    else:
        computed = numpy.arange(n_split)
    split_order = computed[
        numpy.argsort(spec.asset_index[computed], kind="stable")
    ]  # computed splits, grouped by asset
    split_assets = spec.asset_index[split_order]
    is_first = numpy.ones(len(split_order), dtype=bool)
    is_first[1:] = split_assets[1:] != split_assets[:-1]
    asset_starts = numpy.flatnonzero(is_first)  # into split_order
    assets = split_assets[asset_starts]  # computed assets
    asset_ends = numpy.append(asset_starts[1:], len(split_order))
    block_of_asset = asset_starts // spec.block_size
    block_bounds = numpy.flatnonzero(
        numpy.diff(block_of_asset, prepend=-1, append=numpy.inf)
    )  # into assets

    split_fraction = numpy.zeros((n_split, n_event, n_q)) if want_split else None
    split_cost = numpy.zeros((n_split, n_event, n_q)) if want_split else None
    split_ead = numpy.zeros((n_split, n_q)) if want_split and want_ead else None
    asset_cost = numpy.zeros((spec.n_assets, n_event, n_q)) if want_asset else None
    asset_ead = numpy.zeros((spec.n_assets, n_q)) if want_asset and want_ead else None
    totals = numpy.zeros((n_event, n)) if want_total else None
    quantiles = spec.quantiles

    for b0, b1 in pairwise(block_bounds):
        s0, s1 = asset_starts[b0], asset_ends[b1 - 1]
        splits = split_order[s0:s1]  # (b,) original split indices
        block_assets = assets[b0:b1]  # (a_b,)
        local_starts = asset_starts[b0:b1] - s0
        local_asset = numpy.repeat(
            numpy.arange(b1 - b0), asset_ends[b0:b1] - asset_starts[b0:b1]
        )  # (b,) position of each split's asset in the block
        keys = spec.asset_keys[block_assets]
        mean_intensity = spec.intensity[splits]  # (b, e)

        # Damage fraction samples, (b, e, n)
        fraction = numpy.empty((len(splits), n_event, n))
        if hazard_uncertain:
            z_hazard = _normal_draws(spec.seed, "hazard", keys, _COUNTER_LOCAL, n)
            z_hazard = z_hazard[:, sample_order][local_asset]  # (b, n)
            hazard = mean_intensity[:, :, None] * (
                1.0 + spec.hazard_cv[None, :, None] * z_hazard[:, None, :]
            )
            numpy.maximum(0.0, hazard, out=hazard)
            for k, curve in enumerate(spec.curves):
                part = slice(curve_bounds[k], curve_bounds[k + 1])
                if curve_counts[k]:
                    fraction[..., part] = curve.damage_fraction(hazard[..., part])
            del hazard
            curve_values = None
        else:
            # Hazard is the same in every sample: evaluate each curve once
            curve_values = numpy.stack(
                [
                    numpy.asarray(curve.damage_fraction(mean_intensity), dtype=float)
                    for curve in spec.curves
                ],
                axis=-1,
            )  # (b, e, K)
            if not noisy:
                # clip once per curve value rather than per sample
                curve_values = numpy.clip(curve_values, 0.0, 1.0)
            for k in range(n_curves):
                part = slice(curve_bounds[k], curve_bounds[k + 1])
                fraction[..., part] = curve_values[..., k, None]
        if noisy:
            z_damage = _normal_draws(spec.seed, "damage", keys, _COUNTER_LOCAL, n)
            damage_multiplier = numpy.exp(
                spec.eta_sd * z_damage - 0.5 * spec.eta_sd**2
            )[:, sample_order]
            fraction *= damage_multiplier[local_asset][:, None, :]
        if hazard_uncertain or noisy:
            numpy.clip(fraction, 0.0, 1.0, out=fraction)

        wet = (mean_intensity > 0) if zero_at_zero else None  # (b, e)
        if want_split:
            if curve_values is not None and not noisy:
                split_fraction[splits] = _discrete_quantiles(
                    curve_values, curve_counts, quantiles
                )
            else:
                split_fraction[splits] = _summarise_rows(fraction, quantiles, wet)
        # Cost samples reuse the fraction array, unless a test inspects both
        cost_samples = fraction if on_block is None else fraction.copy()

        # Unit cost per asset, (a_b, n)
        unit_cost = spec.cost_mean[block_assets][:, None] * systemic_multiplier
        if spec.cost_asset_cv > 0:
            z_cost = _normal_draws(spec.seed, "cost", keys, _COUNTER_LOCAL, n)
            unit_cost = unit_cost * (1.0 + spec.cost_asset_cv * z_cost[:, sample_order])
        numpy.maximum(0.0, unit_cost, out=unit_cost)

        # Split cost samples, (b, e, n)
        cost_samples *= spec.quantity[splits][:, None, None]
        cost_samples *= unit_cost[local_asset][:, None, :]
        asset_samples = numpy.add.reduceat(cost_samples, local_starts, axis=0)
        if on_block is not None:
            on_block(
                {
                    "split_index": splits,
                    "asset_index": block_assets,
                    "sample_order": sample_order,
                    "fraction": fraction,
                    "split_cost": cost_samples.copy(),
                    "asset_cost": asset_samples.copy(),
                }
            )
        if want_total:
            totals += asset_samples.sum(axis=0)
        if want_asset:
            asset_wet = None
            if wet is not None:
                asset_wet = numpy.logical_or.reduceat(wet, local_starts, axis=0)
            if want_ead:
                asset_ead[block_assets] = _summarise(
                    _event_axis_ead(asset_samples, ead_weights),
                    quantiles,
                    overwrite=True,
                )
            asset_cost[block_assets] = _summarise_rows(
                asset_samples, quantiles, asset_wet, overwrite=True
            )
        del asset_samples
        if want_split:
            if want_ead:
                split_ead[splits] = _summarise(
                    _event_axis_ead(cost_samples, ead_weights),
                    quantiles,
                    overwrite=True,
                )
            split_cost[splits] = _summarise_rows(
                cost_samples, quantiles, wet, overwrite=True
            )
        del fraction, cost_samples

    total_samples = total_cost = total_ead = None
    if want_total:
        total_samples = numpy.empty((n, n_event))
        total_samples[sample_order] = totals.T
        if "total" in levels:
            total_cost = _summarise(totals, quantiles, overwrite=True)
            if want_ead:
                total_ead = _summarise(
                    expected_annual_damage(total_samples, spec.return_periods),
                    quantiles,
                )
        if not spec.keep_total_samples:
            total_samples = None

    metadata = {
        "draw_scheme": DRAW_SCHEME,
        "seed": spec.seed,
        "systemic_key": spec.systemic_key,
        "n_samples": n,
        "quantiles": spec.quantiles.tolist(),
        "return_periods": (
            spec.return_periods.tolist() if spec.return_periods is not None else None
        ),
        "levels": [level for level in _LEVELS if level in levels],
        "keep_total_samples": spec.keep_total_samples,
        "n_splits": n_split,
        "n_assets": spec.n_assets,
        "n_events": n_event,
        "n_computed_splits": len(split_order),
        "n_computed_assets": len(assets),
        "skip_dry": zero_at_zero,
        "block_size": spec.block_size,
        "curve_weights": spec.weights.tolist(),
        "cost_mean": spec.cost_mean_scalar
        if spec.cost_mean_scalar is not None
        else "per_asset",
        "cost_systemic_cv": spec.cost_systemic_cv,
        "cost_asset_cv": spec.cost_asset_cv,
        "hazard_cv": spec.hazard_cv.tolist(),
        "eta_sd": spec.eta_sd,
        "draw_scopes": {
            "damage_curve": "seed_systemic_key",
            "cost_system": "seed_systemic_key",
            "hazard": "seed_asset_key",
            "damage_noise": "seed_asset_key",
            "cost": "seed_asset_key",
        },
    }
    return DirectDamageArrays(
        quantiles=spec.quantiles.copy(),
        return_periods=(
            spec.return_periods.copy() if spec.return_periods is not None else None
        ),
        split_fraction=split_fraction,
        split_cost=split_cost,
        split_ead=split_ead,
        asset_cost=asset_cost,
        asset_ead=asset_ead,
        total_cost=total_cost,
        total_ead=total_ead,
        total_samples=total_samples,
        metadata=metadata,
    )


# TODO check `nismod/east-africa-transport` and `nismod/jamaica-infrastructure`
# manipulations of damage curves


class DamageCurve(ABC):
    """A damage curve"""

    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def damage_fraction(exposure: numpy.array) -> numpy.array:
        """Evaluate damage fraction for exposure to a given hazard intensity"""


class PiecewiseLinearDamageCurveSchema(pandera.DataFrameModel):
    intensity: Series[float]
    damage: Series[float]


class PiecewiseLinearDamageCurve(DamageCurve):
    """A piecewise-linear damage curve"""

    intensity: Series[float]
    damage: Series[float]

    def __init__(self, curve: DataFrame[PiecewiseLinearDamageCurveSchema]):
        curve = curve.copy()
        self.intensity, self.damage = self.clip_curve_data(
            curve.intensity, curve.damage
        )

        bounds = (self.damage.min(), self.damage.max())
        self._interpolate = interp1d(
            self.intensity,
            self.damage,
            kind="linear",
            fill_value=bounds,
            bounds_error=False,
            copy=False,
        )

    def __eq__(self, other):
        damage_eq = (self.damage == other.damage).all()
        intensity_eq = (self.intensity == other.intensity).all()
        return damage_eq and intensity_eq

    @classmethod
    def from_csv(
        cls,
        fname,
        intensity_col="intensity",
        damage_col="damage_ratio",
        comment="#",
        **kwargs,
    ):
        """Read a damage curve from a CSV file.

        By default, the CSV should have columns named "intensity" and "damage_ratio",
        with any additional header lines commented out by "#".

        Any additional keyword arguments are passed through to ``pandas.read_csv``

        Parameters
        ----------
        fname : str, path object or file-like object
        intensity_col : str, default "intensity"
            Column name to read hazard intensity values
        damage_col : str, default "damage_ratio"
            Column name to read damage values
        comment : str, default "#"
            Indicates remainder of the line in the CSV should not be parsed.
            If found at the beginning of a line, the line will be ignored
            altogether.
        kwargs
            see pandas.read_csv documentation

        Returns
        -------
        PiecewiseLinearDamageCurve
        """
        curve_data = pandas.read_csv(fname, comment=comment, **kwargs).rename(
            columns={
                intensity_col: "intensity",
                damage_col: "damage",
            }
        )
        return cls(curve_data)

    @classmethod
    def from_excel(
        cls,
        fname,
        sheet_name=0,
        intensity_col="intensity",
        damage_col="damage_ratio",
        comment="#",
        **kwargs,
    ):
        """Read a damage curve from an Excel file.

        By default, the file should have columns named "intensity" and "damage_ratio",
        with any additional header lines commented out by "#".

        Any additional keyword arguments are passed through to ``pandas.read_excel``

        Parameters
        ----------
        fname : str, path object or file-like object
        sheet_name : str or int
            Strings are used for sheet names. Integers are used in zero-indexed sheet
            positions (chart sheets do not count as a sheet position).
        intensity_col : str, default "intensity"
            Column name to read hazard intensity values
        damage_col : str, default "damage_ratio"
            Column name to read damage values
        comment : str, default "#"
            Indicates remainder of the line in the CSV should not be parsed.
            If found at the beginning of a line, the line will be ignored
            altogether.
        kwargs
            see pandas.read_csv documentation

        Returns
        -------
        PiecewiseLinearDamageCurve
        """
        curve_data = pandas.read_excel(
            fname, sheet_name=sheet_name, comment=comment, **kwargs
        ).rename(
            columns={
                intensity_col: "intensity",
                damage_col: "damage",
            }
        )
        return cls(curve_data)

    @classmethod
    def interpolate(
        cls,
        a,
        b,
        factor: float,
    ):
        """Interpolate damage values between two curves

        ```
        new_curve_damage = a_damage + ((b_damage - a_damage) * factor)
        ```

        Parameters
        ----------
        a : PiecewiseLinearDamageCurve
        b : PiecewiseLinearDamageCurve
        factor : float
            Interpolation factor, used to calculate the new curve

        Returns
        -------
        PiecewiseLinearDamageCurve
        """
        # find the sorted, unique values of intensity used by both curves
        intensity = numpy.unique(numpy.concatenate((a.intensity, b.intensity)))
        # calculate the damage fraction at each intensity
        a_damage = a.damage_fraction(intensity)
        b_damage = b.damage_fraction(intensity)
        # interpolate at each point
        damage = a_damage + ((b_damage - a_damage) * factor)
        return cls(
            pandas.DataFrame(
                {
                    "intensity": intensity,
                    "damage": damage,
                }
            )
        )

    def damage_fraction(self, exposure: numpy.array) -> numpy.array:
        """Evaluate damage fraction for exposure to a given hazard intensity"""
        return self._interpolate(exposure)

    def translate_y(self, y: float):
        """Translate damage by a factor, y"""
        damage = self.damage + y

        return PiecewiseLinearDamageCurve(
            pandas.DataFrame(
                {
                    "intensity": self.intensity,
                    "damage": damage,
                }
            )
        )

    def scale_y(self, y: float):
        """Scale damage by a factor, y"""
        damage = self.damage * y

        return PiecewiseLinearDamageCurve(
            pandas.DataFrame(
                {
                    "intensity": self.intensity,
                    "damage": damage,
                }
            )
        )

    def translate_x(self, x: float):
        """Translate intensity by a factor, x"""
        intensity = self.intensity + x

        return PiecewiseLinearDamageCurve(
            pandas.DataFrame(
                {
                    "intensity": intensity,
                    "damage": self.damage,
                }
            )
        )

    def scale_x(self, x: float):
        """Scale intensity by a factor, x"""
        intensity = self.intensity * x

        return PiecewiseLinearDamageCurve(
            pandas.DataFrame(
                {
                    "intensity": intensity,
                    "damage": self.damage,
                }
            )
        )

    @staticmethod
    def clip_curve_data(intensity, damage):
        """Clip damage curve values to valid 0-1 damage range"""
        if (damage.max() > 1) or (damage.min() < 0):
            # WARNING clipping out-of-bounds damage fractions
            bounds = (
                intensity.min(),
                intensity.max(),
            )
            inverse = interp1d(
                damage,
                intensity,
                kind="linear",
                fill_value=bounds,
                bounds_error=False,
            )

        if damage.max() > 1:
            one_intercept = inverse(1)
            idx = numpy.searchsorted(intensity, one_intercept)
            # if one_intercept is in our intensities
            if intensity[idx] == one_intercept:
                # no action - damage fraction will be set to 1
                pass
            else:
                # else insert new interpolation point
                damage = numpy.insert(damage, idx, 1)
                intensity = numpy.insert(intensity, idx, one_intercept)

        if damage.min() < 0:
            zero_intercept = inverse(0)
            idx = numpy.searchsorted(intensity, zero_intercept)
            # if zero_intercept is in our intensities
            if intensity[idx] == zero_intercept:
                # no action - damage fraction will be set to 0
                pass
            else:
                # else insert new interpolation point
                damage = numpy.insert(damage, idx, 0)
                intensity = numpy.insert(intensity, idx, zero_intercept)

        damage = numpy.clip(damage, 0, 1)

        return intensity, damage

    def plot(self, ax=None):
        """Plot a line chart of the damage curve"""
        import matplotlib.pyplot as plt

        if ax is None:
            fig = plt.figure()
            ax = fig.add_subplot(1, 1, 1)

        ax.plot(self.intensity, self.damage, color="tab:blue")
        ax.set_ylim([0, 1])
        ax.set_ylabel("Damage Fraction")
        ax.set_xlabel("Hazard Intensity")

        return ax
