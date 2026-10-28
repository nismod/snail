"""Damage assessment"""

from abc import ABC
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy
import pandas
import pandera.pandas as pandera
from pandera.typing import DataFrame, Series
from scipy.interpolate import interp1d


@dataclass(frozen=True)
class DirectDamageResult:
    """Summarised sampled direct damages and their reproducibility metadata.

    ``metadata`` records the seed and model inputs. ``quantiles`` is a long-form
    frame with one row per asset, scenario and intensity case, when requested.
    ``ead_quantiles`` contains one row per asset and scenario when requested.
    Quantile columns are named ``q_05``, ``q_50``, etc. Raw samples are not kept.
    """

    metadata: dict[str, Any]
    quantiles: pandas.DataFrame | None
    ead_quantiles: pandas.DataFrame | None = None


def _validate_quantiles(
    quantiles: Sequence[float] | None, name: str
) -> tuple[tuple[float, ...] | None, list[str] | None]:
    if quantiles is None:
        return None, None
    quantiles = tuple(quantiles)
    if len(quantiles) == 0 or any(
        not numpy.isfinite(q) or not 0 <= q <= 1 for q in quantiles
    ):
        raise ValueError(f"{name} must be a non-empty sequence in [0, 1]")
    if len(set(quantiles)) != len(quantiles):
        raise ValueError(f"{name} must not contain duplicates")
    columns = [f"q_{round(q * 100):02d}" for q in quantiles]
    if len(set(columns)) != len(columns):
        raise ValueError(f"{name} must map to distinct percentile column names")
    return quantiles, columns


def _integrate_ead_samples(
    sampled_damage_by_case: Mapping[Any, numpy.ndarray],
    return_periods: Mapping[Any, float],
) -> numpy.ndarray:
    """
    Integrate sampled damages over annual exceedance probability
    """
    cases = sorted(return_periods, key=lambda case: 1.0 / return_periods[case])
    probabilities = numpy.asarray([1.0 / return_periods[case] for case in cases])
    damage = numpy.column_stack([sampled_damage_by_case[case] for case in cases])

    # Set unlikely tail: repeat first provided damage as zero-probability damage
    probabilities = numpy.concatenate(([0.0], probabilities))
    damage = numpy.column_stack((damage[:, 0], damage))

    # No special treatment of most likely end, so if probability 1 is not
    # supplied, we truncate contribution from (unknown) most likely events.

    interval_areas = (
        0.5 * (damage[:, :-1] + damage[:, 1:]) * numpy.diff(probabilities)[None, :]
    )
    return interval_areas.sum(axis=1)


def calculate_direct_damages(
    splits: pandas.DataFrame,
    *,
    asset_id: str,
    length: str,
    intensity_columns: Mapping[Any, str],
    hazard_alpha: float | Mapping[Any, float] = 0.0,
    curves: Mapping[str, "DamageCurve"],
    curve_weights: Mapping[str, float],
    eta_sd: float = 0.0,
    cost_mean: float | str,
    cost_alpha: float = 0.0,
    cost_system_sd: float = 0.0,
    n_samples: int = 1000,
    quantiles: Sequence[float] | None = None,
    return_periods: Mapping[Any, float] | None = None,
    ead_quantiles: Sequence[float] | None = None,
    seed: int | None = None,
    scenario: str | None = None,
) -> DirectDamageResult:
    """Sample and summarise direct damages for pre-split assets.

    Each row in ``splits`` is one asset segment. ``intensity_columns`` maps
    intensity-case labels (such as named events or return-periods) to hazard
    intensity columns. Each simulation selects a weighted damage curve and
    draws a systemic cost offset shared across all assets and scenarios.
    Each asset/scenario group then receives independent hazard, damage-noise
    and local cost draws, shared across its splits and intensity cases.
    Hazard and cost values are clamped at zero, damage fractions are clipped to [0, 1],
    and asset splits damages are summed for each intensity case. The result
    returns requested quantiles and run metadata. Samples are discarded.

    In the equations below, ``Z`` denotes a draw from a standard normal
    distribution with mean 0 and standard deviation 1.

    Parameters
    ----------
    splits : pandas.DataFrame
        Pre-split asset data, with one row per segment. Each segment's length
        and hazard intensities are used to calculate and aggregate its damage.
    asset_id : str
        Name of the column identifying the parent asset for each segment.
    length : str
        Name of the column giving each segment's length, in units compatible
        with the rehabilitation cost.
    intensity_columns : Mapping[Any, str]
        Mapping from an intensity-case label to the column containing its
        central hazard intensity for each segment. Labels identify the cases
        in the returned summaries and may represent any kind of event or
        return-period maps, for example.
    hazard_alpha : float or Mapping[Any, float], default 0.0
        Relative standard deviation of hazard intensity. Supply one value for
        all cases or a mapping from each intensity-case label to its value.
        A case's sampled intensity is ``max(0, mean * (1 + alpha * Z))``.
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
        Baseline rehabilitation cost per unit length. Give a non-negative number
        to use the same mean for all assets, or a column name to read an
        asset-specific mean. Column values must be constant across that
        asset's segments within each scenario.
    cost_alpha : float, default 0.0
        Relative standard deviation of the local normal cost perturbation.
        Sampled unit cost is ``max(0, systemic_cost * (1 + cost_alpha * Z))``,
        with an independent local draw for each asset/scenario group.
        Zero disables local cost variation.
    cost_system_sd : float, default 0.0
        Finite, non-negative standard deviation of the systemic normal cost
        offset, in the same cost-per-unit-length units as ``cost_mean``.
        Each simulation uses ``systemic_cost = max(0, cost_mean +
        cost_system_sd * Z_system)``. The same absolute offset is applied to
        every asset/scenario baseline. Zero disables systemic cost variation.
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

    Returns
    -------
    DirectDamageResult
        ``quantiles`` contains one row per asset, scenario (if supplied), and
        intensity case when requested. ``ead_quantiles`` contains one row per
        asset/scenario group when requested. Both have ``n_splits`` and one
        column per requested quantile. ``metadata`` records the seed and model
        configuration.

    Raises
    ------
    KeyError
        If a required input column is missing.
    ValueError
        If parameters are invalid, numeric inputs are missing or non-finite,
        or a column-based cost mean varies within an asset/scenario group.

    Notes
    -----
    Curve choices and systemic cost offsets are shared across the entire call
    at each simulation index. Hazard, damage noise and local cost draws are
    independent between asset/scenario groups and shared across all splits
    and intensity cases within each group. Both cost stages are clamped at
    zero; the final cost mean may therefore differ from the input baseline.

    Global damage curve choices are made first, then systemic cost draws, then
    local draws for each group. Metadata records cost_system_sd and draw_scopes
    ("call" or "asset_scenario").

    If EAD is requested, each sample's damage values are integrated over annual
    exceedance probability using the trapezoid rule, with the lowest probability
    damage repeated as the zero probability damage. There is no extrapolation at
    the highest probability end.
    """
    required = {asset_id, length, *intensity_columns.values()}
    if scenario is not None:
        required.add(scenario)
    if isinstance(cost_mean, str):
        required.add(cost_mean)
    missing = required - set(splits.columns)
    if missing:
        raise KeyError(f"Required columns not present in splits: {sorted(missing)}")
    if not intensity_columns:
        raise ValueError("intensity_columns must contain at least one intensity case")
    if not isinstance(n_samples, (int, numpy.integer)) or n_samples <= 0:
        raise ValueError("n_samples must be greater than zero")

    if quantiles is None and ead_quantiles is None:
        raise ValueError("request quantiles, ead_quantiles, or both")
    quantiles, quantile_columns = _validate_quantiles(quantiles, "quantiles")
    ead_quantiles, ead_quantile_columns = _validate_quantiles(
        ead_quantiles, "ead_quantiles"
    )
    if (ead_quantiles is None) != (return_periods is None):
        raise ValueError("return_periods and ead_quantiles must be supplied together")

    return_period_values: dict[Any, float] | None = None
    if return_periods is not None:
        if not isinstance(return_periods, Mapping):
            raise ValueError("return_periods must map intensity-case labels to years")
        if set(return_periods) != set(intensity_columns):
            raise ValueError("return_periods keys must match intensity_columns")
        try:
            return_period_values = {
                case: float(period) for case, period in return_periods.items()
            }
        except (TypeError, ValueError) as error:
            raise ValueError(
                "return periods must be finite numbers of at least one year"
            ) from error
        periods = numpy.asarray(list(return_period_values.values()), dtype=float)
        if (
            not numpy.isfinite(periods).all()
            or (periods < 1.0).any()
            or len(set(periods)) != len(periods)
        ):
            raise ValueError(
                "return periods must be finite, unique, and at least one year"
            )

    if (
        not numpy.isfinite(eta_sd)
        or not numpy.isfinite(cost_alpha)
        or eta_sd < 0
        or cost_alpha < 0
    ):
        raise ValueError("eta_sd and cost_alpha must be non-negative")
    if not numpy.isfinite(cost_system_sd) or cost_system_sd < 0:
        raise ValueError("cost_system_sd must be finite and non-negative")
    if isinstance(hazard_alpha, Mapping):
        missing_alpha = set(intensity_columns) - set(hazard_alpha)
        if missing_alpha:
            raise ValueError(
                f"hazard_alpha missing intensity cases: {sorted(missing_alpha)}"
            )
        alpha_by_case = {case: float(hazard_alpha[case]) for case in intensity_columns}
    else:
        alpha_by_case = {case: float(hazard_alpha) for case in intensity_columns}
    if any(not numpy.isfinite(alpha) or alpha < 0 for alpha in alpha_by_case.values()):
        raise ValueError("hazard_alpha values must be non-negative")

    if not curves or set(curves) != set(curve_weights):
        raise ValueError("curves and curve_weights must have the same non-empty keys")
    weights = numpy.asarray([curve_weights[key] for key in curves], dtype=float)
    if not numpy.isfinite(weights).all() or (weights < 0).any() or weights.sum() <= 0:
        raise ValueError("curve_weights must be finite, non-negative, and sum > 0")
    weights /= weights.sum()
    curve_ids = list(curves)
    curve_values = [curves[key] for key in curve_ids]

    if not isinstance(cost_mean, str) and (
        not numpy.isfinite(cost_mean) or cost_mean < 0
    ):
        raise ValueError("cost_mean must be finite and non-negative")
    numeric_columns = list(dict.fromkeys([length, *intensity_columns.values()]))
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
    systemic_cost_offset = cost_system_sd * z_system  # (n,)

    rows: list[dict[str, Any]] = []
    ead_rows: list[dict[str, Any]] = []
    grouper: str | list[str] = group_cols[0] if len(group_cols) == 1 else group_cols
    # Scenario grouping
    for key, group in working.groupby(grouper, sort=False, dropna=False):
        key_values = (key,) if len(group_cols) == 1 else key
        base = dict(zip(group_cols, key_values))
        if isinstance(cost_mean, str):
            costs = group[cost_mean].to_numpy(dtype=float)
            if not numpy.allclose(costs, costs[0]):
                raise ValueError(f"Cost mean must be constant within group {base!r}")
            group_cost_mean = float(costs[0])
        else:
            group_cost_mean = float(cost_mean)

        # Make local draws, shared across asset-scenario
        size = len(group)  # s
        z_hazard = rng.standard_normal(n_samples)  # (n,)
        z_damage = rng.standard_normal(n_samples)  # (n,)
        z_cost = rng.standard_normal(n_samples)  # (n,)
        systemic_cost = numpy.maximum(
            0.0, group_cost_mean + systemic_cost_offset
        )  # (n,)
        cost_draw = numpy.maximum(
            0.0, systemic_cost * (1.0 + cost_alpha * z_cost)
        )  # (n,)
        damage_multiplier = numpy.exp(eta_sd * z_damage - 0.5 * eta_sd**2)  # (n,)
        lengths = group[length].to_numpy(dtype=float)  # (s,)
        sampled_totals: dict[Any, numpy.ndarray] = {}  # each value: (n,)

        # Intensity case (e.g. return-period) iterator
        for case, column in intensity_columns.items():
            means = group[column].to_numpy(dtype=float)  # (s,)
            hazard = numpy.maximum(
                0.0,
                means[None, :] * (1.0 + alpha_by_case[case] * z_hazard[:, None]),
            )  # (n, s)
            damage_fraction = numpy.empty((n_samples, size), dtype=float)  # (n, s)
            for curve_index, curve in enumerate(curve_values):
                mask = selected_curve == curve_index  # (n,)
                if mask.any():
                    damage_fraction[mask] = curve.damage_fraction(hazard[mask])
            damage_fraction = numpy.clip(
                damage_fraction * damage_multiplier[:, None], 0.0, 1.0
            )  # (n, s)
            total = (damage_fraction * lengths[None, :] * cost_draw[:, None]).sum(
                axis=1
            )  # (n,)

            if quantiles is not None:
                summary = dict(base)
                summary["intensity_case"] = case
                summary["n_splits"] = size
                for q, column_name in zip(quantiles, quantile_columns):
                    summary[column_name] = float(numpy.quantile(total, q))
                rows.append(summary)
            if ead_quantiles is not None:
                sampled_totals[case] = total

        if ead_quantiles is not None:
            ead_samples = _integrate_ead_samples(
                sampled_totals, return_period_values
            )  # (n,)
            summary = dict(base)
            summary["n_splits"] = size
            for q, column_name in zip(ead_quantiles, ead_quantile_columns):
                summary[column_name] = float(numpy.quantile(ead_samples, q))
            ead_rows.append(summary)

    metadata = {
        "seed": seed,
        "n_samples": n_samples,
        "quantiles": list(quantiles) if quantiles is not None else None,
        "return_periods": return_period_values,
        "ead_quantiles": (list(ead_quantiles) if ead_quantiles is not None else None),
        "asset_id_column": asset_id,
        "scenario_column": scenario,
        "length_column": length,
        "intensity_columns": dict(intensity_columns),
        "curve_ids": curve_ids,
        "curve_weights": dict(curve_weights),
        "cost_mean": cost_mean,
        "cost_alpha": cost_alpha,
        "cost_system_sd": cost_system_sd,
        "hazard_alpha": alpha_by_case,
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

    return DirectDamageResult(
        quantiles=quantile_frame,
        metadata=metadata,
        ead_quantiles=ead_quantile_frame,
    )


# TODO check `nismod/east-africa-transport` and `nismod/jamaica-infrastructure`
# manipulations of damage curves


class DamageCurve(ABC):
    """A damage curve"""

    def __init__(self):
        pass

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
