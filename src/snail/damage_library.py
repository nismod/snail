"""Utilities to access packaged damage curve data.

The library distributes a curated subset of the
`Nirandjan et al. (2023) Physical Vulnerability Database`_. The upstream
work is licensed under CC-BY 4.0; see ``snail/data/damage_curves`` for the
packaged metadata and curve samples.

The helper functions defined here make it easy to enumerate available
curves, filter by hazard or infrastructure attributes, and instantiate
``PiecewiseLinearDamageCurve`` objects ready for use in damage modelling.

.. _Nirandjan et al. (2023) Physical Vulnerability Database:
   https://doi.org/10.5281/zenodo.10203846
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from importlib import resources

import pandas as pd

_PACKAGE = "snail.data.damage_curves"
_METADATA_FILE = "metadata.csv"
_CURVES_FILE = "curves.csv"
_METADATA_COLUMNS: Sequence[str] = [
    "curve_id",
    "hazard_type",
    "hazard_name",
    "sector",
    "sheet_name",
    "intensity_metric",
    "intensity_axis",
    "intensity_unit",
    "exposed_element",
    "additional_characteristics",
    "curve_type",
    "curve_characteristics",
    "damage_states",
    "derivation_methodology",
    "cost_feature",
    "uncertainty_range",
    "geographical_application",
    "readily_available",
    "source",
    "source_details",
    "original_id",
]


@dataclass(frozen=True)
class DamageCurveMetadata:
    """Metadata describing a packaged damage curve."""

    curve_id: str
    hazard_type: str
    hazard_name: str
    sector: str
    sheet_name: str
    intensity_metric: str
    intensity_axis: str
    intensity_unit: str | None
    exposed_element: str
    additional_characteristics: str | None
    curve_type: str | None
    curve_characteristics: str | None
    damage_states: str | None
    cost_feature: str | None
    uncertainty_range: str | None
    derivation_methodology: str | None
    geographical_application: str | None
    readily_available: str | None
    source: str
    source_details: str | None
    original_id: str | None


def _load_metadata() -> pd.DataFrame:
    with resources.as_file(resources.files(_PACKAGE) / _METADATA_FILE) as path:
        metadata = pd.read_csv(path)
    metadata = metadata[_METADATA_COLUMNS].copy()
    metadata = metadata.where(metadata.notna(), None)
    return metadata


def _load_curves() -> pd.DataFrame:
    with resources.as_file(resources.files(_PACKAGE) / _CURVES_FILE) as path:
        curves = pd.read_csv(path)
    return curves


_METADATA_CACHE: pd.DataFrame | None = None
_CURVES_CACHE: pd.DataFrame | None = None


def available_curves(
    hazard: str | None = None,
    sector: str | None = None,
    exposed_element: str | None = None,
    curve_type: str | None = None,
) -> pd.DataFrame:
    """Return metadata for available curves, with optional filters."""
    global _METADATA_CACHE
    if _METADATA_CACHE is None:
        _METADATA_CACHE = _load_metadata()

    frame = _METADATA_CACHE
    filters = {
        "hazard_name": hazard,
        "sector": sector,
        "exposed_element": exposed_element,
        "curve_type": curve_type,
    }
    mask = pd.Series(True, index=frame.index, dtype=bool)
    for column, value in filters.items():
        if value is None:
            continue
        if isinstance(value, str):
            mask &= frame[column].str.contains(value, case=False, na=False)
        else:
            values = (
                value
                if isinstance(value, Iterable) and not isinstance(value, str)
                else (value,)
            )
            mask &= frame[column].isin(values)
    return frame.loc[mask].copy()


def get_metadata(curve_id: str) -> DamageCurveMetadata:
    """Return the metadata record for a specific curve."""
    global _METADATA_CACHE
    if _METADATA_CACHE is None:
        _METADATA_CACHE = _load_metadata()

    row = _METADATA_CACHE.loc[_METADATA_CACHE["curve_id"] == curve_id]
    if row.empty:
        raise KeyError(f"Curve '{curve_id}' is not packaged with snail.")
    record = {
        key: (None if pd.isna(value) else value) for key, value in row.iloc[0].items()
    }
    return DamageCurveMetadata(**record)  # type: ignore[arg-type]


def load_curve(curve_id: str):
    """Instantiate a PiecewiseLinearDamageCurve from the packaged dataset."""
    global _CURVES_CACHE
    if _CURVES_CACHE is None:
        _CURVES_CACHE = _load_curves()

    frame = _CURVES_CACHE
    selection = frame.loc[frame["curve_id"] == curve_id].sort_values("point_index")
    if selection.empty:
        raise KeyError(f"Curve '{curve_id}' is not packaged with snail.")

    curve_df = pd.DataFrame(
        {
            "intensity": selection["intensity"].to_numpy(),
            "damage": selection["damage"].to_numpy(),
        }
    )

    from .damages import PiecewiseLinearDamageCurve

    return PiecewiseLinearDamageCurve(curve_df)
