#!/usr/bin/env python
"""Build the packaged damage curve library from the Nirandjan et al. dataset.

The upstream dataset is documented in:
    Nirandjan S. et al. (2023) Physical Vulnerability Database for
    Critical Infrastructure Multi-Hazard Risk Assessments.

The raw Excel workbooks are stored under ``nirandjan-2023-vulnerability-database``.
This script parses the metadata (Table D1) and the vulnerability curves (Table D2)
to generate lightweight CSV artefacts that can be shipped inside the Python wheel:

* metadata.csv -- curve level metadata and descriptive fields
* curves.csv -- intensity / expected damage ratios for each published curve

Both outputs are written to ``src/snail/data/damage_curves``.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import pandas as pd


RAW_DATA_DIR = Path("nirandjan-2023-vulnerability-database")
TABLE_D1 = RAW_DATA_DIR / "Table_D1_Summary_CI_Vulnerability_Data_V1.0.0.xlsx"
TABLE_D2 = (
    RAW_DATA_DIR
    / "Table_D2_Multi-Hazard_Fragility_and_Vulnerability_Curves_V1.0.0.xlsx"
)
OUTPUT_DIR = Path("src/snail/data/damage_curves")

# Sheets in Table D1 / D2 to ignore
IGNORED_SHEETS_D1 = {"General_Info", "Reference_List"}
IGNORED_SHEETS_D2 = {"General_Info"}
VULNERABILITY_SHEET_KEYWORD = "Vuln"

HAZARD_CODE_MAP = {
    "F": "flood",
    "E": "earthquake",
    "W": "wind",
    "L": "landslide",
}
EXCLUDED_CURVE_TYPES = {"V (repair rate)", "V (faults/km)"}


def normalise_header(row: pd.Series, original: Iterable[str]) -> List[str]:
    """Fill unnamed Excel columns using header hints from the first row."""
    headers: List[str] = []
    for idx, col in enumerate(original):
        if isinstance(col, str) and not col.startswith("Unnamed"):
            headers.append(col.strip())
            continue

        candidate = row.iloc[idx]
        if isinstance(candidate, str) and candidate.strip():
            headers.append(candidate.strip())
            continue

        # Fallback for completely blank cells
        headers.append(f"column_{idx}")
    return headers


def parse_table_d1(table_path: Path) -> pd.DataFrame:
    """Parse metadata from Table D1."""
    metadata_tables: List[pd.DataFrame] = []
    workbook = pd.ExcelFile(table_path)

    for sheet in workbook.sheet_names:
        if sheet in IGNORED_SHEETS_D1:
            continue

        raw = workbook.parse(sheet_name=sheet, header=0)
        raw.columns = normalise_header(raw.iloc[0], raw.columns)
        data = raw.iloc[1:].copy()
        data = data[data["ID number"].notna()]
        if data.empty:
            continue

        data["ID number"] = data["ID number"].astype(str).str.strip()
        data["sector"] = sheet
        metadata_tables.append(data)

    if not metadata_tables:
        raise RuntimeError("No metadata rows were parsed from Table D1.")

    metadata = pd.concat(metadata_tables, ignore_index=True)

    column_map = {
        "column_0": "source",
        "Hazard": "hazard_type",
        "Intensity metric": "intensity_metric",
        "Infrastructure description": "infrastructure_description",
        "Additional characteristics\u00a0": "additional_characteristics",
        "Additional characteristics": "additional_characteristics",
        "Fragility and/or vulnerability": "curve_type",
        "Vulnerability details": "curve_type",
        "Characteristics of curve": "curve_characteristics",
        "Damage states (in case of fragility)": "damage_states",
        "Cost feature": "cost_feature",
        "Uncertainty range": "uncertainty_range",
        "Derivation methodology\u00a0": "derivation_methodology",
        "Derivation methodology": "derivation_methodology",
        "Derivation method": "derivation_methodology",
        "Geographical application": "geographical_application",
        "Source type\u00a0": "source_type",
        "Readily available": "readily_available",
        "ID number": "curve_id",
        "Orignal ID number": "original_id",
        "Source details": "source_details",
        "Exposed element": "exposed_element",
    }

    metadata = metadata.rename(columns=column_map)

    # Collapse duplicate columns created by near-identical headers
    metadata = metadata.T.groupby(level=0).first().T

    missing_columns = {"curve_id", "hazard_type"} - set(metadata.columns)
    if missing_columns:
        raise KeyError(f"Expected metadata columns missing: {missing_columns}")

    metadata["hazard_type"] = metadata["hazard_type"].astype(str).str.strip()
    metadata["hazard_name"] = (
        metadata["hazard_type"].map(HAZARD_CODE_MAP).fillna(metadata["hazard_type"])
    )

    return metadata


def parse_intensity_label(label: str) -> Tuple[str, str | None]:
    """Parse an intensity label like 'Depth (m)' into (name, unit)."""
    if not isinstance(label, str):
        return str(label), None

    match = re.match(r"(.*?)(?:\(([^()]+)\))?\s*$", label)
    if not match:
        return label.strip(), None

    name = match.group(1).strip()
    unit = match.group(2).strip() if match.group(2) else None
    return name, unit or None


def parse_table_d2(table_path: Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Parse the piecewise linear curves from Table D2."""
    curve_rows: List[Dict[str, float]] = []
    curve_meta: Dict[str, Dict[str, str]] = {}

    workbook = pd.ExcelFile(table_path)
    for sheet in workbook.sheet_names:
        if sheet in IGNORED_SHEETS_D2:
            continue
        if VULNERABILITY_SHEET_KEYWORD not in sheet:
            continue

        raw = workbook.parse(sheet_name=sheet)
        sheet_name = sheet.strip()

        if "ID number" not in raw.columns:
            raise KeyError(f"'ID number' column not found in sheet {sheet}")

        intensity_label = raw.at[3, "ID number"]
        intensity_name, intensity_unit = parse_intensity_label(str(intensity_label))

        data = raw.iloc[4:].copy()
        data = data.rename(columns={"ID number": "intensity"})
        data["intensity"] = pd.to_numeric(data["intensity"], errors="coerce")
        data = data[data["intensity"].notna()]

        curve_columns = [col for col in raw.columns if col != "ID number"]

        for original_col in curve_columns:
            curve_id = str(original_col).strip()
            series = pd.to_numeric(data[original_col], errors="coerce")
            mask = series.notna()
            if not mask.any():
                continue

            curve_meta.setdefault(
                curve_id,
                {
                    "curve_id": curve_id,
                    "sheet_name": sheet_name,
                    "intensity_axis": intensity_name,
                    "intensity_unit": intensity_unit,
                },
            )

            for order, (intensity, damage) in enumerate(
                zip(data.loc[mask, "intensity"], series.loc[mask]), start=0
            ):
                curve_rows.append(
                    {
                        "curve_id": curve_id,
                        "point_index": order,
                        "intensity": float(intensity),
                        "damage": float(damage),
                    }
                )

    if not curve_rows:
        raise RuntimeError("No curve points parsed from Table D2.")

    curves_df = pd.DataFrame(curve_rows)
    axis_df = pd.DataFrame(curve_meta.values())

    return curves_df, axis_df


def build_dataset(
    table_d1: Path, table_d2: Path, output_dir: Path, dry_run: bool = False
) -> None:
    """Generate the packaged dataset."""
    metadata = parse_table_d1(table_d1)
    curves, axis_meta = parse_table_d2(table_d2)

    metadata = metadata[metadata["curve_id"].isin(curves["curve_id"])].copy()
    metadata = metadata.merge(axis_meta, on="curve_id", how="left")
    metadata = metadata[~metadata["curve_type"].isin(EXCLUDED_CURVE_TYPES)].copy()
    curves = curves[curves["curve_id"].isin(metadata["curve_id"])].copy()
    metadata = metadata.sort_values("curve_id").reset_index(drop=True)
    curves = curves.sort_values(["curve_id", "point_index"]).reset_index(drop=True)

    if dry_run:
        print(metadata.head())
        print(curves.head())
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    metadata.to_csv(output_dir / "metadata.csv", index=False)
    curves.to_csv(output_dir / "curves.csv", index=False)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build the Snail damage curve library from the Nirandjan dataset."
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Do not write files; print head instead."
    )
    args = parser.parse_args()

    build_dataset(TABLE_D1, TABLE_D2, OUTPUT_DIR, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
