#!/usr/bin/env python3
"""
Generate a clean Markdown table from results/cot_sensitivity/cot_sensitivity_summary_6.csv.

The output includes per (model, dataset):
  - Baseline change rate
  - Ellipses change rate and delta vs baseline
  - Incorrect CoT change rate and delta vs baseline
  - Parsed rate (minimum across modification types for that dataset)

Writes Markdown to results/cot_summary_6.md (by default).
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Tuple

import pandas as pd


def format_pct(value: float) -> str:
    if pd.isna(value):
        return "—"
    return f"{value * 100:.0f}%"


def build_table(df: pd.DataFrame) -> pd.DataFrame:
    required_cols = {
        "model_name",
        "dataset_name",
        "modification_type",
        "change_rate",
        "baseline_change_rate",
        "parse_rate",
    }
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    # For each (model, dataset), pivot modification types to columns
    # and aggregate stable fields.
    grouped = []
    for (model_name, dataset_name), g in df.groupby(["model_name", "dataset_name"], sort=False):
        baseline = g["baseline_change_rate"].iloc[0]
        parsed_min = g["parse_rate"].min()

        # Fetch change rates by type
        ellipses_row = g.loc[g["modification_type"] == "ellipses"]
        incorrect_row = g.loc[g["modification_type"] == "incorrect_cot"]

        ellipses = ellipses_row["change_rate"].iloc[0] if not ellipses_row.empty else float("nan")
        incorrect = incorrect_row["change_rate"].iloc[0] if not incorrect_row.empty else float("nan")

        grouped.append(
            {
                "Model": model_name,
                "Dataset": dataset_name,
                "Baseline": baseline,
                "Ellipses": ellipses,
                "Δ Ellipses": ellipses - baseline if pd.notna(ellipses) else float("nan"),
                "Incorrect CoT": incorrect,
                "Δ Incorrect": incorrect - baseline if pd.notna(incorrect) else float("nan"),
                "Parsed": parsed_min,
            }
        )

    table = pd.DataFrame(grouped)

    # Sort for readability: by Model then Dataset
    model_order = table["Model"].drop_duplicates().tolist()
    dataset_order = [
        "anachronisms",
        "logical_deduction",
        "social_chemistry",
        "sports_understanding",
    ]
    table["Model"] = pd.Categorical(table["Model"], categories=model_order, ordered=True)
    table["Dataset"] = pd.Categorical(table["Dataset"], categories=dataset_order, ordered=True)
    table = table.sort_values(["Model", "Dataset"]).reset_index(drop=True)

    # Format percentages
    for col in ["Baseline", "Ellipses", "Δ Ellipses", "Incorrect CoT", "Δ Incorrect", "Parsed"]:
        table[col] = table[col].map(format_pct)

    return table


def to_markdown(table: pd.DataFrame) -> str:
    # Build a compact markdown table with aligned columns
    return table.to_markdown(index=False)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--csv",
        type=Path,
        default=Path("results/cot_sensitivity/cot_sensitivity_summary_6.csv"),
        help="Path to results/cot_sensitivity/cot_sensitivity_summary_6.csv",
    )
    parser.add_argument(
        "--outdir",
        type=Path,
        default=Path("results"),
        help="Directory to write outputs",
    )
    args = parser.parse_args()

    df = pd.read_csv(args.csv)
    table = build_table(df)

    args.outdir.mkdir(parents=True, exist_ok=True)
    md_path = args.outdir / "cot_summary_6.md"
    with md_path.open("w", encoding="utf-8") as f:
        f.write(to_markdown(table))

    print(f"Wrote markdown table to: {md_path}")


if __name__ == "__main__":
    main()


