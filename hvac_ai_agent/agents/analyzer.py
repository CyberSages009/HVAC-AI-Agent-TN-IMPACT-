from __future__ import annotations

from typing import Any

import pandas as pd

REQUIRED_COLUMNS = ["Timestamp", "kWh", "iKW_TR", "Temp", "Load"]


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Validate and clean raw HVAC data."""
    working = df.copy()
    working.columns = [str(c).strip() for c in working.columns]

    missing = [col for col in REQUIRED_COLUMNS if col not in working.columns]
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(missing)}")

    working["Timestamp"] = pd.to_datetime(working["Timestamp"], errors="coerce")
    numeric_cols = ["kWh", "iKW_TR", "Temp", "Load"]
    for col in numeric_cols:
        working[col] = pd.to_numeric(working[col], errors="coerce")

    working = working.dropna(subset=REQUIRED_COLUMNS).sort_values("Timestamp").reset_index(drop=True)
    return working


def compute_summary_statistics(df: pd.DataFrame) -> dict[str, Any]:
    """Compute key summary metrics used by downstream agents and UI KPIs."""
    return {
        "rows": int(len(df)),
        "total_kwh": float(df["kWh"].sum()),
        "avg_kwh": float(df["kWh"].mean()),
        "peak_kwh": float(df["kWh"].max()),
        "avg_ikw_tr": float(df["iKW_TR"].mean()),
        "avg_temp": float(df["Temp"].mean()),
        "avg_load": float(df["Load"].mean()),
        "start": df["Timestamp"].min(),
        "end": df["Timestamp"].max(),
    }


def compute_load_kwh_correlation(df: pd.DataFrame) -> float:
    """Correlation between building load and energy usage."""
    return float(df["Load"].corr(df["kWh"]))


def run_analysis(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any], float]:
    cleaned = clean_data(df)
    summary = compute_summary_statistics(cleaned)
    correlation = compute_load_kwh_correlation(cleaned)
    return cleaned, summary, correlation
