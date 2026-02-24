"""Analyzer agent for HVAC data quality checks and KPI extraction."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


ALIAS_MAP = {
    "time": "timestamp",
    "date": "timestamp",
    "datetime": "timestamp",
    "energy": "kwh",
    "energy_kwh": "kwh",
    "energy_consumption": "kwh",
    "kwh_consumption": "kwh",
    "ikwtr": "ikw_tr",
    "ikw_per_tr": "ikw_tr",
    "kw_per_tr": "ikw_tr",
    "ambient_temperature": "ambient_temp",
    "outdoor_temp": "ambient_temp",
    "outside_temp": "ambient_temp",
    "temperature": "ambient_temp",
    "occupancy": "load_profile",
    "load": "load_profile",
}

NUMERIC_COLUMNS = ["kwh", "ikw_tr", "ambient_temp", "load_profile"]


@dataclass
class AnalysisResult:
    cleaned_df: pd.DataFrame
    available_core_params: List[str]
    kpis: Dict[str, float]
    correlations: Dict[str, float]


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    renamed = {}
    for col in df.columns:
        normalized = col.strip().lower().replace(" ", "_").replace("-", "_")
        renamed[col] = ALIAS_MAP.get(normalized, normalized)
    return df.rename(columns=renamed)


def prepare_dataframe(raw_df: pd.DataFrame) -> pd.DataFrame:
    df = _normalize_columns(raw_df.copy())
    if "timestamp" not in df.columns:
        raise ValueError("Dataset must include a timestamp column.")

    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df.dropna(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)

    for col in NUMERIC_COLUMNS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


def analyze_data(raw_df: pd.DataFrame) -> AnalysisResult:
    df = prepare_dataframe(raw_df)

    available_core_params = [col for col in NUMERIC_COLUMNS if col in df.columns]
    if len(available_core_params) < 3:
        raise ValueError(
            "Dataset must contain at least 3 core parameters among: kwh, ikw_tr, ambient_temp, load_profile."
        )

    for col in available_core_params:
        df[col] = df[col].interpolate(limit_direction="both")
        df[col] = df[col].fillna(df[col].median())

    kpis = {
        "records": float(len(df)),
        "avg_kwh": float(df["kwh"].mean()) if "kwh" in df.columns else np.nan,
        "peak_kwh": float(df["kwh"].max()) if "kwh" in df.columns else np.nan,
        "avg_ikw_tr": float(df["ikw_tr"].mean()) if "ikw_tr" in df.columns else np.nan,
        "avg_ambient_temp": float(df["ambient_temp"].mean()) if "ambient_temp" in df.columns else np.nan,
    }

    correlation_series = (
        df[available_core_params].corr().get("kwh", pd.Series(dtype=float)).drop(labels=["kwh"], errors="ignore")
    )
    correlations = {key: float(val) for key, val in correlation_series.items()}

    return AnalysisResult(
        cleaned_df=df,
        available_core_params=available_core_params,
        kpis=kpis,
        correlations=correlations,
    )


def extract_operational_profile(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if "timestamp" not in df.columns or "kwh" not in df.columns:
        return pd.DataFrame(), pd.DataFrame()

    hourly = df.copy()
    hourly["hour"] = hourly["timestamp"].dt.hour
    hourly_profile = hourly.groupby("hour", as_index=False)["kwh"].mean()

    daily = df.copy()
    daily["day"] = daily["timestamp"].dt.date
    daily_profile = daily.groupby("day", as_index=False)["kwh"].sum()

    return hourly_profile, daily_profile
