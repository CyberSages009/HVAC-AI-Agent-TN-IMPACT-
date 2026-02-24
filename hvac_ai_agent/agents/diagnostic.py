"""Diagnostic agent for anomaly and degradation detection."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest


@dataclass
class DiagnosticResult:
    anomaly_df: pd.DataFrame
    summary: Dict[str, float]


def _zscore(series: pd.Series) -> pd.Series:
    std = series.std(ddof=0)
    if std == 0 or np.isnan(std):
        return pd.Series([0.0] * len(series), index=series.index)
    return (series - series.mean()) / std


def run_diagnostics(df: pd.DataFrame) -> DiagnosticResult:
    work_df = df.copy()
    feature_cols = [c for c in ["kwh", "ikw_tr", "ambient_temp", "load_profile"] if c in work_df.columns]

    z_anomaly = pd.Series(False, index=work_df.index)
    if "kwh" in work_df.columns:
        z_scores = _zscore(work_df["kwh"]) 
        z_anomaly = z_scores.abs() > 3.0

    iso_anomaly = pd.Series(False, index=work_df.index)
    if len(feature_cols) >= 2 and len(work_df) >= 20:
        model = IsolationForest(contamination=0.05, random_state=42)
        flags = model.fit_predict(work_df[feature_cols])
        iso_anomaly = pd.Series(flags == -1, index=work_df.index)

    work_df["is_anomaly"] = z_anomaly | iso_anomaly

    degradation_pct = 0.0
    if "ikw_tr" in work_df.columns and len(work_df) >= 10:
        baseline = work_df["ikw_tr"].iloc[: max(5, len(work_df) // 5)].mean()
        recent = work_df["ikw_tr"].iloc[-max(5, len(work_df) // 5):].mean()
        if baseline > 0:
            degradation_pct = float(((recent - baseline) / baseline) * 100.0)

    summary = {
        "anomaly_count": float(work_df["is_anomaly"].sum()),
        "anomaly_ratio_pct": float(work_df["is_anomaly"].mean() * 100.0),
        "efficiency_degradation_pct": degradation_pct,
    }

    return DiagnosticResult(anomaly_df=work_df, summary=summary)
