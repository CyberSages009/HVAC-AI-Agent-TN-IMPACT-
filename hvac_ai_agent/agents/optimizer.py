"""Optimization agent for actionable HVAC recommendations."""

from __future__ import annotations

from typing import Dict, List

import pandas as pd


def generate_recommendations(
    analysis_kpis: Dict[str, float],
    correlations: Dict[str, float],
    diagnostic_summary: Dict[str, float],
    forecast_df: pd.DataFrame,
) -> List[str]:
    recs: List[str] = []

    avg_ikw_tr = analysis_kpis.get("avg_ikw_tr", 0.0)
    if avg_ikw_tr and avg_ikw_tr > 1.05:
        recs.append(
            "Cooling efficiency (iKW-TR) is above target. Tune chiller water setpoint and clean condenser tubes to improve COP."
        )

    if diagnostic_summary.get("efficiency_degradation_pct", 0.0) > 8:
        recs.append(
            "Efficiency degradation trend detected. Schedule maintenance for filters, pumps, and chiller heat-exchange surfaces within 7 days."
        )

    if diagnostic_summary.get("anomaly_ratio_pct", 0.0) > 4:
        recs.append(
            "Frequent anomalies detected. Review BMS sensor calibration and alarm thresholds for temperature and load inputs."
        )

    temp_corr = correlations.get("ambient_temp", 0.0)
    if temp_corr > 0.5:
        recs.append(
            "High weather sensitivity observed. Apply weather-reset control and pre-cooling during lower tariff hours."
        )

    if not forecast_df.empty:
        peak = float(forecast_df["pred_kwh"].max())
        avg = float(forecast_df["pred_kwh"].mean())
        if peak > avg * 1.2:
            recs.append(
                "Forecast indicates upcoming demand spike. Pre-stage chillers and rebalance load across units before peak window."
            )

    if not recs:
        recs.append(
            "System performance appears stable. Keep current schedule and continue monitoring with weekly KPI review."
        )

    return recs
