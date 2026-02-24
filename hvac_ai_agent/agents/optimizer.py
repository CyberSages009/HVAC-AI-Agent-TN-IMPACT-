from __future__ import annotations

import pandas as pd


def generate_recommendations(
    summary: dict,
    forecast_df: pd.DataFrame,
    anomalies_df: pd.DataFrame,
    full_df: pd.DataFrame,
    efficiency_threshold: float = 1.2,
) -> list[str]:
    """Generate action-oriented recommendations from forecast, efficiency, and anomalies."""
    recommendations: list[str] = []

    avg_kwh = float(summary.get("avg_kwh", 0.0))
    forecast_avg = float(forecast_df["forecast_kWh"].mean()) if not forecast_df.empty else avg_kwh
    recent_eff = float(full_df["iKW_TR"].tail(24).mean())
    anomaly_count = int(len(anomalies_df))

    if forecast_avg > avg_kwh * 1.1:
        recommendations.append(
            "Forecasted demand for the next 24 hours is high. Pre-cool during off-peak hours and adjust chiller staging to reduce peak draw."
        )
    else:
        recommendations.append(
            "Forecasted demand is near baseline. Maintain current scheduling and continue monitoring occupancy-driven spikes."
        )

    if recent_eff > efficiency_threshold:
        recommendations.append(
            f"Average iKW_TR over the last 24 hours is {recent_eff:.3f}, above threshold {efficiency_threshold:.2f}. Inspect condenser water loop and coil cleaning schedule."
        )
    else:
        recommendations.append(
            f"Recent iKW_TR is {recent_eff:.3f}, within target threshold {efficiency_threshold:.2f}. Keep current efficiency controls in place."
        )

    if anomaly_count > 0:
        recommendations.append(
            f"Detected {anomaly_count} energy anomalies. Review BAS logs for sensor drift, valve hunting, or abnormal occupancy events around flagged timestamps."
        )
    else:
        recommendations.append("No strong anomalies detected. Continue periodic fault checks and trend review.")

    return recommendations
