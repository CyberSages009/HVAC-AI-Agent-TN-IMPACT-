from __future__ import annotations

import pandas as pd


def detect_anomalies_zscore(df: pd.DataFrame, target_col: str = "kWh", threshold: float = 2.5) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Detect anomalies using absolute Z-score threshold."""
    working = df.copy()
    mean_val = working[target_col].mean()
    std_val = working[target_col].std(ddof=0)

    if std_val == 0 or pd.isna(std_val):
        working["z_score"] = 0.0
    else:
        working["z_score"] = (working[target_col] - mean_val) / std_val

    anomalies = working.loc[working["z_score"].abs() > threshold].copy()
    return anomalies, working
