from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression


def forecast_next_24(df: pd.DataFrame, target_col: str = "kWh", horizon: int = 24) -> pd.DataFrame:
    """Forecast next horizon steps using a simple linear regression trend model."""
    if df.empty:
        raise ValueError("Cannot forecast from an empty dataset.")

    series = df[target_col].astype(float).to_numpy()
    n_samples = len(series)

    x_train = np.arange(n_samples, dtype=float).reshape(-1, 1)
    model = LinearRegression()
    model.fit(x_train, series)

    x_future = np.arange(n_samples, n_samples + horizon, dtype=float).reshape(-1, 1)
    forecast_vals = model.predict(x_future)

    last_ts = pd.to_datetime(df["Timestamp"].iloc[-1])
    future_index = pd.date_range(start=last_ts + pd.Timedelta(hours=1), periods=horizon, freq="h")

    forecast_df = pd.DataFrame({"Timestamp": future_index, "forecast_kWh": forecast_vals})
    return forecast_df
